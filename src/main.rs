//! Ellis wormhole fly-through renderer, on the GPU.
//!
//! Two ways to run it, both tracing the same flight with the same shader:
//!
//!     cargo run --release              -- live window, fly the path in real time
//!     cargo run --release -- --render  -- write out/frame_###.png and the GIF
//!
//! See `scene.rs` for the geometry and the camera path, `tracer.wgsl` for the
//! per-ray integration, and `model.rs` for the glTF chase model.

mod assets;
mod cubemap;
mod gpu;
mod model;
mod scene;

use anyhow::{Context, Result};
use scene::Shot;
use std::path::PathBuf;
use std::sync::Arc;

const SKY_A: &str = "cubemap/milky_way_crotch.png"; // sky seen at l -> +inf
const SKY_B: &str = "cubemap/ponyville_by_estories_mirrored.png"; // sky seen at l -> -inf
const MODEL: &str = "gltf_models/space_pony/space_pony.gltf";

const GIF_EVERY: usize = 1; // put every n-th frame into the GIF
const GIF_DELAY_MS: u32 = 50; // 20 fps

struct Options {
    headless: bool,
    width: u32,
    height: u32,
    ssaa: u32,
    model: PathBuf,
}

fn parse_args() -> Result<Options, String> {
    let mut o = Options {
        headless: false,
        width: 0,
        height: 0,
        // The live view has a frame budget; the offline render does not.
        ssaa: 0,
        model: PathBuf::from(MODEL),
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} needs a value"))?
                .parse::<u32>()
                .map_err(|e| format!("bad value for {name}: {e}"))
        };
        match a.as_str() {
            "--render" => o.headless = true,
            "--width" => o.width = val("--width")?,
            "--height" => o.height = val("--height")?,
            "--ssaa" => o.ssaa = val("--ssaa")?,
            "--model" => {
                o.model = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--model needs a path".to_string())?,
                )
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: wormhole [--render] [--width N] [--height N] [--ssaa N] \
                     [--model FILE.gltf]\n\
                     \n\
                       --render   write out/frame_###.png and out/wormhole.gif instead\n\
                                  of opening a window\n\
                       --ssaa N   N x N rays per pixel (default 1 live, 2 offline)"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if o.width == 0 {
        o.width = if o.headless { 1920 } else { 1280 };
    }
    if o.height == 0 {
        o.height = if o.headless { 1080 } else { 720 };
    }
    if o.ssaa == 0 {
        o.ssaa = if o.headless { 2 } else { 1 };
    }
    Ok(o)
}

fn main() {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("wormhole: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(opts) {
        eprintln!("wormhole: {e}");
        // anyhow keeps the chain of `with_context` calls; print the causes so
        // "cannot open the base colour texture" still says which model asked
        // for it.
        for cause in e.chain().skip(1) {
            eprintln!("\ncaused by: {cause}");
        }
        std::process::exit(1);
    }
}

fn run(opts: Options) -> Result<()> {
    // Load everything before touching the GPU, so a missing asset fails fast
    // and with a message about the asset rather than about a bind group.
    let model = model::Model::load(&opts.model)?;
    let sky_a = cubemap::Sky::load(
        std::path::Path::new(SKY_A),
        "the sky cube map for universe A",
    )?;
    let sky_b = cubemap::Sky::load(
        std::path::Path::new(SKY_B),
        "the sky cube map for universe B",
    )?;
    eprintln!(
        "cube maps: {SKY_A}, {SKY_B}  ({}x{} per face)",
        sky_a.size, sky_a.size
    );

    let shots = scene::flight_plan();
    scene::report_continuity(&shots);

    if opts.headless {
        render_headless(opts, &model, &sky_a, &sky_b, &shots)
    } else {
        run_live(opts, model, sky_a, sky_b, shots)
    }
}

// -------------------------------- headless ---------------------------------

fn render_headless(
    opts: Options,
    model: &model::Model,
    sky_a: &cubemap::Sky,
    sky_b: &cubemap::Sky,
    shots: &[Shot],
) -> Result<()> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame, RgbImage, Rgba, RgbaImage};

    assets::create_dir(std::path::Path::new("out"), "the output directory")?;
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let mut r = pollster::block_on(gpu::Renderer::new(
        &instance,
        None,
        model,
        sky_a,
        sky_b,
        opts.ssaa,
        scene::MAX_STEPS,
    ))?;
    r.resize(opts.width, opts.height);

    let gif_file = assets::create_file(std::path::Path::new("out/wormhole.gif"), "the GIF")?;
    let mut gif = GifEncoder::new_with_speed(gif_file, 10);
    gif.set_repeat(Repeat::Infinite)?;

    let frames = shots.len();
    for (f, shot) in shots.iter().enumerate() {
        let start = std::time::Instant::now();
        r.trace(shot);
        let rgba = r.read_back();

        let mut img = RgbImage::new(opts.width, opts.height);
        for (px, out) in rgba.chunks_exact(4).zip(img.pixels_mut()) {
            *out = image::Rgb([px[0], px[1], px[2]]);
        }
        let path = format!("out/frame_{:04}.png", f);
        img.save(&path)
            .with_context(|| format!("cannot write frame {} to {path}", f + 1))?;
        eprintln!(
            "frame {:3}/{}  l = {:+.3}  psi = {:+.1} deg  look-vs-axis = {:.1} deg  \
             model at l = {:+.3}  ({:.2} s)",
            f + 1,
            frames,
            shot.cam.l,
            shot.cam.u.y.atan2(shot.cam.u.x).to_degrees(),
            (-shot.cam.fwd.dot(&shot.cam.u)).acos().to_degrees(),
            shot.body.l,
            start.elapsed().as_secs_f64()
        );

        // GIF_EVERY is a knob, not a constant to fold away: it defaults to 1
        // (every frame) but exists so a long flight can be thinned down.
        #[allow(clippy::modulo_one)]
        if f % GIF_EVERY == 0 {
            let frame_img: RgbaImage = RgbaImage::from_fn(opts.width, opts.height, |x, y| {
                let p = img.get_pixel(x, y);
                Rgba([p[0], p[1], p[2], 255])
            });
            let frame =
                Frame::from_parts(frame_img, 0, 0, Delay::from_numer_denom_ms(GIF_DELAY_MS, 1));
            gif.encode_frame(frame)?;
        }
    }
    eprintln!("done: out/frame_***.png and out/wormhole.gif");
    Ok(())
}

// ---------------------------------- live -----------------------------------

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

struct Live {
    instance: wgpu::Instance,
    model: model::Model,
    sky_a: cubemap::Sky,
    sky_b: cubemap::Sky,
    shots: Vec<Shot>,
    ssaa: u32,
    size: (u32, u32),

    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    surface_format: Option<wgpu::TextureFormat>,
    renderer: Option<gpu::Renderer>,

    frame: usize,
    playing: bool,
    last_report: std::time::Instant,
    frames_since_report: u32,
}

fn run_live(
    opts: Options,
    model: model::Model,
    sky_a: cubemap::Sky,
    sky_b: cubemap::Sky,
    shots: Vec<Shot>,
) -> Result<()> {
    let mut app = Live {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env()),
        model,
        sky_a,
        sky_b,
        shots,
        ssaa: opts.ssaa,
        size: (opts.width, opts.height),
        window: None,
        surface: None,
        surface_format: None,
        renderer: None,
        frame: 0,
        playing: true,
        last_report: std::time::Instant::now(),
        frames_since_report: 0,
    };
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    eprintln!(
        "live: space pauses, left/right step frames, home rewinds, escape quits \
         ({} frames)",
        app.shots.len()
    );
    event_loop.run_app(&mut app)?;
    Ok(())
}

impl ApplicationHandler for Live {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ellis wormhole")
            .with_inner_size(winit::dpi::PhysicalSize::new(self.size.0, self.size.1));
        let window = Arc::new(el.create_window(attrs).expect("cannot create a window"));
        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("cannot create a surface");
        let renderer = pollster::block_on(gpu::Renderer::new(
            &self.instance,
            Some(&surface),
            &self.model,
            &self.sky_a,
            &self.sky_b,
            self.ssaa,
            scene::MAX_STEPS,
        ))
        .expect("cannot start the renderer");

        self.window = Some(window);
        self.surface = Some(surface);
        self.renderer = Some(renderer);
        self.reconfigure();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                self.size = (size.width.max(1), size.height.max(1));
                self.reconfigure();
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => {
                let n = self.shots.len();
                match event.logical_key {
                    Key::Named(NamedKey::Escape) => el.exit(),
                    Key::Named(NamedKey::Space) => self.playing = !self.playing,
                    Key::Named(NamedKey::ArrowRight) => {
                        self.playing = false;
                        self.frame = (self.frame + 1) % n;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        self.playing = false;
                        self.frame = (self.frame + n - 1) % n;
                    }
                    Key::Named(NamedKey::Home) => self.frame = 0,
                    _ => {}
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => self.draw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl Live {
    fn reconfigure(&mut self) {
        let (Some(surface), Some(renderer)) = (&self.surface, &mut self.renderer) else {
            return;
        };
        let caps = surface.get_capabilities(renderer.adapter());
        // The compute pass already gamma-encoded, so present through a plain
        // unorm format: an sRGB one would encode a second time and wash the
        // image out. Every desktop backend offers the unorm twin.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        self.surface_format = Some(format);
        let (w, h) = self.size;
        surface.configure(
            &renderer.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: w,
                height: h,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: caps.alpha_modes[0],
                color_space: wgpu::SurfaceColorSpace::Auto,
                view_formats: vec![],
            },
        );
        renderer.resize(w, h);
    }

    fn draw(&mut self) {
        let (Some(surface), Some(renderer), Some(format)) =
            (&self.surface, &mut self.renderer, self.surface_format)
        else {
            return;
        };
        use wgpu::CurrentSurfaceTexture as Cst;
        let frame = match surface.get_current_texture() {
            Cst::Success(f) | Cst::Suboptimal(f) => f,
            // Resized or gone stale between the configure and here; rebuild the
            // swapchain and let the next redraw pick it up.
            Cst::Outdated | Cst::Lost => {
                self.reconfigure();
                return;
            }
            // Nothing to draw into, or nothing worth drawing: skip the frame.
            Cst::Timeout | Cst::Occluded => return,
            other => {
                eprintln!("wormhole: surface unavailable: {other:?}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        renderer.trace(&self.shots[self.frame]);
        renderer.blit(format, &view);
        renderer.queue.present(frame);

        if self.playing {
            self.frame = (self.frame + 1) % self.shots.len();
        }
        self.frames_since_report += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed.as_secs_f64() >= 2.0 {
            eprintln!(
                "live: {:.1} fps at {}x{} (ssaa {})",
                self.frames_since_report as f64 / elapsed.as_secs_f64(),
                self.size.0,
                self.size.1,
                self.ssaa
            );
            self.frames_since_report = 0;
            self.last_report = std::time::Instant::now();
        }
    }
}
