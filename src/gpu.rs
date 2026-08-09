//! wgpu setup: upload the scene once, then one compute dispatch per frame.
//!
//! The mesh, its BVH, the base colour textures and the two skies are static
//! for the whole flight, so they live in one bind group that is built at
//! startup and never touched again. Everything that changes per frame is a
//! 176-byte uniform block.
//!
//! The compute pass writes tone-mapped, gamma-encoded bytes into an
//! `rgba8unorm` storage texture. That texture is the single source of truth
//! for both outputs: the window blits it through a non-sRGB surface (so the
//! hardware does not gamma-encode a second time) and the headless renderer
//! copies it straight to a buffer and out to a PNG. The two modes therefore
//! cannot drift apart.

use crate::cubemap::Sky;
use crate::model::Model;
use crate::scene::{self, Shot};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
struct Uniforms {
    cam_u: [f32; 3],
    cam_l: f32,
    cam_fwd: [f32; 3],
    half_fov: f32,
    cam_up: [f32; 3],
    aspect: f32,
    cam_right: [f32; 3],
    sky_gain: f32,

    body_b1: [f32; 3],
    body_l: f32,
    body_b2: [f32; 3],
    body_r: f32,
    body_b3: [f32; 3],
    spin_c: f32,
    light: [f32; 3],
    spin_s: f32,

    width: u32,
    height: u32,
    ssaa: u32,
    max_steps: u32,

    a: f32,
    h0: f32,
    l_escape: f32,
    pixel_angle: f32,

    tex_size: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

/// The render target and everything sized to it. Rebuilt when the window
/// resizes; in headless mode it is created once.
struct Target {
    width: u32,
    height: u32,
    texture: wgpu::Texture,
    compute_bg: wgpu::BindGroup,
    blit_bg: wgpu::BindGroup,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    uniform_buf: wgpu::Buffer,
    compute_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    blit_layout: wgpu::BindGroupLayout,
    blit_module: wgpu::ShaderModule,
    blit_pipeline: Option<(wgpu::TextureFormat, wgpu::RenderPipeline)>,
    blit_sampler: wgpu::Sampler,
    /// Everything static: mesh, BVH, textures, skies.
    scene_entries: SceneResources,
    target: Option<Target>,
    ssaa: u32,
    max_steps: u32,
    tex_size: f32,
}

struct SceneResources {
    tris: wgpu::Buffer,
    nodes: wgpu::Buffer,
    materials: wgpu::Buffer,
    model_view: wgpu::TextureView,
    model_samp: wgpu::Sampler,
    sky_a: wgpu::TextureView,
    sky_b: wgpu::TextureView,
    sky_samp: wgpu::Sampler,
}

impl Renderer {
    /// Pick an adapter, upload the scene and build the pipelines. Pass the
    /// surface when there is a window, so the adapter chosen can present to it.
    pub async fn new(
        instance: &wgpu::Instance,
        surface: Option<&wgpu::Surface<'static>>,
        model: &Model,
        sky_a: &Sky,
        sky_b: &Sky,
        ssaa: u32,
        max_steps: u32,
    ) -> anyhow::Result<Renderer> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                ..Default::default()
            })
            .await?;
        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        // The mesh and its BVH are far past the default 128 MB storage buffer
        // limit's smaller cousins, so ask for what the adapter actually has.
        let mut limits = wgpu::Limits::default();
        let adapter_limits = adapter.limits();
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_texture_array_layers = adapter_limits.max_texture_array_layers.max(6);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("wormhole"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await?;

        // Surface a validation error rather than letting a bad dispatch turn
        // into a silently black frame.
        device.on_uncaptured_error(std::sync::Arc::new(|e| panic!("wgpu error: {e}")));

        let tris = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("triangles"),
            contents: bytemuck::cast_slice(&model.tris),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let nodes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bvh"),
            contents: bytemuck::cast_slice(&model.nodes),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let materials = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: bytemuck::cast_slice(&model.materials),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let model_view = upload_model_textures(&device, &queue, model);
        let model_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("model sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let sky_a_view = upload_sky(&device, &queue, sky_a, "sky A");
        let sky_b_view = upload_sky(&device, &queue, sky_b, "sky B");
        let sky_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compute_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tracer layout"),
            entries: &[
                uniform_entry(0),
                storage_texture_entry(1),
                storage_buffer_entry(2),
                storage_buffer_entry(3),
                texture_entry(4, wgpu::TextureViewDimension::D2Array),
                sampler_entry(5),
                texture_entry(6, wgpu::TextureViewDimension::Cube),
                texture_entry(7, wgpu::TextureViewDimension::Cube),
                sampler_entry(8),
                storage_buffer_entry(9),
            ],
        });
        let tracer_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tracer"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tracer.wgsl").into()),
        });
        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tracer pipeline layout"),
                bind_group_layouts: &[Some(&compute_layout)],
                immediate_size: 0,
            });
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tracer"),
            layout: Some(&compute_pipeline_layout),
            module: &tracer_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit layout"),
            entries: &[
                texture_entry(0, wgpu::TextureViewDimension::D2),
                sampler_entry(1),
            ],
        });
        let blit_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blit sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Renderer {
            device,
            queue,
            adapter,
            uniform_buf,
            compute_layout,
            compute_pipeline,
            blit_layout,
            blit_module,
            blit_pipeline: None,
            blit_sampler,
            scene_entries: SceneResources {
                tris,
                nodes,
                materials,
                model_view,
                model_samp,
                sky_a: sky_a_view,
                sky_b: sky_b_view,
                sky_samp,
            },
            target: None,
            ssaa,
            max_steps,
            tex_size: model.tex_size as f32,
        })
    }

    /// The adapter the device came from, for querying surface capabilities.
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn size(&self) -> (u32, u32) {
        self.target.as_ref().map_or((0, 0), |t| (t.width, t.height))
    }

    /// Create or re-create the render target. Cheap enough to call every frame;
    /// it only does work when the size actually changed.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (width, height) = (width.max(1), height.max(1));
        if let Some(t) = &self.target
            && t.width == width
            && t.height == height
        {
            return;
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("traced image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let s = &self.scene_entries;
        let compute_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tracer bind group"),
            layout: &self.compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: s.tris.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: s.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&s.model_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&s.model_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&s.sky_a),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&s.sky_b),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&s.sky_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: s.materials.as_entire_binding(),
                },
            ],
        });
        let blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });
        self.target = Some(Target {
            width,
            height,
            texture,
            compute_bg,
            blit_bg,
        });
    }

    fn uniforms_for(&self, shot: &Shot, width: u32, height: u32) -> Uniforms {
        let cam = shot.cam;
        let body = shot.body;
        let right = cam.fwd.cross(&cam.up);
        let (b1, b2, b3) = body.frame();
        let light = {
            let l = scene::V3::new(
                scene::MODEL_LIGHT[0],
                scene::MODEL_LIGHT[1],
                scene::MODEL_LIGHT[2],
            )
            .normalize();
            [l.x as f32, l.y as f32, l.z as f32]
        };
        let half_fov = (scene::FOV_DEG.to_radians() / 2.0).tan();
        let ssaa = self.ssaa.max(1);
        Uniforms {
            cam_u: v(cam.u),
            cam_l: cam.l as f32,
            cam_fwd: v(cam.fwd),
            half_fov: half_fov as f32,
            cam_up: v(cam.up),
            aspect: (height as f64 / width as f64) as f32,
            cam_right: v(right),
            sky_gain: scene::SKY_GAIN as f32,

            body_b1: v(b1),
            body_l: body.l as f32,
            body_b2: v(b2),
            body_r: body.r as f32,
            body_b3: v(b3),
            spin_c: body.spin.cos() as f32,
            light,
            spin_s: body.spin.sin() as f32,

            width,
            height,
            ssaa,
            max_steps: self.max_steps,

            a: scene::A as f32,
            h0: scene::H0 as f32,
            l_escape: scene::L_ESCAPE as f32,
            // Angular size of one supersample, which is what sets how wide a
            // ray's cone has spread by the time it reaches the model.
            pixel_angle: (2.0 * half_fov / (width * ssaa) as f64) as f32,

            tex_size: self.tex_size,
            pad0: 0.0,
            pad1: 0.0,
            pad2: 0.0,
        }
    }

    /// Trace one frame into the render target.
    pub fn trace(&mut self, shot: &Shot) {
        let (width, height) = self.size();
        let u = self.uniforms_for(shot, width, height);
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));

        let target = self.target.as_ref().expect("resize() before trace()");
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("trace"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &target.compute_bg, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Draw the traced image to a surface texture.
    pub fn blit(&mut self, format: wgpu::TextureFormat, dest: &wgpu::TextureView) {
        if self.blit_pipeline.as_ref().map(|(f, _)| *f) != Some(format) {
            let layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("blit pipeline layout"),
                    bind_group_layouts: &[Some(&self.blit_layout)],
                    immediate_size: 0,
                });
            let pipeline = self
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("blit"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &self.blit_module,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &self.blit_module,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(format.into())],
                    }),
                    multiview_mask: None,
                    cache: None,
                });
            self.blit_pipeline = Some((format, pipeline));
        }
        let (_, pipeline) = self.blit_pipeline.as_ref().unwrap();
        let target = self.target.as_ref().expect("resize() before blit()");

        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit"),
            });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: dest,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &target.blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
    }

    /// Read the traced image back as tightly packed RGBA8 rows.
    pub fn read_back(&mut self) -> Vec<u8> {
        let target = self.target.as_ref().expect("resize() before read_back()");
        let (width, height) = (target.width, target.height);
        // copy_texture_to_buffer wants rows aligned to 256 bytes, which a plain
        // 4-byte-per-texel row only happens to satisfy at some widths.
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit([enc.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device lost while reading back the frame");
        rx.recv().unwrap().expect("failed to map the readback buffer");

        let data = slice
            .get_mapped_range()
            .expect("readback buffer did not map");
        let mut out = Vec::with_capacity((unpadded * height) as usize);
        for row in 0..height {
            let start = (row * padded) as usize;
            out.extend_from_slice(&data[start..start + unpadded as usize]);
        }
        drop(data);
        buffer.unmap();
        out
    }
}

fn v(x: scene::V3) -> [f32; 3] {
    [x.x as f32, x.y as f32, x.z as f32]
}

// ------------------------------ texture upload ------------------------------

fn upload_model_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    model: &Model,
) -> wgpu::TextureView {
    let size = model.tex_size;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("model base colour"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: model.textures.len() as u32,
        },
        mip_level_count: model.mip_levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // sRGB, so the sampler hands the shader linear light -- the same thing
        // the CPU tracer did by hand when it loaded the cube maps.
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, tex) in model.textures.iter().enumerate() {
        for (level, data) in tex.mips.iter().enumerate() {
            let w = (size >> level).max(1);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 4),
                    rows_per_image: Some(w),
                },
                wgpu::Extent3d {
                    width: w,
                    height: w,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn upload_sky(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sky: &Sky,
    label: &str,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: sky.size,
            height: sky.size,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (face, data) in sky.faces.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(sky.size * 4),
                rows_per_image: Some(sky.size),
            },
            wgpu::Extent3d {
                width: sky.size,
                height: sky.size,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

// --------------------------- bind group boilerplate -------------------------

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_buffer_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, dim: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: dim,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}
