# Guidance for Claude

## Dependencies

Use external crates freely whenever fit to task. Avoid using crates with a low
amount of downloads (e.g. clearly less than 100 000 downloads).

## Project

An Ellis wormhole fly-through ray tracer. Rays are geodesics of a static 3D
spatial metric, integrated with RK4; the per-pixel work runs as a wgpu compute
shader. See `README.md` for the physics and the derivations.

| Path | Role |
| --- | --- |
| `src/scene.rs` | Geodesic math, flight path, camera solving. f64, once per flight. |
| `src/tracer.wgsl` | The tracer: RK4, cube-map sky, mesh BVH. One thread per pixel. |
| `src/model.rs` | glTF load, scaling, mip chains, binned-SAH BVH. |
| `src/cubemap.rs` | 4x3 cross image to six cube faces. |
| `src/gpu.rs` | wgpu setup, bind groups, per-frame uniforms, readback. |
| `src/main.rs` | Args, live winit loop, headless PNG/GIF export. |

## Build & run

    cargo run --release              # live window
    cargo run --release -- --render  # write out/frame_###.png and the GIF

Both modes trace with the same shader into the same texture, so they must stay
pixel-identical: the window blits through a deliberately non-sRGB surface
because the shader has already tone mapped and gamma encoded.

## Conventions

- Comments explain *why*, especially where the curved-space geometry makes an
  obvious-looking approach wrong. Match the density of the surrounding code.
- `scene.rs` and `tracer.wgsl` deliberately duplicate the RK4 integrator: the
  path solving needs f64, the ray tracing needs speed. Keep them in step.
- Constants that shape the flight live at the top of `scene.rs`. Prefer adding
  one there over hard-coding a number further down.
- The flight is one continuous curve; the "two acts" are just ramps keyed to
  `l`, not separate code paths. Frames are placed by walking that curve and
  measuring view change, so anything that changes `pose_at` changes the timing
  too — check the startup lines rather than assuming.
- Rays composite front-to-back: the march carries a colour and a remaining
  transmittance rather than returning on the first hit. Anything new that
  terminates a ray early has to account for `throughput`, or transparent
  surfaces silently stop working.
- Open files through `assets::read` / `assets::open_image` rather than `std::fs`
  or `image::open` directly. They turn a missing file into a message naming the
  asset, the expanded path, how far down it exists, and where the real file is
  if it is nearby — which is the failure users actually hit, since every asset
  is named relative to the working directory.
