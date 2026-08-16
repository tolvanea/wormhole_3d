# Ellis wormhole fly-through renderer

A Rust + wgpu ray tracer that renders a camera flight *through* an Ellis
wormhole, either live in a window or offline to PNG frames. There is no time
and no gravity: light rays are simply geodesics of the static 3D spatial
metric

    ds² = dl² + (l² + a²)(dθ² + sin²θ dφ²)

whose embedded throat profile is the catenoid r = a·cosh(z/a). The coordinate
l ∈ (−∞, +∞) is proper radial distance; l > 0 is "universe A", l < 0 is
"universe B", and l = 0 is the throat.

The whole per-pixel integration runs as a wgpu compute shader
(`src/tracer.wgsl`); the CPU side (`src/scene.rs`) only solves the camera path,
once per flight, and hands the GPU a 176-byte uniform block per frame.

## How it works
See [math.md](/math.md) for more.

## Build & run

Live, in a window:

    cargo run --release

Space pauses, left/right step a frame at a time, Home rewinds, Escape quits.
Defaults to 1280×720 with no supersampling, which is vsync-bound on a discrete
GPU.

Offline, to disk:

    cargo run --release -- --render

Outputs `out/frame_####.png` (168 frames, 1920×1080, 2×2 supersampled) and an
animated `out/wormhole.gif`. Both modes trace with the same shader into the
same texture — the window blits it through a deliberately non-sRGB surface so
that what you see is byte-for-byte what gets written.

Flags: `--width`, `--height`, `--ssaa`, and `--model FILE.gltf` to fly
something other than the helmet (it is centred and scaled to `MODEL_R`
automatically). The scene constants live at the top of `src/scene.rs`:

| | |
| --- | --- |
| `A`, `FOV_DEG` | throat radius, field of view |
| `INTRO_FRAMES` / `FRAMES` | frames for the approach and for the crossing |
| `L_INTRO_START` | how far out the approach begins |
| `INTRO_DIST_START` | how close the camera starts to the model, along the flight axis |
| `INTRO_RISE` / `INTRO_AIM` | how high it stands, and how high it looks; equal values give a level shot |
| `L_START` / `L_END` | where the crossing begins and ends |
| `LOOP_SWEEP` | how far the path loops around the throat while crossing |
| `ORBIT_TURNS` / `ORBIT_SPREAD` | turns the camera makes around the model, and how tightly they cluster at the wormhole |
| `ORBIT_DIST` / `LOOK_GRIP` | how far the camera stands off, and how firmly it holds the model once it swings to the side |
| `THROAT_DWELL` | extra frames spent on the crossing, beyond what even motion already gives it |
| `END_RAMP` | how much of the flight is spent getting up to speed and back down |
| `MODEL_R` / `MODEL_LEAD` / `MODEL_RISE` | the chase model's size, lead, and height above the flight plane |
| `MODEL_TURNS` | turns the model makes about its own axis; 0 keeps it still |

For reference, the full 3000-frame 1080p 4×4 render takes about 30min end to end
on an RX 7900

To make an mp4 instead of the gif:

```
ffmpeg -framerate 60 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p -crf 18 wormhole.mp4
```

## Credits
* Claude for the code
* Poninnahka for [SFM model](https://ponysfm.com/poninnahka-s-female-pony-models) on which Pony Love's model is built upon
* Sindroom for pony [space suit model](https://ponysfm.com/dl-space-suit)
* Salaryman for writing a [guide](https://sfmlab.com/tutorials/view/be13f1f3-a517-4c38-aee6-47cb116add39/) to port SFM models to blender
* [SourceIO](https://github.com/REDxEYE/SourceIO) for porting SFM models to blender.
* EStories for [ponyville background](https://www.deviantart.com/estories/art/Ponyville-Large-1163632729)
