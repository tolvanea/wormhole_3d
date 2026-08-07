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

1. **One geodesic per ray.** For every (sub)pixel the camera emits a unit
   direction. By spherical symmetry each geodesic lies in a plane, so the ray
   is reduced to the planar ODE system

       dl/ds     = v_l
       dφ/ds     = v_φ
       dv_l/ds   = l · v_φ²
       dv_φ/ds   = −2l/(l² + a²) · v_l · v_φ

   integrated with classic RK4 (step size ∝ √(l² + a²), i.e. fine near the
   throat, coarse far away).

2. **Escape and sky lookup.** When |l| exceeds an escape radius while moving
   outward, the asymptotic straight-line direction is reconstructed by
   rotating the ray's plane basis by the swept angle φ, and a cube map is
   sampled — `cubemap/cubemap_1.jpg` for universe A, `cubemap/cubemap_2.png`
   for universe B. Both are 4×3 cross-form images with the cross on the left
   (vertical arm in column 1, −Z trailing off to the right); they are cut into
   six faces and uploaded as real cube textures in `sRGB` format, so the face
   selection, the bilinear filtering and the linearization all happen in
   hardware. (The face order and projection WebGPU uses turn out to be exactly
   the convention the old CPU lookup implemented by hand.)
   Rays that exhaust the step budget are asymptotically winding
   on the throat's photon ring and are painted black — that is the thin dark
   ring you see at the critical impact parameter.

3. **Solid geometry: the chase model.** A glTF model — Khronos's FlightHelmet,
   94,722 triangles across six primitives — flies the same path 2.5a ahead of
   the camera (and 0.7a above the flight plane, so it does not eclipse the
   mouth), turning steadily about its own z axis. Hitting it needs the ray's 3D
   position, which the planar reduction still has: after sweeping φ the ray
   sits at angular position u(φ) = p̂ cos φ + t̂ sin φ, so each RK4 step gives a
   manifold point (l, u).

   There is no global Cartesian chart to test against — the 3D wormhole embeds
   in R⁴, not R³ — but none is needed. The model is small compared with a, so
   its neighbourhood is flat and the offset of (l, u) from the centre (l_c, u_c)
   is, to first order,

       O = (l − l_c)·u_c + r_mid·σ·ê,   r_mid = r((l + l_c)/2)

   where σ = ∠(u, u_c) and ê is the unit vector from u_c towards u across the
   sphere. Note σ, the arc, and not the chord |u − u_c(u·u_c)|: the chord
   vanishes at the antipode of the centre as well as at the centre itself,
   which would read as a second, spurious copy of the model for rays that have
   wound half way round.

   The two terms are orthogonal and |O| is the proper distance. Mapping *both*
   endpoints of an RK4 step through it, projecting onto the model's own
   orthonormal frame and winding back by the spin, turns the step into a
   straight segment in the model's local box — and a segment is something a
   BVH can answer. So the whole intersection test is a segment-vs-BVH query,
   valid right through the throat.

   Almost every ray is rejected by a single comparison on l; near the model the
   step is clamped to a quarter of its bounding radius, which keeps the chord
   short enough that replacing the curved step by a straight one stays
   accurate. Unlike the sphere's inside/outside test, the segment query needs
   no bisection afterwards: it lands exactly on the triangle, with barycentric
   coordinates for the normal and the UV.

   The BVH is a binned-SAH build (12 bins, ≤4 triangles per leaf, 65,457 nodes)
   flattened depth-first with children allocated in pairs, so one index per
   interior node locates both; the shader walks it iteratively with a 32-deep
   stack, visiting the nearer child first.

   Since the ray is a geodesic, the model is lensed for free: on the way in you
   see a second copy of it inside the mouth's ring, and from the throat it is
   stretched tangentially.

   **Shading and texturing.** Base colour only: the five distinct base colour
   images become a 2048² texture array, with the material → layer table read
   from the glTF (the hose and wood parts share one). A ray tracer has no
   screen-space derivatives to pick a mip from, so each triangle carries
   √(UV area / world area) and the shader reconstructs a level from that and
   the width of the ray's cone at the hit — one sample's angular size times the
   proper distance travelled. Without it a 2048² texture on a model a few
   hundred pixels across shimmers badly, especially under lensing, which
   stretches the model unevenly across the frame. Lighting is the same as the
   sphere's: a Lambert key light plus flat ambient, fixed in the model's own
   frame, since a distant light source would be a fiction here.

   Note that the model is scaled to a bounding radius of 0.43a rather than the
   old sphere's 0.25a. A bounding sphere is a loose fit around a helmet, and
   normalising the bound to 0.25a left it covering about a third of the screen
   area the ball did; 0.43a matches its apparent size instead. This leans
   harder on the "small compared with a, so locally flat" approximation, though
   less than the number suggests — the bound is set by the thin hose sticking
   up, and the surface rays actually hit sits far closer to the centre.
   `MODEL_R` in `src/scene.rs` trades the two off.

4. **Camera path.** The camera is a point (l, u) — radial coordinate plus a
   position on the sphere of radius r(l) = √(l² + a²) — carrying an
   orthonormal (forward, up) frame, so it can look anywhere, not just down
   the axis. The flight is in three acts:

   * **Dive (frames 1–45).** l runs from +14a to 0, but the camera also
     slides around the throat sphere by

         ψ(l) = K (g(14a) − g(l)),   g(l) = l / √(l² + a²)

     so that dψ/dl = −K a²/r³: the sideways drift falls off like 1/l³ and
     essentially the whole turn is spent within a couple of throat radii of
     l = 0. K is fixed by ψ(−14a) = π, i.e. **half a loop** around the
     inside of the throat over the traversal. Instead of falling straight
     in, the camera banks and the mouth swings out of frame.

     The heading is the path tangent taken not at the camera but at `l_aim`,
     a fraction `AIM_LEAD` of the way to the chase model; since the tangent
     at the middle of an arc is its chord, `AIM_LEAD` ≈ 0.5 aims straight at
     the model and less than that lets it drift around the frame. A pure
     tangent follower (`AIM_LEAD = 0`) looks 57.6° off the axis at l = 0.

   * **Orbit (frames 46–81).** Parked at l = 0, the camera circles the chase
     model once rather than spinning on the spot, so the model stays in
     frame while both universes and the throat sweep past behind it.

     A circle "around" something is awkward here: the connecting geodesic is
     bent, so aiming along the naive chord would miss, and there is no chart
     in which a constant-radius circle is a circle. Instead the camera is
     put at the far end of a geodesic fired *out of* the model: sweep the
     launch direction through 2π about the up axis, integrate a fixed proper
     length, and stand there facing back down the geodesic. Geodesics are
     reversible, so this is exactly a constant-proper-radius orbit with the
     model exactly in the middle of the frame, lensing and all. The launch
     direction that starts the orbit on the parked pose comes from a
     one-dimensional search (`sight_line`) for the geodesic joining the two —
     the shortest one, since strong lensing means there are several.

     Over the first radian of the orbit the camera sheds the 26° between its
     flight heading and the true line of sight, and takes it back on over the
     last, so the dive and the exit still join up cleanly. That lean is applied
     as a rotation *of the sight line*, not as a blend towards the parked
     heading: once the camera has moved, a fixed direction in map axes no
     longer means what it did at the parked pose.

     The frames are not spaced evenly in orbit angle. The orbit crosses the
     throat twice, and there the lensing swings the view about ten times
     faster than it does out in the open, so equal angular steps tear through
     those few frames and crawl through the rest. `orbit_schedule` samples the
     orbit finely, measures how much the view actually changes along it
     (turning, plus translation counted at one radian per orbit radius) and
     spaces the frames evenly in that instead, eased at both ends. Every frame
     is still an exact pose on the true orbit; only the timing along it
     changes. Startup prints the resulting average and worst per-frame turn
     and the two joins, which is where a cut would show up.

   * **Continue (frames 82–126).** The same curve, mirrored: l runs 0 →
     −14a while ψ finishes its half turn to π, heading the way it was
     originally going.

   Watching the frames in order you see universe B grow from a lensed disc,
   an Einstein ring of universe A form and invert as you swing through the
   throat, the two universes wheeling around the model during the orbit, and
   universe A shrink to a receding disc behind you — with the helmet leading
   the way through, lensed into a second copy and then stretched out as it
   crosses.

The integrator conserves |v|² = 1 and the angular momentum
L = √(l² + a²)·v̂_φ over a full traversal. Note the shader runs in f32, not the
f64 the CPU tracer used; typical rays escape in a few hundred steps, so the
drift stays far below a pixel, and the rays that do take tens of thousands of
steps are the ones winding on the photon ring, which come out black anyway.

## Build & run

Live, in a window:

    cargo run --release

Space pauses, left/right step a frame at a time, Home rewinds, Escape quits.
Defaults to 1280×720 with no supersampling, which is vsync-bound on a discrete
GPU.

Offline, to disk:

    cargo run --release -- --render

Outputs `out/frame_###.png` (126 frames, 1920×1080, 2×2 supersampled) and an
animated `out/wormhole.gif`. Both modes trace with the same shader into the
same texture — the window blits it through a deliberately non-sRGB surface so
that what you see is byte-for-byte what gets written.

Flags: `--width`, `--height`, `--ssaa`, and `--model FILE.gltf` to fly
something other than the helmet (it is centred and scaled to `MODEL_R`
automatically). The scene constants live at the top of `src/scene.rs`: FOV,
throat radius `A`, start/end distance (`L_START`/`L_END`), how far the camera
loops around the throat (`LOOP_SWEEP`), how tightly the frames bunch up there
(`PATH_BIAS`), the number of times the camera circles the model while parked
(`ORBIT_TURNS`), the chase model itself
(`MODEL_R`/`MODEL_LEAD`/`MODEL_RISE`/`MODEL_TURNS`/`AIM_LEAD`) and the per-act
frame counts.

For reference, the full 126-frame 1080p 2×2 render takes about 26 s end to end
on an RX 7900 — and most of that is PNG and GIF encoding on the CPU, with the
slowest frame tracing in 0.06 s.

To make an mp4 instead of the gif:

    ffmpeg -framerate 20 -i out/frame_%03d.png -pix_fmt yuv420p wormhole.mp4

## Assets

`gltf_models/FlightHelmet` is the Khronos glTF-Sample-Assets FlightHelmet
(CC0); see the `LICENSE.md` beside it. Only the base colour textures are used —
the normal and occlusion/roughness/metallic maps are left on disk.
