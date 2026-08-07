# Ellis wormhole fly-through renderer

A small Rust ray tracer that renders a camera flight *through* an Ellis
wormhole. There is no time and no gravity: light rays are simply geodesics of
the static 3D spatial metric

    ds² = dl² + (l² + a²)(dθ² + sin²θ dφ²)

whose embedded throat profile is the catenoid r = a·cosh(z/a). The coordinate
l ∈ (−∞, +∞) is proper radial distance; l > 0 is "universe A", l < 0 is
"universe B", and l = 0 is the throat.

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
   sampled — `cubemap/cubemap_1.jpg` for universe A, `cubemap/cubemap_2.jpg`
   for universe B. Both are 4×3 cross-form images with the cross on the left
   (vertical arm in column 1, −Z trailing off to the right); they are cut
   into six faces, linearized from sRGB and filtered bilinearly.
   Rays that exhaust the step budget are asymptotically winding
   on the throat's photon ring and are painted black — that is the thin dark
   ring you see at the critical impact parameter.

3. **Solid geometry: the chase sphere.** A checkered sphere of proper radius
   0.25a flies the same path 2.5a ahead of the camera (and 0.7a above the
   flight plane, so it does not eclipse the mouth), turning steadily about its
   own z axis. Hitting it needs the ray's 3D position, which the planar
   reduction still has: after sweeping φ the ray sits at angular position
   u(φ) = p̂ cos φ + t̂ sin φ, so each RK4 step gives a manifold point (l, u).

   There is no global Cartesian chart to test against — the 3D wormhole embeds
   in R⁴, not R³ — but none is needed. The sphere is small compared with a, so
   its neighbourhood is flat and the offset of (l, u) from the centre (l_c, u_c)
   is, to first order,

       O = (l − l_c)·u_c + r_mid·σ·ê,   r_mid = r((l + l_c)/2)

   where σ = ∠(u, u_c) and ê is the unit vector from u_c towards u across the
   sphere. Note σ, the arc, and not the chord |u − u_c(u·u_c)|: the chord
   vanishes at the antipode of the centre as well as at the centre itself,
   which would read as a second, spurious sphere for rays that have wound half
   way round.

   The two terms are orthogonal, |O| is the proper distance, and O is also the
   outward surface normal — so `|O| < R` is the whole intersection test, valid
   right through the throat. Almost every ray is rejected by a single
   comparison on l; near the sphere the step is clamped so it cannot tunnel
   through, and the step that lands inside is bisected onto the surface.

   Since the ray is a geodesic, the sphere is lensed for free: on the way in
   you see a second copy of it inside the mouth's ring, and from the throat it
   is stretched tangentially into an ellipse. Swapping this for a triangle mesh
   means replacing `Sphere::offset` with a segment-vs-BVH query over the
   per-step segments — the marching, rejection and refinement are the same.

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
     a fraction `AIM_LEAD` of the way to the chase sphere; since the tangent
     at the middle of an arc is its chord, `AIM_LEAD` ≈ 0.5 aims straight at
     the sphere and less than that lets it drift around the frame. A pure
     tangent follower (`AIM_LEAD = 0`) looks 57.6° off the axis at l = 0.

   * **Orbit (frames 46–81).** Parked at l = 0, the camera circles the chase
     sphere once rather than spinning on the spot, so the sphere stays in
     frame while both universes and the throat sweep past behind it.

     A circle "around" something is awkward here: the connecting geodesic is
     bent, so aiming along the naive chord would miss, and there is no chart
     in which a constant-radius circle is a circle. Instead the camera is
     put at the far end of a geodesic fired *out of* the sphere: sweep the
     launch direction through 2π about the up axis, integrate a fixed proper
     length, and stand there facing back down the geodesic. Geodesics are
     reversible, so this is exactly a constant-proper-radius orbit with the
     sphere exactly in the middle of the frame, lensing and all. The launch
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
   throat, the two universes wheeling around the sphere during the orbit, and
   universe A shrink to a receding disc behind you — with the sphere leading
   the way through, lensed into a second copy and then an ellipse as it
   crosses.

The integrator conserves |v|² = 1 and the angular momentum
L = √(l² + a²)·v̂_φ to ~10⁻⁸ over a full traversal (printed at startup).

## Build & run

    cargo run --release

Outputs `out/frame_###.png` (126 frames, 640×360, 2×2 supersampled) and an
animated `out/wormhole.gif`. Tweak the constants at the top of
`src/main.rs`: resolution, FOV, throat radius `A`, start/end distance
(`L_START`/`L_END`), how far the camera loops around the throat
(`LOOP_SWEEP`), how tightly the frames bunch up there (`PATH_BIAS`), the
number of times the camera circles the sphere while parked (`ORBIT_TURNS`),
the chase sphere itself (`SPHERE_R`/`SPHERE_LEAD`/`SPHERE_RISE`/`SPHERE_TURNS`
/`AIM_LEAD`) and the per-act frame counts. On older toolchains (rustc 1.75) pin rayon:

    cargo update rayon --precise 1.10.0
    cargo update rayon-core --precise 1.12.1

To make an mp4 instead of the gif:

    ffmpeg -framerate 20 -i out/frame_%03d.png -pix_fmt yuv420p wormhole.mp4
