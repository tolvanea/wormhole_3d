# How it works

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

3. **Solid geometry: the chase model.** A glTF model — a pony in a space suit,
   39,289 triangles across eleven primitives — flies the same path 2.5a ahead of
   the camera (and 0.7a above the flight plane, so it does not eclipse the
   mouth). It can turn about its own z axis, though `MODEL_TURNS` is 0 for this
   flight: the camera is doing the moving, and a spinning subject fights that.
   Hitting it needs the ray's 3D
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

   The BVH is a binned-SAH build (12 bins, ≤4 triangles per leaf, 26,121 nodes)
   flattened depth-first with children allocated in pairs, so one index per
   interior node locates both; the shader walks it iteratively with a 32-deep
   stack, visiting the nearer child first.

   Since the ray is a geodesic, the model is lensed for free: on the way in you
   see a second copy of it inside the mouth's ring, and from the throat it is
   stretched tangentially.

   **Materials.** Triangles carry a material index, and a small storage buffer
   holds one record per material: `baseColorFactor` (RGBA), a base colour layer
   or −1 for none, alpha mode and cutoff, and an emissive colour. Indexing by
   material rather than straight to a texture layer is what lets a material have
   no texture at all — the pony's teeth, tongue, eyelashes and glowing pip are
   pure factors, and pointing those at "layer 0" would paint them with whatever
   image happened to be first.

   **Base colour and mips.** The distinct images become a texture array, sized
   to the next power of two at or above the largest source rather than a fixed
   2048, so a model of mostly-1024 images does not pay triple for one 2000-pixel
   outlier. A ray tracer has no screen-space derivatives to pick a mip from, so
   each triangle carries √(UV area / world area) and the shader reconstructs a
   level from that and the width of the ray's cone at the hit — one sample's
   angular size times the proper distance travelled. Without it a 2048² texture
   on a model a few hundred pixels across shimmers badly, especially under
   lensing, which stretches the model unevenly across the frame.

   **Transparency.** `BLEND` and `MASK` materials are composited front to back
   along the ray. The march carries a colour and a remaining transmittance; an
   opaque hit ends the ray, a blended one takes its share (`colour += throughput
   · α · shade`, `throughput ·= 1 − α`) and lets the rest carry on, and a masked
   texel below its cutoff costs nothing at all. Whatever transmittance survives
   to the end multiplies the sky, so a visor still shows the universe through it.

   The subtlety is that a single RK4 step can cross several surfaces — the near
   and far walls of a helmet dome, most obviously. So each step's segment is
   walked repeatedly, each pass starting just past the previous hit, until the
   segment is exhausted or the transmittance drops below a cut-off. The BVH
   already returns the nearest hit, so front-to-back ordering comes for free and
   nothing has to be sorted. Note the ray is *not* bent at the interface: there
   is no refraction here, only absorption, so a lens tints and dims what is
   behind it but does not displace it.

   **Emission.** `emissiveFactor` is folded together with
   `KHR_materials_emissive_strength` at load time and added at the hit, with an
   emissive texture supported but unused by this model. It is deliberately
   naive: the surface glows at itself and nothing else in the scene is any
   brighter for it, because with one ray per sample and no secondary rays there
   is nothing to gather the light with. It reads correctly on the pony's cyan
   pips and is the term a future path tracer would sample as a light source
   rather than simply add on arrival.

   Startup prints a line for every non-opaque or emissive material, including
   the mean alpha of a blended material's texture — a material marked `BLEND`
   whose texture is solid alpha renders exactly like an opaque one, and that is
   worth telling apart from the transparency being broken.

   **Lighting** is unchanged: a Lambert key light plus flat ambient, fixed in
   the model's own frame, since a distant light source would be a fiction here.

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
   the axis. The flight reads as two acts but is built as a single continuous
   move; nothing stops and there is no join to stitch.

   **The approach** (`INTRO_FRAMES`). The pony drifts backwards towards the
   wormhole — she faces the camera the whole way, so the mouth is behind her and
   largely hidden by her — while the camera withdraws from just off her face
   (`INTRO_DIST_START`) to the distance it keeps thereafter (`ORBIT_DIST`, 2.5a).
   Nothing rotates yet.

   The camera also *stands* `INTRO_RISE` above the flight axis and *looks*
   `INTRO_AIM` above it, both falling to zero by the crossing, so it drifts back
   and slightly down onto the axis over the approach. Standing at eye height and
   looking level is not the same shot as standing on the axis and tilting up:
   the first is a portrait, the second looks up her nose. With the two constants
   equal the view comes out level — the geodesic is fired out of the flight
   plane by the rise, and only the *difference* between where the camera looks
   and where it stands is applied as a tilt, so equal values mean no tilt at all.

   Note that `INTRO_DIST_START` and `INTRO_RISE` are perpendicular, so what has
   to clear the model is √(dist² + rise²), not either alone. Below the model's
   bounding radius the opening frames render the inside of its faces; startup
   warns when that is the case.

   Both ramps are smootherstep in l rather than smoothstep, because its *second*
   derivative vanishes at the ends too: the pull-back has to arrive at the
   crossing with no residual acceleration or the join reads as a small lurch.
   The bearing is clamped at zero over this stretch, so the turn begins exactly
   when the camera reaches its final distance.

   **The crossing** (`FRAMES`, 126) is everything below.

   The model flies the path. l runs monotonically +14a → −14a while the model
   also slides around the throat sphere by

       ψ(l) = K (g(14a) − g(l)),   g(l) = l / √(l² + a²)

   so that dψ/dl = −K a²/r³: the sideways drift falls off like 1/l³ and
   essentially the whole turn is spent within a couple of throat radii of
   l = 0. K is fixed by ψ(−14a) = π, i.e. **half a loop** around the inside of
   the throat over the traversal.

   The camera rides a geodesic sphere about it. Rather than flying its own
   path, the camera stays a constant proper distance `ORBIT_DIST` from the
   model, at a bearing that sweeps a full 2π over the flight. A circle
   "around" something is awkward here: the connecting geodesic is bent, so
   aiming along the naive chord would miss, and there is no chart in which a
   constant-radius circle is a circle. Instead the camera is put at the far end
   of a geodesic fired *out of* the model — sweep the launch direction, integrate
   a fixed proper length, stand there facing back down it. Geodesics are
   reversible, so this is exactly a constant-proper-distance orbit, lensing and
   all. Bearing zero fires it back down the path, which puts the camera exactly
   where a plain chase camera would be, so the orbit starts and ends in the
   chase pose with nothing to blend.

   The turn is concentrated at the wormhole. The bearing rate is a Lorentzian
   in l, dθ/dl ∝ 1/(1 + (l/`ORBIT_SPREAD`)²), which integrates in closed form
   to spread·atan(l/spread). The camera is therefore nearly steady out in the
   open, eases into the turn a few throat radii out, and does the bulk of it
   while crossing — and normalising by the total makes it exactly one turn end
   to end, with no accumulated drift.

   Where the camera looks is blended, not fixed. Facing straight back down the
   geodesic holds the model dead centre, which is what an orbit should do — but
   held for the whole flight it also parks the model in front of the wormhole
   mouth and eclipses it, which is the very thing `MODEL_RISE` exists to
   prevent. So the heading blends between the flight direction and the model,
   weighted by **bearing** rather than by l. The model sits at roughly the
   bearing angle off the flight axis, so sin(θ/2) is zero behind the model
   (where the two agree anyway), one in front of it, and enough in between to
   keep the model within 28° of centre the whole way round — comfortably inside
   the 40° half-FOV, and printed at startup so a regression cannot hide.
   Keying that blend to l instead is a subtle trap: during the orbit the camera
   swings beside and even ahead of the model, so an l-keyed weight releases far
   too early and the model silently leaves the shot for a third of the flight.

   The two acts get the frames they were asked for. Left alone the split would
   fall out of whatever the geometry happened to measure, which is no way to tune
   a shot — and the measure badly overstates the approach, since it counts the
   camera's absolute travel while the pony travels with it and the sky is at
   infinity, so almost nothing on screen actually changes. So the approach's
   share of the measure is scaled by a factor solved for in closed form (the
   measure is linear in it), applied through the same smootherstep as the
   distance ramp so that it is exactly 1 at the join and introduces no speed
   step. Startup reports the split it actually achieved against the one asked
   for; they agree to within a frame.

   The timing comes from measuring, not from a formula. `pose_at` defines the
   *curve* — a purely geometric object with no timing in it — and then the curve
   is walked once, measuring how much the view actually changes along it
   (turning, plus translation counted at one radian per orbit radius), and the
   frames are placed at equal intervals of that. Near the throat the lensing
   swings the view about ten times faster per unit of l than out in the open, so
   equal steps in l would tear through the crossing and crawl through the rest;
   equal steps in view change do the opposite. That is where the heavy slowdown
   in the middle comes from — it is not imposed, it falls out of the geometry.
   The model crosses the throat about **12× slower** than it cruises, and never
   stops. `THROAT_DWELL` biases the measure further if you want more.

   Easing is applied at the two ends only (`END_RAMP`), as a smoothstep ramp on
   the velocity. The obvious choice — a half cosine over the whole flight — puts
   its peak speed exactly at the midpoint, which is precisely where the crossing
   wants to be slowest, and it was the single largest source of leftover jerk.

   Startup prints the average and worst per-frame turn and translation. With
   the flight built as one move these sit close together (4.4°/frame average
   against an 11° worst, where the old three-act version managed 5.4° against
   27°), and the peak now falls in the middle of the crossing rather than at a
   frame index where two pieces used to meet.

   Watching the frames in order you see universe B grow from a lensed disc,
   an Einstein ring of universe A form and invert as you swing through the
   throat, the two universes wheeling around the model as the camera comes all
   the way round it, and universe A shrink to a receding disc behind you — with
   the helmet leading the way through, lensed into a second copy and then
   stretched out along the mouth as it crosses.

The integrator conserves |v|² = 1 and the angular momentum
L = √(l² + a²)·v̂_φ over a full traversal. Note the shader runs in f32, not the
f64 the CPU tracer used; typical rays escape in a few hundred steps, so the
drift stays far below a pixel, and the rays that do take tens of thousands of
steps are the ones winding on the photon ring, which come out black anyway.
