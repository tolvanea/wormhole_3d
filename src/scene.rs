//! Geometry of the Ellis wormhole, the flight path and the camera.
//!
//! This is the CPU half of the renderer: everything here runs once per frame
//! (or once per flight) in f64, and hands the GPU a small block of per-frame
//! camera state. The per-pixel geodesic integration lives in `tracer.wgsl`.
//!
//! Spatial metric (no time, no gravity):
//!     ds^2 = dl^2 + (l^2 + a^2) (dtheta^2 + sin^2 theta dphi^2)
//!
//! By spherical symmetry each geodesic lies in a plane through the "center",
//! so a ray reduces to the planar system (l, phi, v_l, v_phi):
//!
//!     dl/ds     = v_l
//!     dphi/ds   = v_phi
//!     dv_l/ds   = l * v_phi^2
//!     dv_phi/ds = -2 l / (l^2 + a^2) * v_l * v_phi
//!
//! integrated with classic RK4. The same system is integrated in the shader;
//! the duplication is deliberate, since the path solving here needs f64 and
//! the ray tracing there needs to run a few hundred million times a second.

use nalgebra::Vector3;
use std::f64::consts::PI;

pub type V3 = Vector3<f64>;

// ----------------------------- configuration -----------------------------

pub const A: f64 = 1.0; // throat radius (sets the length unit)

pub const FOV_DEG: f64 = 80.0; // horizontal field of view

pub const L_START: f64 = 14.0; // camera starts here (universe A) ...
pub const L_END: f64 = -14.0; // ... and ends here (universe B)

/// The camera does not fall straight down the axis: while it crosses it also
/// slides around the throat sphere by LOOP_SWEEP radians (half a turn), almost
/// all of it within a few throat radii of l = 0 -- a half loop inside the walls.
pub const LOOP_SWEEP: f64 = PI;

/// >1 puts more frames near the throat, where the turn happens.
pub const PATH_BIAS: f64 = 1.6;

pub const FRAMES_IN: usize = 45; // universe A -> throat
pub const FRAMES_ORBIT: usize = 36; // parked at the throat, circling the model
pub const FRAMES_OUT: usize = 45; // throat -> universe B
pub const ORBIT_TURNS: f64 = 1.0; // times the camera circles the model
/// Radians of orbit spent settling from the flight heading onto the model,
/// and again at the end returning to it.
pub const ORBIT_LOCK: f64 = 1.0;

// The glTF model flies the same path ahead of the camera. l decreases
// monotonically over the whole flight, so "ahead" is simply l - MODEL_LEAD.

/// Proper *bounding* radius of the chase model, which the loader scales it to.
///
/// The sphere this replaced had radius 0.25a and that radius was also its
/// silhouette. A bounding sphere is a much looser fit around a helmet: the
/// FlightHelmet's box is 0.37 x 0.72 x 0.37, so most of the bounding volume is
/// empty and normalising the bound to 0.25a left it covering barely a third of
/// the screen area the ball did. Sized by apparent area instead -- the
/// equivalent disc radius of its side-on silhouette works out at 0.58 of the
/// bounding radius -- 0.43a puts it back at the ball's on-screen presence.
///
/// This does lean harder on the "model is small compared with the throat, so
/// its neighbourhood is flat" approximation than the ball did. It stays
/// defensible because the bound is set by the thin hose sticking up: the
/// surface that rays actually hit sits far closer to the centre than 0.43a.
/// Turn it down if you would rather have the geometry exact than legible.
pub const MODEL_R: f64 = 0.43 * A;
pub const MODEL_LEAD: f64 = 2.5 * A; // how far ahead of the camera it flies
/// How far it flies above the flight plane, so that it does not eclipse the
/// wormhole mouth on the way in.
pub const MODEL_RISE: f64 = 0.7 * A;
pub const MODEL_LIGHT: [f64; 3] = [0.35, -0.25, 0.90]; // key light, model's frame
pub const MODEL_TURNS: f64 = 4.0; // turns about its own z axis, whole flight

/// How much the camera leans its heading toward the model: 0 follows its own
/// tangent (model drifts freely), ~0.5 is the chord to it (locked on).
pub const AIM_LEAD: f64 = 0.45;

pub const L_ESCAPE: f64 = 25.0 * A; // |l| beyond which the ray is "at infinity"
pub const H0: f64 = 0.02; // base integration step (scaled by sqrt(l^2+a^2))
pub const MAX_STEPS: u32 = 40_000; // safety cap; exceeded => photon ring, black

pub const SKY_GAIN: f64 = 2.0; // exposure applied to the sampled cube maps

// --------------------------- geodesic integrator --------------------------

/// Planar geodesic state: (l, phi, dl/ds, dphi/ds).
#[derive(Clone, Copy)]
pub struct GState {
    pub l: f64,
    pub phi: f64,
    pub vl: f64,
    pub vphi: f64,
}

impl GState {
    fn axpy(self, h: f64, d: GState) -> GState {
        GState {
            l: self.l + h * d.l,
            phi: self.phi + h * d.phi,
            vl: self.vl + h * d.vl,
            vphi: self.vphi + h * d.vphi,
        }
    }
}

/// Right-hand side of the ODE system derived from the Ellis metric.
fn deriv(s: &GState) -> GState {
    let r2 = s.l * s.l + A * A;
    GState {
        l: s.vl,
        phi: s.vphi,
        vl: s.l * s.vphi * s.vphi,
        vphi: -2.0 * s.l / r2 * s.vl * s.vphi,
    }
}

/// One classic RK4 step of size h.
pub fn rk4_step(s: &GState, h: f64) -> GState {
    let k1 = deriv(s);
    let k2 = deriv(&s.axpy(0.5 * h, k1));
    let k3 = deriv(&s.axpy(0.5 * h, k2));
    let k4 = deriv(&s.axpy(h, k3));
    GState {
        l: s.l + h / 6.0 * (k1.l + 2.0 * k2.l + 2.0 * k3.l + k4.l),
        phi: s.phi + h / 6.0 * (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi),
        vl: s.vl + h / 6.0 * (k1.vl + 2.0 * k2.vl + 2.0 * k3.vl + k4.vl),
        vphi: s.vphi + h / 6.0 * (k1.vphi + 2.0 * k2.vphi + 2.0 * k3.vphi + k4.vphi),
    }
}

/// Angular position of the ray on the sphere after it has swept `phi` radians
/// in its own geodesic plane, spanned by (p_hat, t_hat).
pub fn u_at(p_hat: V3, t_hat: V3, phi: f64) -> V3 {
    p_hat * phi.cos() + t_hat * phi.sin()
}

/// The plane's tangential basis vector, rotated along with it.
pub fn t_at(p_hat: V3, t_hat: V3, phi: f64) -> V3 {
    -p_hat * phi.sin() + t_hat * phi.cos()
}

/// Some unit vector perpendicular to `u`.
pub fn any_perp(u: V3) -> V3 {
    let a = if u.z.abs() < 0.9 { V3::z() } else { V3::x() };
    (a - u * a.dot(&u)).normalize()
}

/// Direction of travel in map axes, from a planar state and its plane basis.
fn heading(s: &GState, p_hat: V3, t_hat: V3) -> V3 {
    let r = (s.l * s.l + A * A).sqrt();
    (u_at(p_hat, t_hat, s.phi) * s.vl + t_at(p_hat, t_hat, s.phi) * (r * s.vphi)).normalize()
}

/// First-order proper offset of the manifold point (l, u) from (l0, u0), as a
/// vector in map axes: the radial part along u0 plus the tangential part along
/// the great circle from u0 towards u, scaled to a proper length by the radius
/// midway between them. The two are orthogonal, so |O| is the proper distance.
///
/// The tangential part uses the arc, not the chord |u - u0 (u . u0)|: the chord
/// vanishes at the antipode of u0 as well as at u0 itself, which would read as
/// a second, spurious coincidence for rays that have wound half way round.
///
/// This is the exact function the shader applies to each RK4 segment endpoint
/// to bring it into the model's local, effectively flat frame.
pub fn offset_from(l0: f64, u0: V3, l: f64, u: V3) -> V3 {
    let r_mid = (0.25 * (l + l0) * (l + l0) + A * A).sqrt();
    let c = u.dot(&u0);
    let across = u - u0 * c;
    let s = across.norm();
    let tangential = if s > 1e-12 {
        across * (r_mid * s.atan2(c) / s)
    } else if c > 0.0 {
        V3::zeros()
    } else {
        any_perp(u0) * (r_mid * PI)
    };
    u0 * (l - l0) + tangential
}

// ------------------------------- chase body --------------------------------

/// The chase model's placement, as a bounding sphere plus a spin.
///
/// The renderer intersects the real triangle mesh, but every *path* question
/// asked here -- where to stand to orbit it, which geodesic joins it to the
/// camera -- only needs to know roughly where it is and how big it is. Using
/// the bounding sphere for those keeps the flight planning independent of
/// which model is loaded.
#[derive(Clone, Copy)]
pub struct Body {
    pub l: f64,    // radial coordinate of the centre
    pub u: V3,     // angular position of the centre
    pub r: f64,    // proper bounding radius
    pub spin: f64, // rotation about its own z axis
}

impl Body {
    pub fn offset(&self, l: f64, u: V3) -> V3 {
        offset_from(self.l, self.u, l, u)
    }

    /// The model's local orthonormal frame in map axes: outward radial, then
    /// the two directions across it. `b3` is the model's own z axis, which is
    /// the map's +z while it rides the flight plane -- so "up" for the model
    /// is up for the camera too.
    pub fn frame(&self) -> (V3, V3, V3) {
        let b2 = V3::new(-self.u.y, self.u.x, 0.0).normalize();
        let b3 = self.u.cross(&b2);
        (self.u, b2, b3)
    }
}

// ----------------------- looking at the model, exactly ----------------------

const PROBE_STEPS: usize = 6_000;

/// Closest approach to `target`'s centre of the geodesic that leaves
/// (l0, p_hat) at angle `theta` from the outward radial, in the plane
/// (p_hat, t_hat), as (miss distance, arc length there).
fn probe(l0: f64, p_hat: V3, t_hat: V3, theta: f64, target: &Body) -> (f64, f64) {
    let mut s = GState {
        l: l0,
        phi: 0.0,
        vl: theta.cos(),
        vphi: theta.sin() / (l0 * l0 + A * A).sqrt(),
    };
    let mut arc = 0.0;
    let mut best = (f64::INFINITY, 0.0);
    for _ in 0..PROBE_STEPS {
        let d = target.offset(s.l, u_at(p_hat, t_hat, s.phi)).norm();
        if d < best.0 {
            best = (d, arc);
        }
        if s.l.abs() > L_ESCAPE && s.l * s.vl > 0.0 {
            break;
        }
        let h = H0 * (s.l * s.l + A * A).sqrt();
        s = rk4_step(&s, h);
        arc += h;
    }
    best
}

/// The geodesic that joins (l0, u0) to `target`: its length, and the direction
/// it sets off in. Any geodesic through two points lies in the plane spanned by
/// the two positions, so this is a one-dimensional search over the launch angle
/// in that plane -- run once per flight, not per ray.
///
/// Strong lensing means several geodesics connect the two points; the direct
/// one is the shortest, so among the launch angles that reach the target the
/// one with the least arc length wins.
pub fn sight_line(l0: f64, u0: V3, target: &Body) -> (f64, V3) {
    let t_hat = (target.u - u0 * u0.dot(&target.u)).normalize();
    let scan = 720;
    let step = 2.0 * PI / scan as f64;
    let samples: Vec<(f64, f64, f64)> = (0..scan) // (theta, miss, arc)
        .map(|i| {
            let theta = -PI + step * i as f64;
            let (miss, arc) = probe(l0, u0, t_hat, theta, target);
            (theta, miss, arc)
        })
        .collect();
    // The direct image is the shortest geodesic that reaches the target. Its
    // arc length identifies which launch angles belong to it; of those, the one
    // aimed best is the one that passes closest -- picking by arc length alone
    // would drift to the near edge of the target instead of its centre.
    let mut arc0 = f64::INFINITY;
    for s in &samples {
        if s.1 < target.r && s.2 < arc0 {
            arc0 = s.2;
        }
    }
    let mut pick = 0.0;
    let mut best_miss = f64::INFINITY;
    for s in &samples {
        let direct = arc0.is_infinite() || (s.1 < target.r && s.2 < 1.2 * arc0);
        if direct && s.1 < best_miss {
            best_miss = s.1;
            pick = s.0;
        }
    }
    // Ternary search on the miss distance to line the geodesic up exactly.
    let (mut lo, mut hi) = (pick - step, pick + step);
    for _ in 0..25 {
        let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
        if probe(l0, u0, t_hat, a, target).0 < probe(l0, u0, t_hat, b, target).0 {
            hi = b;
        } else {
            lo = a;
        }
    }
    let theta = 0.5 * (lo + hi);
    (
        probe(l0, u0, t_hat, theta, target).1,
        u_at(u0, t_hat, theta),
    )
}

/// A camera standing on the geodesic sphere of radius `dist` about the model:
/// shoot a geodesic out of the model in direction `e` and stand at its far end
/// facing back down it. Geodesics are reversible, so the model then sits in the
/// middle of the frame however badly the space in between is bending the light.
/// Returns (l, u, forward).
pub fn orbit_camera(body: &Body, e: V3, dist: f64) -> (f64, V3, V3) {
    let p_hat = body.u;
    let vl0 = e.dot(&p_hat);
    let tan = e - p_hat * vl0;
    let (t_hat, vph0) = if tan.norm() < 1e-12 {
        (any_perp(p_hat), 0.0)
    } else {
        (tan.normalize(), tan.norm())
    };
    let mut s = GState {
        l: body.l,
        phi: 0.0,
        vl: vl0,
        vphi: vph0 / (body.l * body.l + A * A).sqrt(),
    };
    let mut arc = 0.0;
    while arc < dist {
        let h = (H0 * (s.l * s.l + A * A).sqrt()).min(dist - arc);
        s = rk4_step(&s, h);
        arc += h;
    }
    (
        s.l,
        u_at(p_hat, t_hat, s.phi),
        -heading(&s, p_hat, t_hat), // face back the way we came
    )
}

/// Orbit angle for each of `frames` frames.
///
/// Not simply equal steps in angle: the orbit crosses the throat twice, and
/// there the lensing swings the view an order of magnitude faster than it does
/// out in the open, so equal angular steps tear through those few frames and
/// crawl through the rest. Sample the orbit finely, measure how much the view
/// really changes along it -- turning, plus translation counted at one radian
/// per orbit radius -- and space the frames evenly in *that*, eased at both
/// ends. Every frame is still an exact pose on the true orbit; only the timing
/// along it changes.
pub fn orbit_schedule(body: &Body, e0: V3, dist: f64, total: f64, frames: usize) -> Vec<f64> {
    let n = 2000;
    let mut cumulative = Vec::with_capacity(n + 1);
    let mut seen = 0.0;
    let mut prev: Option<(f64, V3, V3)> = None;
    for i in 0..=n {
        let a = total * i as f64 / n as f64;
        let (l, u, fwd) = orbit_camera(body, rotate_about(e0, V3::z(), a), dist);
        if let Some((pl, pu, pfwd)) = prev {
            let moved = offset_from(pl, pu, l, u).norm() / dist;
            let turned = pfwd.dot(&fwd).clamp(-1.0, 1.0).acos();
            seen += (moved * moved + turned * turned).sqrt();
        }
        cumulative.push(seen);
        prev = Some((l, u, fwd));
    }

    let mut out = Vec::with_capacity(frames);
    let mut i = 0;
    for f in 0..frames {
        let t = f as f64 / frames as f64;
        let want = seen * 0.5 * (1.0 - (PI * t).cos()); // slow-in / slow-out
        while i + 1 < n && cumulative[i + 1] < want {
            i += 1;
        }
        let (lo, hi) = (cumulative[i], cumulative[i + 1]);
        let frac = if hi > lo {
            (want - lo) / (hi - lo)
        } else {
            0.0
        };
        out.push(total * (i as f64 + frac) / n as f64);
    }
    out
}

// ------------------------------- camera path -------------------------------

/// Where the camera is and where it points.
///
/// A point of the manifold is (l, u) with u a unit vector on the sphere of
/// radius r(l) = sqrt(l^2 + a^2); the local orthonormal frame is u itself
/// (outward radial) plus the two directions perpendicular to it, so any
/// viewing direction is simply a unit 3-vector in these "map" axes.
#[derive(Clone, Copy)]
pub struct Camera {
    pub l: f64,  // radial coordinate
    pub u: V3,   // angular position on the sphere
    pub fwd: V3, // viewing direction (unit)
    pub up: V3,  // screen up (unit, perpendicular to fwd)
}

/// Rodrigues rotation of `v` about the unit axis `k` by `ang`.
pub fn rotate_about(v: V3, k: V3, ang: f64) -> V3 {
    let (c, s) = (ang.cos(), ang.sin());
    v * c + k.cross(&v) * s + k * (k.dot(&v) * (1.0 - c))
}

/// Angular position along the flight, as a function of l.
///
///     psi(l) = K * (g(L_START) - g(l)),   g(l) = l / sqrt(l^2 + a^2)
///
/// so dpsi/dl = -K a^2 / r^3: the sideways drift dies off like 1/l^3 and the
/// whole turn is spent near the throat. K is fixed by psi(L_END) = LOOP_SWEEP.
fn sweep_g(l: f64) -> f64 {
    l / (l * l + A * A).sqrt()
}

pub fn sweep_k() -> f64 {
    LOOP_SWEEP / (sweep_g(L_START) - sweep_g(L_END))
}

pub fn path_psi(l: f64) -> f64 {
    sweep_k() * (sweep_g(L_START) - sweep_g(l))
}

/// The chase model for a camera at radial coordinate `l`: same path, further
/// along it.
pub fn body_at(l: f64) -> Body {
    let ls = l - MODEL_LEAD;
    let psi = path_psi(ls);
    // Tilted out of the flight plane by a proper distance MODEL_RISE, which at
    // radius r is an angle MODEL_RISE / r (so it barely leaves the axis far
    // away and rides well clear of it near the throat).
    let rise = MODEL_RISE / (ls * ls + A * A).sqrt();
    Body {
        l: ls,
        u: V3::new(psi.cos(), psi.sin(), 0.0) * rise.cos() + V3::new(0.0, 0.0, rise.sin()),
        r: MODEL_R,
        spin: 0.0,
    }
}

/// Camera flying the path at radial coordinate `l`.
///
/// The heading is the path tangent evaluated at `l_aim` rather than at `l`,
/// then re-expressed in the camera's own (radial, psi) basis. `l_aim = l`
/// gives a pure tangent follower; pulling `l_aim` toward the chase model
/// swings the heading into the turn, and since the tangent at the midpoint of
/// an arc is its chord, `l_aim` half way to the model aims roughly at it.
pub fn camera_on_path(l: f64, l_aim: f64) -> Camera {
    let psi = path_psi(l);
    let u = V3::new(psi.cos(), psi.sin(), 0.0); // position on the sphere
    let e_psi = V3::new(-psi.sin(), psi.cos(), 0.0); // direction of increasing psi
    // Unit tangent of the path, travelling toward decreasing l. The tangential
    // term is r * |dpsi/dl| = K a^2 / r^2, i.e. ~57 degrees off the axis at the
    // throat and negligible far away.
    let fwd = (-u + e_psi * (sweep_k() * A * A / (l_aim * l_aim + A * A))).normalize();
    // The path stays in the z = 0 plane of the map axes, so +z is a valid,
    // constant "up" everywhere along it (no roll).
    let up = V3::new(0.0, 0.0, 1.0);
    Camera { l, u, fwd, up }
}

/// Screen up for a free camera: +z straightened against the viewing direction.
pub fn up_for(fwd: V3) -> V3 {
    (V3::z() - fwd * fwd.z).normalize()
}

pub fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

// ------------------------------- flight plan -------------------------------

/// One posed frame of the flight: where the camera is, and where the model is.
#[derive(Clone, Copy)]
pub struct Shot {
    pub cam: Camera,
    pub body: Body,
}

/// The whole scripted flight: dive in (turning half way around the throat on
/// the way), park at l = 0 and circle the model, then resume the curve outward.
///
/// Both the live viewer and the headless renderer walk this same list, so what
/// you see in the window is exactly what gets written to disk.
pub fn flight_plan() -> Vec<Shot> {
    let ease = |t: f64| 0.5 * (1.0 - (PI * t).cos()); // slow-in / slow-out
    // Fraction of the way out from the throat after an eased time t of a leg:
    // 1 at the far end, 0 at the throat, and with zero speed at both.
    let leg = |t: f64| (1.0 - ease(t)).powf(PATH_BIAS);
    // The camera chases the model: it leans AIM_LEAD of the way from its own
    // tangent toward the chord, so the model sits loosely in frame instead of
    // being pinned to the centre.
    let pose = |l: f64| camera_on_path(l, l - AIM_LEAD * MODEL_LEAD);

    let mut shots: Vec<Shot> = Vec::new();
    for f in 0..FRAMES_IN {
        // Decelerates into the throat; the l = 0 frame belongs to the orbit.
        let t = f as f64 / FRAMES_IN as f64;
        let l = L_START * leg(t);
        shots.push(Shot {
            cam: pose(l),
            body: body_at(l),
        });
    }
    {
        // Parked at the throat, the camera circles the model rather than
        // spinning on the spot, so the model stays in frame throughout.
        let park = pose(0.0);
        let model = body_at(0.0);
        // Solve for the geodesic that leaves the model and arrives at the
        // parked camera: its length is the orbit radius, its launch direction
        // is where the orbit starts. Searching outwards from the model rather
        // than inwards from the camera puts the residual error at the camera
        // end, instead of letting the lensing in between amplify it.
        let eye = Body {
            l: park.l,
            u: park.u,
            r: MODEL_R,
            spin: 0.0,
        };
        let (dist, e0) = sight_line(model.l, model.u, &eye);
        let start = orbit_camera(&model, e0, dist);
        eprintln!(
            "orbit: radius {:.3}, starts {:.3} from the parked camera, \
             {:.1} deg off its heading",
            dist,
            eye.offset(start.0, start.1).norm(),
            start.2.dot(&park.fwd).acos().to_degrees()
        );
        // The flight heading and the line to the model differ by a fixed
        // rotation. Shedding it over the first radians of the orbit (and
        // taking it back on over the last) lets the dive and the exit -- still
        // tangent followers -- join up cleanly, while never swinging further
        // than that from the model. Blending towards the parked heading as a
        // *vector* would not do: once the camera has moved, a fixed direction
        // in map axes no longer means the same thing.
        // Axis order matters: rotating the sight line about (sight x heading)
        // by +angle lands on the heading, the other way round doubles the gap.
        let lean = start.2.cross(&park.fwd);
        let (lean_axis, lean_angle) = if lean.norm() > 1e-9 {
            (
                lean.normalize(),
                start.2.dot(&park.fwd).clamp(-1.0, 1.0).acos(),
            )
        } else {
            (V3::z(), 0.0)
        };
        let total = ORBIT_TURNS * 2.0 * PI;
        let schedule = orbit_schedule(&model, e0, dist, total, FRAMES_ORBIT);
        for &a in &schedule {
            let (l, u, fwd_model) = orbit_camera(&model, rotate_about(e0, V3::z(), a), dist);
            // Shed the lean over the first radians of the orbit and take it
            // back on over the last. Measured in orbit angle, not frames, so
            // that it is fully off at a = 0 and fully back on as a returns to
            // 2*pi -- which is exactly where the two joins with the flight are.
            let w = smoothstep(a.min(total - a) / ORBIT_LOCK);
            let axis = rotate_about(lean_axis, V3::z(), a);
            let fwd = rotate_about(fwd_model, axis, (1.0 - w) * lean_angle);
            shots.push(Shot {
                cam: Camera {
                    l,
                    u,
                    fwd,
                    up: up_for(fwd),
                },
                body: model,
            });
        }
    }
    for f in 0..FRAMES_OUT {
        // Mirror image of the dive: accelerates away from the throat.
        let t = (f + 1) as f64 / FRAMES_OUT as f64;
        let l = L_END * leg(1.0 - t);
        shots.push(Shot {
            cam: pose(l),
            body: body_at(l),
        });
    }

    // The model turns steadily on its own axis for the whole flight, so it is
    // visibly spinning even while the camera is parked and circling it.
    let frames = shots.len();
    for (f, shot) in shots.iter_mut().enumerate() {
        shot.body.spin = MODEL_TURNS * 2.0 * PI * f as f64 / frames as f64;
    }
    shots
}

/// A cut shows up as one frame that turns much further than its neighbours, so
/// report the worst of them: it should stay in the same league as the average,
/// and in particular not spike at the two joins with the flight.
pub fn report_continuity(shots: &[Shot]) {
    let turn = |a: &Camera, b: &Camera| a.fwd.dot(&b.fwd).clamp(-1.0, 1.0).acos().to_degrees();
    let (mut worst, mut at, mut sum) = (0.0f64, 0, 0.0);
    for (f, w) in shots.windows(2).enumerate() {
        let d = turn(&w[0].cam, &w[1].cam);
        sum += d;
        if d > worst {
            worst = d;
            at = f;
        }
    }
    let join = |f: usize| turn(&shots[f].cam, &shots[f + 1].cam);
    eprintln!(
        "camera: turns {:.1} deg/frame on average, at most {:.1} deg (frames {} -> {}); \
         joins {:.2} and {:.2} deg",
        sum / (shots.len() - 1) as f64,
        worst,
        at,
        at + 1,
        join(FRAMES_IN - 1),
        join(FRAMES_IN + FRAMES_ORBIT - 1)
    );
}
