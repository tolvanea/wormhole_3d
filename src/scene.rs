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

// ------------------------- the flight, in two acts --------------------------
//
// Act one is the approach: the pony drifts backwards towards the wormhole with
// the camera up at her face, and the camera draws steadily away from her until
// it reaches the distance it will keep for the rest of the flight. Nothing
// rotates yet. Act two is everything that was here before -- the crossing, with
// the camera swinging once all the way around her.
//
// The two are one continuous curve, so nothing has to be blended at the join;
// the only thing that marks it is the distance ramp finishing.

/// Frames spent on the approach, and on the crossing. Both are honoured
/// exactly: the pacing solves for the split rather than hoping for it, and the
/// startup line reports what it actually got.
pub const INTRO_FRAMES: usize = 122;
pub const FRAMES: usize = 252;

/// Where the approach begins. The wormhole is a speck at this range, and the
/// pony is between the camera and it.
pub const L_INTRO_START: f64 = 40.0 * A;

/// Proper distance from the pony's centre at the very first frame.
///
/// Her local box runs -0.35..+0.35 along the axis she faces, so this wants to
/// stay comfortably above 0.35 or the camera starts out inside her nose --
/// startup warns if it does not. The default sits just off her face.
pub const INTRO_DIST_START: f64 = 0.35 * A;

/// How far up the model the camera looks during the approach, in proper units
/// above its centre. Aimed at the centre it would be staring at her chest from
/// that range; this lifts the shot onto her eyes and settles back to the centre
/// as the camera withdraws.
pub const INTRO_AIM: f64 = 0.35 * A;


/// Turns the camera makes around the model over the whole flight.
pub const ORBIT_TURNS: f64 = 1.0;

/// Half-width, in throat radii, of the region the camera does its turning in.
///
/// The orbit rate is weighted by 1 / (1 + (l/spread)^2), so the camera is
/// already easing into the turn a few radii out and does the bulk of it while
/// crossing. Because the weights are normalised over the frames, the total
/// comes to exactly ORBIT_TURNS however the speed profile is tuned.
pub const ORBIT_SPREAD: f64 = 3.0 * A;

/// Proper distance the camera keeps from the model.
///
/// Constant for the whole flight, which is what makes the motion read as one
/// move: the camera is on a geodesic sphere about the model, and the only
/// thing that changes is the bearing.
pub const ORBIT_DIST: f64 = MODEL_LEAD;

/// How firmly the camera holds the model once it has swung off to one side.
///
/// Out in the open the camera looks along its own flight path, so the wormhole
/// mouth is dead ahead and the model rides off to one side on its MODEL_RISE --
/// the original framing, and the reason that constant exists. But that only
/// works while the camera is *behind* the model. Once the orbit has carried it
/// round to the side, the flight direction and the model are nearly a right
/// angle apart, and looking along the flight would leave the model out of shot
/// entirely.
///
/// So the blend is keyed to the bearing rather than to l: the model sits at
/// about `bearing` off the flight axis, so weighting by sin(bearing/2) is zero
/// where the camera is behind it (and the two agree anyway), one where the
/// camera is in front, and enough in between to keep the residual inside 30
/// degrees the whole way round. Raising this exponent grips harder.
pub const LOOK_GRIP: f64 = 0.65;

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
/// Turns the model makes about its own z axis over the whole flight. Zero
/// keeps it still, which is what the approach wants -- the camera is doing the
/// moving now, and a spinning subject fights that.
pub const MODEL_TURNS: f64 = 0.0;

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

// ------------------------ standing off the model ---------------------------

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

/// Unit tangent of the flight path at radial coordinate `l`, pointing the way
/// the flight is going (toward decreasing l).
///
/// The tangential term is r * |dpsi/dl| = K a^2 / r^2, i.e. ~57 degrees off
/// the axis at the throat and negligible far away.
pub fn path_tangent(l: f64) -> V3 {
    let psi = path_psi(l);
    let u = V3::new(psi.cos(), psi.sin(), 0.0); // position on the sphere
    let e_psi = V3::new(-psi.sin(), psi.cos(), 0.0); // direction of increasing psi
    (-u + e_psi * (sweep_k() * A * A / (l * l + A * A))).normalize()
}

/// Screen up for a free camera: +z straightened against the viewing direction.
pub fn up_for(fwd: V3) -> V3 {
    (V3::z() - fwd * fwd.z).normalize()
}

// ------------------------------- the flight ---------------------------------

/// One posed frame of the flight: where the camera is, and where the model is.
#[derive(Clone, Copy)]
pub struct Shot {
    pub cam: Camera,
    pub body: Body,
}

/// How densely the curve is sampled before the frames are placed on it, and
/// how much that sampling leans toward the throat. Neither is a timing choice
/// -- the timing comes out of the measurement below -- they only decide how
/// finely the interesting part gets measured.
const CURVE_SAMPLES: usize = 20_000;
const SAMPLE_BIAS: f64 = 0.05;

/// A monotone sweep of l from L_INTRO_START to L_END, used only to enumerate the
/// curve. The odd cubic s(x) = c*x + (1-c)*x^3 is increasing everywhere
/// (s'(x) = c + 3(1-c)x^2 >= c > 0), so this never doubles back, and its
/// shallow slope at x = 0 puts more samples near the throat.
fn l_sample(q: f64) -> f64 {
    let x = 1.0 - 2.0 * q.clamp(0.0, 1.0);
    let s = SAMPLE_BIAS * x + (1.0 - SAMPLE_BIAS) * x * x * x;
    // Mapped per side so that s = 0 is the throat even if the two ends are not
    // the same distance out.
    if s >= 0.0 { L_INTRO_START * s } else { -L_END * s }
}

/// Bearing of the camera around the model, as a function of where the model is.
///
/// The turn rate is a Lorentzian in l,
///
///     dtheta/dl  proportional to  1 / (1 + (l/spread)^2)
///
/// which integrates to spread * atan(l/spread) in closed form. So the camera is
/// nearly steady out in the open, eases into the turn a few throat radii out,
/// and does the bulk of it while crossing -- and normalising by the total makes
/// it exactly ORBIT_TURNS from one end of the flight to the other, with no
/// accumulated drift.
fn bearing_at(l: f64) -> f64 {
    let f = |x: f64| ORBIT_SPREAD * (x / ORBIT_SPREAD).atan();
    // Clamped at zero, so the whole approach is spent squarely behind her and
    // the turn only begins once the camera has reached its final distance.
    (ORBIT_TURNS * 2.0 * PI * (f(L_START) - f(l)) / (f(L_START) - f(L_END))).max(0.0)
}

/// How far into the approach `l` is: 1 at the very start, easing to 0 exactly
/// where the crossing begins.
///
/// Smootherstep rather than smoothstep because its second derivative vanishes
/// at both ends too -- the camera's pull-back has to arrive at the crossing
/// with no residual acceleration, or the join reads as a small lurch.
fn intro_t(l: f64) -> f64 {
    if L_INTRO_START <= L_START {
        return 0.0;
    }
    let x = ((l - L_START) / (L_INTRO_START - L_START)).clamp(0.0, 1.0);
    x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
}

/// Proper distance from the model at `l`: withdrawing during the approach,
/// constant from the crossing onwards.
pub fn dist_at(l: f64) -> f64 {
    ORBIT_DIST + (INTRO_DIST_START - ORBIT_DIST) * intro_t(l)
}

/// How far above the model's centre the camera is looking at `l`.
fn aim_at(l: f64) -> f64 {
    INTRO_AIM * intro_t(l)
}

/// Where the camera is, and where the model is, when the model has reached `l`.
///
/// The camera is placed by firing a geodesic *out of* the model and standing at
/// its far end facing back down it. Geodesics are reversible, so this is exactly
/// a constant-proper-distance orbit with the model held in frame, lensing and
/// all. There is no chart here in which a constant-radius circle would be a
/// circle, and aiming along the naive chord would simply miss.
///
/// During the approach the distance it is fired to is still shrinking back
/// towards the camera, so the same construction doubles as the pull-back.
fn pose_at(l: f64) -> Shot {
    let body = body_at(l);
    // Bearing zero fires the geodesic back down the path, which puts the camera
    // exactly where a chase camera would be: behind the model, looking the way
    // it is going. Everything after that is one long swing around to the same
    // place.
    let behind = -path_tangent(body.l);
    let bearing = bearing_at(l);
    let e = rotate_about(behind, V3::z(), bearing);
    let (cl, cu, at_model) = orbit_camera(&body, e, dist_at(l));
    // Facing straight back down the geodesic holds the model in the middle of
    // the frame, which is what an orbit should do -- but held for the whole
    // flight it also parks the model right in front of the wormhole mouth and
    // eclipses it. So only look at the model near the throat, and look where
    // the flight is going the rest of the time.
    let w = (0.5 * bearing).sin().abs().powf(LOOK_GRIP);
    // "The way the flight is going", evaluated at the camera's *own* position.
    // Taking it from the path instead only works while the camera is still on
    // the path; once the orbit has carried it round, the path's tangent is a
    // direction belonging to some other point entirely, and aiming along it
    // points the camera at empty sky. -u is the way l decreases at any point,
    // which is the direction of travel everywhere, both universes included.
    let psi_cam = cu.y.atan2(cu.x);
    let e_psi = V3::new(-psi_cam.sin(), psi_cam.cos(), 0.0);
    let along = (-cu + e_psi * (sweep_k() * A * A / (cl * cl + A * A))).normalize();
    let axis = along.cross(&at_model);
    let mut fwd = if axis.norm() > 1e-9 {
        let angle = along.dot(&at_model).clamp(-1.0, 1.0).acos();
        rotate_about(along, axis.normalize(), w * angle)
    } else {
        at_model
    };

    // Tilt up onto the model's head for the close approach. Rotating about
    // (fwd x up) is what swings the view towards `up`: for a small angle the
    // rotation moves fwd by (fwd x up) x fwd, which is exactly up.
    let aim = aim_at(l);
    if aim.abs() > 1e-9 {
        let axis = fwd.cross(&up_for(fwd));
        if axis.norm() > 1e-9 {
            fwd = rotate_about(fwd, axis.normalize(), (aim / dist_at(l)).atan());
        }
    }

    Shot {
        cam: Camera {
            l: cl,
            u: cu,
            fwd,
            up: up_for(fwd),
        },
        body,
    }
}

/// How much the view changes between two poses: turning, plus translation
/// counted at one radian per `radius`. This is the quantity the frames are
/// spaced evenly in, and it is why the flight comes out smooth.
///
/// `radius` is the camera's current distance from the model rather than a fixed
/// scale, so a metre of travel counts for more when the camera is close. That
/// is what makes the pull-back start slowly instead of leaping away from her
/// face in the first few frames.
fn view_change(a: &Camera, b: &Camera, radius: f64) -> f64 {
    let moved = offset_from(a.l, a.u, b.l, b.u).norm() / radius;
    let turned = a.fwd.dot(&b.fwd).clamp(-1.0, 1.0).acos();
    (moved * moved + turned * turned).sqrt()
}

/// Fraction of the flight spent ramping up to speed at the start, and back down
/// at the end.
pub const END_RAMP: f64 = 0.18;

/// Eased progress through the flight: a gentle ramp in and out at the two ends,
/// steady in between.
///
/// The obvious easing -- a half cosine over the whole flight -- has its peak
/// speed exactly at the midpoint, which is precisely where the throat crossing
/// wants to be slowest. It was the single largest source of the leftover jerk.
/// Ramping only over the first and last END_RAMP keeps the gentle start and
/// stop without stealing from the middle.
///
/// The velocity is a smoothstep over each ramp and flat between, so this is its
/// integral, normalised to land on exactly 1.
fn eased(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let r = END_RAMP.clamp(1e-6, 0.5);
    // Integral of the smoothstep 3x^2 - 2x^3, which comes to 1/2 over [0, 1].
    let ramp = |x: f64| x * x * x - 0.5 * x * x * x * x;
    let total = 1.0 - r;
    let d = if t <= r {
        r * ramp(t / r)
    } else if t < 1.0 - r {
        t - 0.5 * r
    } else {
        total - r * ramp((1.0 - t) / r)
    };
    d / total
}

/// Extra frames to spend near the throat, over and above what even motion
/// already asks for.
///
/// Spacing the frames evenly in view change already slows the crossing down a
/// long way by itself, because the lensing there swings the view about ten
/// times faster per unit of l than it does out in the open. This weights the
/// measure on top of that: at the throat a unit of view change is treated as
/// THROAT_DWELL units, so it gets that many times more frames. 1.0 disables it.
pub const THROAT_DWELL: f64 = 1.6;

// ------------------------------- flight plan -------------------------------

/// Total frames written.
pub const TOTAL_FRAMES: usize = INTRO_FRAMES + FRAMES;

/// The whole flight, as one continuous move.
///
/// Nothing comes to a stop and there is nothing to stitch: `l` runs
/// monotonically from L_INTRO_START to L_END while the camera withdraws from
/// the model's face to its final distance and then swings ORBIT_TURNS all the
/// way around it.
///
/// What makes it smooth is that the frames are not spaced evenly in `l`, nor
/// evenly in bearing. `pose_at` defines the *curve* -- a purely geometric
/// object with no timing in it -- and then the curve is walked once, measuring
/// how much the view actually changes along it, and the frames are placed at
/// equal intervals of that. Near the throat the lensing swings the view about
/// ten times faster per unit of l than it does out in the open, so equal steps
/// in l would tear through the crossing and crawl through the rest; equal steps
/// in view change do the opposite, which is exactly the heavy slowdown in the
/// middle that the crossing wants. Every frame is still an exact pose on the
/// true curve -- only the timing along it changes.
///
/// Both the live viewer and the headless renderer walk this same list, so what
/// you see in the window is exactly what gets written to disk.
pub fn flight_plan() -> Vec<Shot> {
    // Walk the curve once, recording per-interval how much the view moved and
    // how much of that belongs to the approach. Doing it in one pass matters:
    // `pose_at` integrates a geodesic, so it is much the most expensive thing
    // here, and the pacing solve below needs to re-weight these numbers several
    // times over.
    let ls: Vec<f64> = (0..=CURVE_SAMPLES)
        .map(|i| l_sample(i as f64 / CURVE_SAMPLES as f64))
        .collect();
    let mut step = Vec::with_capacity(ls.len()); // view change of each interval
    let mut share = Vec::with_capacity(ls.len()); // how "approach" each one is
    let mut prev = pose_at(ls[0]).cam;
    for w in ls.windows(2) {
        let (l0, l1) = (w[0], w[1]);
        let cam = pose_at(l1).cam;
        let mid = 0.5 * (l0 + l1);
        // Weighted so the throat can be given more frames than even motion
        // alone would hand it.
        let dwell = 1.0 + (THROAT_DWELL - 1.0) / (1.0 + (mid / ORBIT_SPREAD).powi(2));
        step.push(dwell * view_change(&prev, &cam, dist_at(mid)));
        share.push(intro_t(mid));
        prev = cam;
    }

    // Pace the approach so it gets the frames it was asked for.
    //
    // Left alone, the split would fall out of whatever the geometry happened to
    // measure, which is no way to tune a shot. Scaling the approach's share of
    // the measure by `k` moves frames into or out of it, and because the scale
    // is applied through the same smootherstep the distance ramp uses, it is
    // 1 exactly at the join -- no speed step there. The measure is linear in
    // `k`, so the right value is a closed form rather than a search.
    // Unweighted measure either side of the join, and the part of the approach
    // that `k` actually scales. `share` is zero exactly where the crossing
    // begins, so these three partition the curve.
    let approach: f64 = step
        .iter()
        .zip(&share)
        .filter(|(_, sh)| **sh > 0.0)
        .map(|(s, _)| *s)
        .sum();
    let crossing: f64 = step
        .iter()
        .zip(&share)
        .filter(|(_, sh)| **sh <= 0.0)
        .map(|(s, _)| *s)
        .sum();
    let weighted: f64 = step.iter().zip(&share).map(|(s, sh)| s * sh).sum();
    // Fraction of the *timeline* the approach should occupy. Taken through the
    // easing, since that is what maps frame number to distance travelled, and
    // half a frame in so the boundary lands cleanly between two frames.
    let want_frac = eased((INTRO_FRAMES as f64 - 0.5) / (TOTAL_FRAMES - 1) as f64);
    let k = if weighted > 1e-12 && want_frac < 1.0 && INTRO_FRAMES > 0 {
        (1.0 + (want_frac * crossing / (1.0 - want_frac) - approach) / weighted).clamp(0.02, 200.0)
    } else {
        1.0
    };

    let mut cumulative = Vec::with_capacity(ls.len());
    cumulative.push(0.0);
    let mut seen = 0.0;
    for (s, sh) in step.iter().zip(&share) {
        seen += s * (1.0 + (k - 1.0) * sh);
        cumulative.push(seen);
    }

    // Place the frames at equal intervals of that, eased at both ends so the
    // flight opens and closes gently rather than starting at full speed.
    let mut shots: Vec<Shot> = Vec::with_capacity(TOTAL_FRAMES);
    let mut i = 0;
    for f in 0..TOTAL_FRAMES {
        let t = f as f64 / (TOTAL_FRAMES - 1) as f64;
        let want = seen * eased(t);
        while i + 1 < CURVE_SAMPLES && cumulative[i + 1] < want {
            i += 1;
        }
        let (lo, hi) = (cumulative[i], cumulative[i + 1]);
        let frac = if hi > lo { (want - lo) / (hi - lo) } else { 0.0 };
        shots.push(pose_at(ls[i] + frac * (ls[i + 1] - ls[i])));
    }

    // The model can turn steadily on its own axis over the flight; at
    // MODEL_TURNS = 0 it simply holds still.
    if MODEL_TURNS != 0.0 {
        let frames = shots.len();
        for (f, shot) in shots.iter_mut().enumerate() {
            shot.body.spin = MODEL_TURNS * 2.0 * PI * f as f64 / frames as f64;
        }
    }
    shots
}

/// Report how evenly the flight moves.
///
/// A cut, or a leftover seam between two differently-parametrised stretches,
/// shows up as one frame that turns or travels much further than its
/// neighbours. So print the worst of each against the average: with the flight
/// built as a single move these should now sit close together, and the peak
/// should be somewhere in the middle of the throat crossing rather than at a
/// particular frame index where two pieces used to meet.
pub fn report_continuity(shots: &[Shot]) {
    // Where the approach actually ended, and how the camera got there. These
    // are the numbers to watch when tuning INTRO_FRAMES / L_INTRO_START /
    // INTRO_DIST_START, since the pacing solve only aims at the frame split.
    // The curve is parametrised by the trailing coordinate `l`; the model rides
    // MODEL_LEAD ahead of it, so recover the parameter before asking the ramps
    // about it or the two disagree by exactly that lead.
    let param = |s: &Shot| s.body.l + MODEL_LEAD;
    let intro = shots
        .iter()
        .take_while(|s| intro_t(param(s)) > 0.0)
        .count()
        .clamp(1, shots.len() - 1);
    eprintln!(
        "approach: {} of {} frames, model l {:+.1} -> {:+.1}, camera {:.2} -> {:.2} out \
         (asked for {})",
        intro,
        shots.len(),
        shots[0].body.l,
        shots[intro].body.l,
        dist_at(param(&shots[0])),
        dist_at(param(&shots[intro])),
        INTRO_FRAMES,
    );
    if INTRO_DIST_START <= MODEL_R {
        eprintln!(
            "  warning: INTRO_DIST_START {:.2} is inside the model's bounding radius {:.2}; \
             the camera may start inside it",
            INTRO_DIST_START, MODEL_R
        );
    }

    let turn = |a: &Camera, b: &Camera| a.fwd.dot(&b.fwd).clamp(-1.0, 1.0).acos().to_degrees();
    let (mut worst_turn, mut turn_at, mut turn_sum) = (0.0f64, 0, 0.0);
    let (mut worst_move, mut move_at, mut move_sum) = (0.0f64, 0, 0.0);
    for (f, w) in shots.windows(2).enumerate() {
        let d = turn(&w[0].cam, &w[1].cam);
        turn_sum += d;
        if d > worst_turn {
            worst_turn = d;
            turn_at = f;
        }
        // Proper distance the camera actually covered between the two frames.
        let m = offset_from(w[0].cam.l, w[0].cam.u, w[1].cam.l, w[1].cam.u).norm();
        move_sum += m;
        if m > worst_move {
            worst_move = m;
            move_at = f;
        }
    }
    let n = (shots.len() - 1) as f64;
    eprintln!(
        "camera: turns {:.1} deg/frame on average, at most {:.1} deg (frames {} -> {}); \
         moves {:.3}/frame, at most {:.3} (frames {} -> {})",
        turn_sum / n,
        worst_turn,
        turn_at,
        turn_at + 1,
        move_sum / n,
        worst_move,
        move_at,
        move_at + 1,
    );

    // Where the model actually sits in frame. Keeping it there is the whole
    // job of the bearing-keyed look blend, and it is the thing that silently
    // broke when the blend was keyed to l instead -- the model simply left the
    // shot for a third of the flight. Measured in the model's flat
    // neighbourhood, so it reads a little off where the lensing is strongest,
    // but more than enough to catch it walking out of frame.
    let (mut worst_off, mut off_at) = (0.0f64, 0);
    for (f, s) in shots.iter().enumerate() {
        let to_model = offset_from(s.cam.l, s.cam.u, s.body.l, s.body.u);
        if to_model.norm() > 1e-9 {
            let off = s
                .cam
                .fwd
                .dot(&to_model.normalize())
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees();
            if off > worst_off {
                worst_off = off;
                off_at = f;
            }
        }
    }
    eprintln!(
        "framing: model stays within {:.0} deg of centre (worst at frame {}), \
         against a half-FOV of {:.0} deg",
        worst_off,
        off_at,
        0.5 * FOV_DEG
    );

    // The model's own pace along l, which is the thing that is meant to dip in
    // the middle without reaching zero. Measured against the fastest stretch of
    // the flight rather than against the slowest: the slowest steps of all are
    // the first and last, where the flight is deliberately ramping up to speed,
    // and comparing the throat with those would flatter it enormously.
    let steps: Vec<f64> = shots
        .windows(2)
        .map(|w| (w[0].body.l - w[1].body.l).abs())
        .collect();
    let cruise = steps.iter().copied().fold(0.0, f64::max);
    // The step straddling the throat, i.e. the one whose midpoint sits closest
    // to l = 0.
    let throat_i = (0..steps.len())
        .min_by(|&i, &j| {
            let m = |k: usize| (shots[k].body.l + shots[k + 1].body.l).abs();
            m(i).total_cmp(&m(j))
        })
        .unwrap_or(0);
    let throat = steps[throat_i];
    eprintln!(
        "model: crosses the throat at {:.3} in l per frame against {:.3} at full \
         cruise ({:.1}x slower, never stopped), camera turns {:.2} of a circle",
        throat,
        cruise,
        cruise / throat.max(1e-12),
        ORBIT_TURNS,
    );
}
