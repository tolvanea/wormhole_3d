//! Ellis wormhole fly-through renderer.
//!
//! Spatial metric (no time, no gravity):
//!     ds^2 = dl^2 + (l^2 + a^2) (dtheta^2 + sin^2 theta dphi^2)
//! Embedding profile of the throat: r = a cosh(z/a)  (catenoid).
//!
//! Every light ray is a geodesic of this metric. By spherical symmetry each
//! geodesic lies in a plane through the "center", so per ray we integrate the
//! planar system (l, phi, v_l, v_phi):
//!
//!     dl/ds     = v_l
//!     dphi/ds   = v_phi
//!     dv_l/ds   = l * v_phi^2
//!     dv_phi/ds = -2 l / (l^2 + a^2) * v_l * v_phi
//!
//! with classic RK4. Rays escaping to l -> +inf see "universe A", rays
//! escaping to l -> -inf see "universe B" (two different cube-map skies read
//! from `cubemap/`). Rays that never escape (asymptotically winding on the
//! throat's photon ring) are painted black.

use image::codecs::gif::{GifEncoder, Repeat};
use image::{Delay, Frame, Rgb, RgbImage, Rgba, RgbaImage};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs;
use std::sync::OnceLock;

type V3 = Vector3<f64>;

// ----------------------------- configuration -----------------------------

const A: f64 = 1.0; // throat radius (sets the length unit)

const WIDTH: u32 = 1920; // output image width
const HEIGHT: u32 = 1080; // output image height
const SSAA: u32 = 2; // supersampling factor (2 => 4 rays per pixel)
const FOV_DEG: f64 = 80.0; // horizontal field of view

const L_START: f64 = 14.0; // camera starts here (universe A) ...
const L_END: f64 = -14.0; // ... and ends here (universe B)

// The camera does not fall straight down the axis: while it crosses it also
// slides around the throat sphere by LOOP_SWEEP radians (half a turn), almost
// all of it within a few throat radii of l = 0 -- a half loop inside the walls.
const LOOP_SWEEP: f64 = PI;

const PATH_BIAS: f64 = 1.6; // >1 puts more frames near the throat, where the
// turn happens (l ~ (1 - eased time)^PATH_BIAS)

const FRAMES_IN: usize = 45; // universe A -> throat
const FRAMES_ORBIT: usize = 36; // parked at the throat, circling the sphere
const FRAMES_OUT: usize = 45; // throat -> universe B
const ORBIT_TURNS: f64 = 1.0; // times the camera circles the sphere
const ORBIT_LOCK: f64 = 1.0; // radians of orbit spent settling from the flight
// heading onto the sphere, and again at the end returning to it

// A solid sphere flies the same path ahead of the camera. l decreases
// monotonically over the whole flight, so "ahead" is simply l - SPHERE_LEAD.
const SPHERE_R: f64 = 0.25 * A; // proper radius of the chase sphere
const SPHERE_LEAD: f64 = 2.5 * A; // how far ahead of the camera it flies
const SPHERE_RISE: f64 = 0.7 * A; // and how far it flies above the flight plane,
// so that it does not eclipse the wormhole mouth on the way in
const SPHERE_LIGHT: [f64; 3] = [0.35, -0.25, 0.90]; // key light, sphere's frame
const SPHERE_TURNS: f64 = 4.0; // turns it makes about its own z axis, whole flight
// How much the camera leans its heading toward the sphere: 0 follows its own
// tangent (sphere drifts freely), ~0.5 is the chord to the sphere (locked on).
const AIM_LEAD: f64 = 0.45;

const L_ESCAPE: f64 = 25.0 * A; // |l| beyond which the ray is "at infinity"
const H0: f64 = 0.02; // base integration step (scaled by sqrt(l^2+a^2))
const MAX_STEPS: usize = 40_000; // safety cap; exceeded => photon ring, black

const GIF_EVERY: usize = 1; // put every n-th frame into the GIF
const GIF_DELAY_MS: u32 = 50; // 20 fps

const SKY_A: &str = "cubemap/cubemap_1.jpg"; // sky seen at l -> +inf
const SKY_B: &str = "cubemap/cubemap_2.png"; // sky seen at l -> -inf
const SKY_GAIN: f64 = 2.0; // exposure applied to the sampled cube maps

// --------------------------- geodesic integrator --------------------------

/// Planar geodesic state: (l, phi, dl/ds, dphi/ds).
#[derive(Clone, Copy)]
struct GState {
    l: f64,
    phi: f64,
    vl: f64,
    vphi: f64,
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
fn rk4_step(s: &GState, h: f64) -> GState {
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
/// in its own geodesic plane, spanned by (p_hat, t_hat), and the plane's
/// tangential basis vector rotated along with it.
fn u_at(p_hat: V3, t_hat: V3, phi: f64) -> V3 {
    p_hat * phi.cos() + t_hat * phi.sin()
}

fn t_at(p_hat: V3, t_hat: V3, phi: f64) -> V3 {
    -p_hat * phi.sin() + t_hat * phi.cos()
}

/// Some unit vector perpendicular to `u`.
fn any_perp(u: V3) -> V3 {
    let a = if u.z.abs() < 0.9 { V3::z() } else { V3::x() };
    (a - u * a.dot(&u)).normalize()
}

/// Direction of travel in map axes, from a planar state and its plane basis.
fn heading(s: &GState, p_hat: V3, t_hat: V3) -> V3 {
    let r = (s.l * s.l + A * A).sqrt();
    (u_at(p_hat, t_hat, s.phi) * s.vl + t_at(p_hat, t_hat, s.phi) * (r * s.vphi)).normalize()
}

/// How a ray ended.
enum Stop {
    /// Struck the chase sphere; carries the offset of the hit point from its
    /// centre (length = sphere radius, direction = outward surface normal).
    Hit(V3),
    /// Escaped to |l| > L_ESCAPE, as (l, phi, v_l, v_phihat).
    Escaped(f64, f64, f64, f64),
    /// Still winding on the photon ring when the step budget ran out.
    Trapped,
}

/// Integrate a ray starting at radial coordinate l0 with initial velocity
/// components (v_l, v_phihat) in the local orthonormal frame (unit speed:
/// v_l^2 + v_phihat^2 = 1), in the geodesic plane spanned by (p_hat, t_hat).
fn trace(l0: f64, p_hat: V3, t_hat: V3, vl0: f64, vphihat0: f64, sph: &Sphere) -> Stop {
    let mut s = GState {
        l: l0,
        phi: 0.0,
        vl: vl0,
        vphi: vphihat0 / (l0 * l0 + A * A).sqrt(),
    };
    for _ in 0..MAX_STEPS {
        let r = (s.l * s.l + A * A).sqrt();
        if s.l.abs() > L_ESCAPE && s.l * s.vl > 0.0 {
            return Stop::Escaped(s.l, s.phi, s.vl, r * s.vphi);
        }
        // Step size grows with distance from the throat: the geometry (and
        // the bending) only has structure on scales ~ sqrt(l^2+a^2).
        let mut h = H0 * r;

        // Is the sphere close enough to be worth a full distance test? The
        // radial reject is a single comparison and throws out almost every
        // ray; including h in the band means no step can jump clean across it
        // (the ray has unit speed, so one step advances arc length h).
        let near = (s.l - sph.l).abs() < 6.0 * SPHERE_R + h
            && sph.offset(s.l, u_at(p_hat, t_hat, s.phi)).norm() < 6.0 * SPHERE_R + h;
        if near {
            h = h.min(0.15 * SPHERE_R); // do not tunnel through the surface
        }

        let next = rk4_step(&s, h);
        if near {
            let o = sph.offset(next.l, u_at(p_hat, t_hat, next.phi));
            if o.norm() < sph.r {
                // Entered the sphere during this step: bisect it to land on
                // the surface, so the silhouette and the shading are smooth.
                let (mut lo, mut hi, mut hit) = (0.0, h, o);
                for _ in 0..8 {
                    let mid = 0.5 * (lo + hi);
                    let m = rk4_step(&s, mid);
                    let om = sph.offset(m.l, u_at(p_hat, t_hat, m.phi));
                    if om.norm() < sph.r {
                        hi = mid;
                        hit = om;
                    } else {
                        lo = mid;
                    }
                }
                return Stop::Hit(hit);
            }
        }
        s = next;
    }
    Stop::Trapped
}

// ------------------------------- chase sphere ------------------------------

/// First-order proper offset of the manifold point (l, u) from (l0, u0), as a
/// vector in map axes: the radial part along u0 plus the tangential part along
/// the great circle from u0 towards u, scaled to a proper length by the radius
/// midway between them. The two are orthogonal, so |O| is the proper distance.
///
/// The tangential part uses the arc, not the chord |u - u0 (u . u0)|: the chord
/// vanishes at the antipode of u0 as well as at u0 itself, which would read as
/// a second, spurious coincidence for rays that have wound half way round.
fn offset_from(l0: f64, u0: V3, l: f64, u: V3) -> V3 {
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

/// A solid sphere sitting on the flight path.
///
/// It is small compared with the throat radius, so its neighbourhood can be
/// treated as flat: the offset of a manifold point (l, u) from the centre
/// (l_c, u_c) is, to first order,
///
///     O = (l - l_c) u_c + r_mid * (u - u_c (u . u_c))
///
/// with r_mid = r at the midpoint. The two terms are orthogonal and |O| is the
/// proper distance, which is all the intersection test needs -- no global
/// Cartesian chart is involved, so this stays valid right through the throat.
/// O doubles as the outward surface normal.
#[derive(Clone, Copy)]
struct Sphere {
    l: f64,    // radial coordinate of the centre
    u: V3,     // angular position of the centre
    r: f64,    // proper radius
    spin: f64, // rotation about its own z axis (turns the pattern, not the light)
}

impl Sphere {
    /// A sphere no ray can reach, for traces that only want the sky.
    fn none() -> Sphere {
        Sphere {
            l: f64::INFINITY,
            u: V3::new(1.0, 0.0, 0.0),
            r: 0.0,
            spin: 0.0,
        }
    }

    fn offset(&self, l: f64, u: V3) -> V3 {
        offset_from(self.l, self.u, l, u)
    }

    /// Checkerboard albedo plus a Lambert key light and a flat ambient term.
    /// The pattern is what makes the lensed copies of the sphere legible; the
    /// light is fixed in the sphere's own frame, so it stays lit the same way
    /// all the way round (a distant light source would be a fiction here).
    fn shade(&self, o: V3) -> [f64; 3] {
        let n = o.normalize();
        // Body frame: outward radial, then the two directions across it. b3 is
        // the sphere's z axis (the map's +z while it rides the flight plane).
        let b2 = V3::new(-self.u.y, self.u.x, 0.0).normalize();
        let b3 = self.u.cross(&b2);
        let (nx, ny, nz) = (n.dot(&self.u), n.dot(&b2), n.dot(&b3));
        // The sphere turns about b3, so the pattern's longitude shifts with it
        // while the light -- fixed in the body frame -- does not. 12 columns is
        // an even count, so the wrap-around of the shifted longitude is seamless.
        let lon = ny.atan2(nx) - self.spin;
        let tile = (lon / PI * 6.0).floor() + (nz.clamp(-1.0, 1.0).asin() / PI * 6.0).floor();
        let albedo = if (tile as i64).rem_euclid(2) == 0 {
            [0.85, 0.28, 0.10]
        } else {
            [0.92, 0.90, 0.86]
        };
        let ld = V3::new(SPHERE_LIGHT[0], SPHERE_LIGHT[1], SPHERE_LIGHT[2]).normalize();
        let k = 0.25 + 1.15 * (nx * ld.x + ny * ld.y + nz * ld.z).max(0.0);
        [albedo[0] * k, albedo[1] * k, albedo[2] * k]
    }
}

// ----------------------- looking at the sphere, exactly ---------------------

const PROBE_STEPS: usize = 6_000;

/// Closest approach to `target`'s centre of the geodesic that leaves
/// (l0, p_hat) at angle `theta` from the outward radial, in the plane
/// (p_hat, t_hat), as (miss distance, arc length there).
fn probe(l0: f64, p_hat: V3, t_hat: V3, theta: f64, target: &Sphere) -> (f64, f64) {
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
fn sight_line(l0: f64, u0: V3, target: &Sphere) -> (f64, V3) {
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

/// A camera standing on the geodesic sphere of radius `dist` about the ball:
/// shoot a geodesic out of the ball in direction `e` and stand at its far end
/// facing back down it. Geodesics are reversible, so the ball then sits in the
/// middle of the frame however badly the space in between is bending the light.
/// Returns (l, u, forward).
fn orbit_camera(sph: &Sphere, e: V3, dist: f64) -> (f64, V3, V3) {
    let p_hat = sph.u;
    let vl0 = e.dot(&p_hat);
    let tan = e - p_hat * vl0;
    let (t_hat, vph0) = if tan.norm() < 1e-12 {
        (any_perp(p_hat), 0.0)
    } else {
        (tan.normalize(), tan.norm())
    };
    let mut s = GState {
        l: sph.l,
        phi: 0.0,
        vl: vl0,
        vphi: vph0 / (sph.l * sph.l + A * A).sqrt(),
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
fn orbit_schedule(sph: &Sphere, e0: V3, dist: f64, total: f64, frames: usize) -> Vec<f64> {
    let n = 2000;
    let mut cumulative = Vec::with_capacity(n + 1);
    let mut seen = 0.0;
    let mut prev: Option<(f64, V3, V3)> = None;
    for i in 0..=n {
        let a = total * i as f64 / n as f64;
        let (l, u, fwd) = orbit_camera(sph, rotate_about(e0, V3::z(), a), dist);
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

// -------------------------------- cube maps -------------------------------

/// One cube face, stored as linear-light RGB with the origin in its top-left
/// corner (row-major, `size` texels per side).
struct Face {
    size: usize,
    px: Vec<[f32; 3]>,
}

impl Face {
    /// Bilinear fetch. `(s, t)` are in [0,1] with `t` running downward.
    fn sample(&self, s: f64, t: f64) -> [f64; 3] {
        let n = self.size as f64;
        let x = (s * n - 0.5).clamp(0.0, n - 1.0);
        let y = (t * n - 0.5).clamp(0.0, n - 1.0);
        let (fx, fy) = (x.fract(), y.fract());
        let (x0, y0) = (x as usize, y as usize);
        let x1 = (x0 + 1).min(self.size - 1);
        let y1 = (y0 + 1).min(self.size - 1);
        let (a, b) = (&self.px[y0 * self.size + x0], &self.px[y0 * self.size + x1]);
        let (c, d) = (&self.px[y1 * self.size + x0], &self.px[y1 * self.size + x1]);
        let mut out = [0.0f64; 3];
        for k in 0..3 {
            let top = a[k] as f64 * (1.0 - fx) + b[k] as f64 * fx;
            let bot = c[k] as f64 * (1.0 - fx) + d[k] as f64 * fx;
            out[k] = top * (1.0 - fy) + bot * fy;
        }
        out
    }
}

/// Cell `(column, row)` of each face inside the 4x3 cross, in the order
/// +X, -X, +Y, -Y, +Z, -Z. The cross sits on the left of the image: its
/// vertical arm is column 1, and the -Z face trails off to the right.
///
///     .   +Y  .   .
///     -X  +Z  +X  -Z
///     .   -Y  .   .
const FACE_CELLS: [(u32, u32); 6] = [(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)];

struct CubeMap {
    faces: [Face; 6],
}

impl CubeMap {
    /// Cut a cross-form image into its six faces (sRGB is linearized here so
    /// that filtering and the tone map both work in linear light).
    fn load(path: &str) -> CubeMap {
        let img = image::open(path)
            .unwrap_or_else(|e| panic!("cannot read {path}: {e}"))
            .to_rgb8();
        let (w, h) = img.dimensions();
        assert!(
            w % 4 == 0 && h % 3 == 0 && w / 4 == h / 3,
            "{path}: expected a 4x3 cross of square faces, got {w}x{h}"
        );
        let size = (w / 4) as usize;
        let faces = FACE_CELLS.map(|(col, row)| {
            let (ox, oy) = (col * size as u32, row * size as u32);
            let mut px = Vec::with_capacity(size * size);
            for y in 0..size as u32 {
                for x in 0..size as u32 {
                    let p = img.get_pixel(ox + x, oy + y);
                    px.push([
                        srgb_to_linear(p[0]),
                        srgb_to_linear(p[1]),
                        srgb_to_linear(p[2]),
                    ]);
                }
            }
            Face { size, px }
        });
        CubeMap { faces }
    }

    /// Standard cube-map lookup: pick the face by the dominant axis, then
    /// project onto it.
    fn sample(&self, d: V3) -> [f64; 3] {
        let (ax, ay, az) = (d.x.abs(), d.y.abs(), d.z.abs());
        let (face, sc, tc, ma) = if ax >= ay && ax >= az {
            if d.x > 0.0 {
                (0, -d.z, -d.y, ax)
            } else {
                (1, d.z, -d.y, ax)
            }
        } else if ay >= az {
            if d.y > 0.0 {
                (2, d.x, d.z, ay)
            } else {
                (3, d.x, -d.z, ay)
            }
        } else if d.z > 0.0 {
            (4, d.x, -d.y, az)
        } else {
            (5, -d.x, -d.y, az)
        };
        self.faces[face].sample(0.5 * (sc / ma + 1.0), 0.5 * (tc / ma + 1.0))
    }
}

fn srgb_to_linear(v: u8) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

static SKIES: OnceLock<[CubeMap; 2]> = OnceLock::new();

fn skies() -> &'static [CubeMap; 2] {
    SKIES.get_or_init(|| [CubeMap::load(SKY_A), CubeMap::load(SKY_B)])
}

/// Sky of the universe the ray escaped into. side = +1 (A) or -1 (B).
fn sky(n: V3, side: i32) -> [f64; 3] {
    // Map axes put "up" on +z, the cube maps are authored with +y up; the
    // cyclic swap (x,y,z) -> (y,z,x) is the rotation that lines them up.
    let c = skies()[if side > 0 { 0 } else { 1 }].sample(V3::new(n.y, n.z, n.x));
    [c[0] * SKY_GAIN, c[1] * SKY_GAIN, c[2] * SKY_GAIN]
}

// ------------------------------- camera path -------------------------------

/// Where the camera is and where it points.
///
/// A point of the manifold is (l, u) with u a unit vector on the sphere of
/// radius r(l) = sqrt(l^2 + a^2); the local orthonormal frame is u itself
/// (outward radial) plus the two directions perpendicular to it, so any
/// viewing direction is simply a unit 3-vector in these "map" axes.
#[derive(Clone, Copy)]
struct Camera {
    l: f64,  // radial coordinate
    u: V3,   // angular position on the sphere
    fwd: V3, // viewing direction (unit)
    up: V3,  // screen up (unit, perpendicular to fwd)
}

/// Rodrigues rotation of `v` about the unit axis `k` by `ang`.
fn rotate_about(v: V3, k: V3, ang: f64) -> V3 {
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

fn sweep_k() -> f64 {
    LOOP_SWEEP / (sweep_g(L_START) - sweep_g(L_END))
}

fn path_psi(l: f64) -> f64 {
    sweep_k() * (sweep_g(L_START) - sweep_g(l))
}

/// The chase sphere for a camera at radial coordinate `l`: same path, further
/// along it.
fn sphere_at(l: f64) -> Sphere {
    let ls = l - SPHERE_LEAD;
    let psi = path_psi(ls);
    // Tilted out of the flight plane by a proper distance SPHERE_RISE, which
    // at radius r is an angle SPHERE_RISE / r (so it barely leaves the axis
    // far away and rides well clear of it near the throat).
    let rise = SPHERE_RISE / (ls * ls + A * A).sqrt();
    Sphere {
        l: ls,
        u: V3::new(psi.cos(), psi.sin(), 0.0) * rise.cos() + V3::new(0.0, 0.0, rise.sin()),
        r: SPHERE_R,
        spin: 0.0,
    }
}

/// Camera flying the path at radial coordinate `l`.
///
/// The heading is the path tangent evaluated at `l_aim` rather than at `l`,
/// then re-expressed in the camera's own (radial, psi) basis. `l_aim = l`
/// gives a pure tangent follower; pulling `l_aim` toward the chase sphere
/// swings the heading into the turn, and since the tangent at the midpoint of
/// an arc is its chord, `l_aim` half way to the sphere aims roughly at it.
fn camera_on_path(l: f64, l_aim: f64) -> Camera {
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
fn up_for(fwd: V3) -> V3 {
    (V3::z() - fwd * fwd.z).normalize()
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

// ------------------------------- ray casting -------------------------------

/// Color of one camera ray. `lc` is the camera's radial coordinate, `u` its
/// angular position on the sphere and `d` the unit ray direction, both as
/// vectors in the "map" axes.
fn ray_color(lc: f64, u: V3, d: V3, sph: &Sphere) -> [f64; 3] {
    let p_hat = u; // radial unit vector at the camera
    let vl0 = d.dot(&u); // radial velocity component
    let dt = d - u * vl0; // tangential part
    let tlen = dt.norm();
    let (t_hat, vph0) = if tlen < 1e-12 {
        // Purely radial ray; the geodesic plane is arbitrary, take any normal.
        (any_perp(u), 0.0)
    } else {
        (dt / tlen, tlen)
    };

    match trace(lc, p_hat, t_hat, vl0, vph0, sph) {
        Stop::Trapped => [0.0, 0.0, 0.0], // photon ring: never escapes
        Stop::Hit(o) => sph.shade(o),
        Stop::Escaped(lf, phif, vlf, vphf) => {
            let side = if lf >= 0.0 { 1 } else { -1 };
            // Rotate the plane basis by the total swept angle phif.
            let p_f = u_at(p_hat, t_hat, phif);
            let t_f = t_at(p_hat, t_hat, phif);
            // Asymptotic straight-line direction in the escape universe
            // (radius grows like |l|, hence the sign on the radial part).
            let n = (p_f * (vlf * side as f64) + t_f * vphf).normalize();
            sky(n, side)
        }
    }
}

fn render_frame(cam: Camera, sph: &Sphere) -> RgbImage {
    let half = (FOV_DEG.to_radians() / 2.0).tan();
    let aspect = HEIGHT as f64 / WIDTH as f64;
    let right = cam.fwd.cross(&cam.up);
    let rows: Vec<Vec<[f64; 3]>> = (0..HEIGHT)
        .into_par_iter()
        .map(|j| {
            let mut row = Vec::with_capacity(WIDTH as usize);
            for i in 0..WIDTH {
                let mut acc = [0.0f64; 3];
                for sj in 0..SSAA {
                    for si in 0..SSAA {
                        let u = (i as f64 + (si as f64 + 0.5) / SSAA as f64) / WIDTH as f64;
                        let v = (j as f64 + (sj as f64 + 0.5) / SSAA as f64) / HEIGHT as f64;
                        let px = (2.0 * u - 1.0) * half;
                        let py = (1.0 - 2.0 * v) * half * aspect;
                        let d = (cam.fwd + right * px + cam.up * py).normalize();
                        let c = ray_color(cam.l, cam.u, d, sph);
                        acc[0] += c[0];
                        acc[1] += c[1];
                        acc[2] += c[2];
                    }
                }
                let ns = (SSAA * SSAA) as f64;
                row.push([acc[0] / ns, acc[1] / ns, acc[2] / ns]);
            }
            row
        })
        .collect();

    let mut img = RgbImage::new(WIDTH, HEIGHT);
    for (j, row) in rows.iter().enumerate() {
        for (i, c) in row.iter().enumerate() {
            img.put_pixel(i as u32, j as u32, Rgb(tonemap(*c)));
        }
    }
    img
}

fn tonemap(c: [f64; 3]) -> [u8; 3] {
    let mut out = [0u8; 3];
    for k in 0..3 {
        let t = 1.0 - (-c[k].max(0.0)).exp(); // soft exposure roll-off
        out[k] = ((t.powf(1.0 / 2.2)) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    }
    out
}

// ---------------------------------- main -----------------------------------

fn main() {
    fs::create_dir_all("out").unwrap();

    // Fail fast (and outside the render loop) if the cube maps are missing.
    let size = skies()[0].faces[0].size;
    eprintln!("cube maps: {SKY_A}, {SKY_B}  ({size}x{size} per face)");

    // Sanity check: unit-speed and angular-momentum conservation on one ray.
    {
        let (l0, vl0, vph0) = (7.0, -0.62, (1.0f64 - 0.62 * 0.62).sqrt());
        let (p, t) = (V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0));
        if let Stop::Escaped(lf, _phif, vlf, vphf) = trace(l0, p, t, vl0, vph0, &Sphere::none()) {
            let c = vlf * vlf + vphf * vphf; // should stay 1
            let lz0 = (l0 * l0 + A * A).sqrt() * vph0;
            let lzf = (lf * lf + A * A).sqrt() * vphf;
            eprintln!(
                "conservation check: |v|^2 = {:.9}, L = {:.9} -> {:.9}",
                c, lz0, lzf
            );
        }
    }

    let gif_file = fs::File::create("out/wormhole.gif").unwrap();
    let mut gif = GifEncoder::new_with_speed(gif_file, 10);
    gif.set_repeat(Repeat::Infinite).unwrap();

    // Flight plan: dive in (turning half way around the throat on the way),
    // park at l = 0 and circle the chase sphere, then resume the curve outward.
    let ease = |t: f64| 0.5 * (1.0 - (PI * t).cos()); // slow-in / slow-out
    // Fraction of the way out from the throat after an eased time t of a leg:
    // 1 at the far end, 0 at the throat, and with zero speed at both.
    let leg = |t: f64| (1.0 - ease(t)).powf(PATH_BIAS);
    // The camera chases the sphere: it leans AIM_LEAD of the way from its own
    // tangent toward the chord, so the sphere sits loosely in frame instead of
    // being pinned to the centre.
    let pose = |l: f64| camera_on_path(l, l - AIM_LEAD * SPHERE_LEAD);
    let mut shots: Vec<(Camera, Sphere)> = Vec::new();
    for f in 0..FRAMES_IN {
        // Decelerates into the throat; the l = 0 frame belongs to the orbit.
        let t = f as f64 / FRAMES_IN as f64;
        let l = L_START * leg(t);
        shots.push((pose(l), sphere_at(l)));
    }
    {
        // Parked at the throat, the camera circles the sphere rather than
        // spinning on the spot, so the sphere stays in frame throughout.
        let park = pose(0.0);
        let ball = sphere_at(0.0);
        // Solve for the geodesic that leaves the sphere and arrives at the
        // parked camera: its length is the orbit radius, its launch direction
        // is where the orbit starts. Searching outwards from the sphere rather
        // than inwards from the camera puts the residual error at the camera
        // end, instead of letting the lensing in between amplify it.
        let eye = Sphere {
            l: park.l,
            u: park.u,
            r: SPHERE_R,
            spin: 0.0,
        };
        let (dist, e0) = sight_line(ball.l, ball.u, &eye);
        let start = orbit_camera(&ball, e0, dist);
        eprintln!(
            "orbit: radius {:.3}, starts {:.3} from the parked camera, \
             {:.1} deg off its heading",
            dist,
            eye.offset(start.0, start.1).norm(),
            start.2.dot(&park.fwd).acos().to_degrees()
        );
        // The flight heading and the line to the sphere differ by a fixed
        // rotation. Shedding it over the first radians of the orbit (and
        // taking it back on over the last) lets the dive and the exit -- still
        // tangent followers -- join up cleanly, while never swinging further
        // than that from the sphere. Blending towards the parked heading as a
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
        let schedule = orbit_schedule(&ball, e0, dist, total, FRAMES_ORBIT);
        for &a in &schedule {
            let (l, u, fwd_ball) = orbit_camera(&ball, rotate_about(e0, V3::z(), a), dist);
            // Shed the lean over the first radians of the orbit and take it back
            // on over the last. Measured in orbit angle, not frames, so that it
            // is fully off at a = 0 and fully back on as a returns to 2*pi --
            // which is exactly where the two joins with the flight are.
            let w = smoothstep(a.min(total - a) / ORBIT_LOCK);
            let axis = rotate_about(lean_axis, V3::z(), a);
            let fwd = rotate_about(fwd_ball, axis, (1.0 - w) * lean_angle);
            shots.push((
                Camera {
                    l,
                    u,
                    fwd,
                    up: up_for(fwd),
                },
                ball,
            ));
        }
    }
    for f in 0..FRAMES_OUT {
        // Mirror image of the dive: accelerates away from the throat.
        let t = (f + 1) as f64 / FRAMES_OUT as f64;
        let l = L_END * leg(1.0 - t);
        shots.push((pose(l), sphere_at(l)));
    }
    let frames = shots.len();
    // The sphere turns steadily on its own axis for the whole flight, so it is
    // visibly spinning even while the camera is parked and circling it.
    for (f, (_, sph)) in shots.iter_mut().enumerate() {
        sph.spin = SPHERE_TURNS * 2.0 * PI * f as f64 / frames as f64;
    }

    // A cut shows up as one frame that turns much further than its neighbours,
    // so report the worst of them: it should stay in the same league as the
    // average, and in particular not spike at the two joins with the flight.
    {
        let turn = |a: &Camera, b: &Camera| a.fwd.dot(&b.fwd).clamp(-1.0, 1.0).acos().to_degrees();
        let (mut worst, mut at, mut sum) = (0.0f64, 0, 0.0);
        for (f, w) in shots.windows(2).enumerate() {
            let d = turn(&w[0].0, &w[1].0);
            sum += d;
            if d > worst {
                worst = d;
                at = f;
            }
        }
        let join = |f: usize| turn(&shots[f].0, &shots[f + 1].0);
        eprintln!(
            "camera: turns {:.1} deg/frame on average, at most {:.1} deg (frames {} -> {}); \
             joins {:.2} and {:.2} deg",
            sum / (frames - 1) as f64,
            worst,
            at,
            at + 1,
            join(FRAMES_IN - 1),
            join(FRAMES_IN + FRAMES_ORBIT - 1)
        );
    }

    for (f, (cam, sph)) in shots.into_iter().enumerate() {
        let start = std::time::Instant::now();
        let img = render_frame(cam, &sph);
        let path = format!("out/frame_{:03}.png", f);
        img.save(&path).unwrap();
        eprintln!(
            "frame {:3}/{}  l = {:+.3}  psi = {:+.1} deg  look-vs-axis = {:.1} deg  \
             sphere at l = {:+.3}  ({:.1} s)",
            f + 1,
            frames,
            cam.l,
            cam.u.y.atan2(cam.u.x).to_degrees(),
            (-cam.fwd.dot(&cam.u)).acos().to_degrees(),
            sph.l,
            start.elapsed().as_secs_f64()
        );

        if f % GIF_EVERY == 0 {
            let rgba: RgbaImage = RgbaImage::from_fn(WIDTH, HEIGHT, |x, y| {
                let p = img.get_pixel(x, y);
                Rgba([p[0], p[1], p[2], 255])
            });
            let frame = Frame::from_parts(rgba, 0, 0, Delay::from_numer_denom_ms(GIF_DELAY_MS, 1));
            gif.encode_frame(frame).unwrap();
        }
    }
    eprintln!("done: out/frame_***.png and out/wormhole.gif");
}
