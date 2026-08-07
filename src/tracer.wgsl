// Ellis wormhole fly-through, one geodesic per sample.
//
// This is a direct port of the CPU tracer's inner loop. Every ray is a
// geodesic of the spatial metric
//
//     ds^2 = dl^2 + (l^2 + a^2) (dtheta^2 + sin^2 theta dphi^2)
//
// and by spherical symmetry lies in a plane, so the ray reduces to the planar
// system (l, phi, v_l, v_phi) integrated with classic RK4. Rays escaping to
// l -> +inf see universe A, to l -> -inf universe B; rays that never escape
// are winding on the photon ring and come out black.
//
// The one thing that differs from the CPU version is what the ray can hit: a
// glTF triangle mesh instead of an analytic sphere. Each RK4 step gives a
// manifold point (l, u); `offset_from` maps a pair of them into the model's
// own local frame, which is flat to first order because the model is small
// compared with the throat radius, and the resulting straight segment is
// tested against a BVH. No global Cartesian chart is involved, so this stays
// valid right through the throat -- which is the whole point, since the model
// crosses it in shot.

const PI: f32 = 3.14159265358979;
const NO_HIT: u32 = 0xffffffffu;
const FAR: f32 = 1e30;

struct Uniforms {
    cam_u: vec3<f32>,     cam_l: f32,
    cam_fwd: vec3<f32>,   half_fov: f32,
    cam_up: vec3<f32>,    aspect: f32,
    cam_right: vec3<f32>, sky_gain: f32,

    // The model's local orthonormal frame in map axes: outward radial, across,
    // and up. `spin_c`/`spin_s` are the cosine and sine of its rotation about
    // its own z axis, which turns the model but not the light on it.
    body_b1: vec3<f32>,   body_l: f32,
    body_b2: vec3<f32>,   body_r: f32,
    body_b3: vec3<f32>,   spin_c: f32,
    light: vec3<f32>,     spin_s: f32,

    width: u32, height: u32, ssaa: u32, max_steps: u32,
    a: f32, h0: f32, l_escape: f32, pixel_angle: f32,
    tex_size: f32, pad0: f32, pad1: f32, pad2: f32,
}

struct Tri {
    p0: vec3<f32>, mat: u32,
    p1: vec3<f32>, uv_density: f32,
    p2: vec3<f32>, pad2: u32,
    n0: vec3<f32>, pad3: u32,
    n1: vec3<f32>, pad4: u32,
    n2: vec3<f32>, pad5: u32,
    uv0: vec2<f32>, uv1: vec2<f32>,
    uv2: vec2<f32>, pad6: vec2<f32>,
}

// `count > 0` marks a leaf holding `count` triangles from `left_first`;
// otherwise the children are `left_first` and `left_first + 1`.
struct Node {
    bmin: vec3<f32>, left_first: u32,
    bmax: vec3<f32>, count: u32,
}

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> tris: array<Tri>;
@group(0) @binding(3) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var model_tex: texture_2d_array<f32>;
@group(0) @binding(5) var model_samp: sampler;
@group(0) @binding(6) var sky_a: texture_cube<f32>;
@group(0) @binding(7) var sky_b: texture_cube<f32>;
@group(0) @binding(8) var sky_samp: sampler;

// --------------------------- geodesic integrator ---------------------------

struct GS { l: f32, phi: f32, vl: f32, vphi: f32 }

fn deriv(s: GS) -> GS {
    let r2 = s.l * s.l + U.a * U.a;
    return GS(s.vl, s.vphi, s.l * s.vphi * s.vphi, -2.0 * s.l / r2 * s.vl * s.vphi);
}

fn axpy(s: GS, h: f32, d: GS) -> GS {
    return GS(s.l + h * d.l, s.phi + h * d.phi, s.vl + h * d.vl, s.vphi + h * d.vphi);
}

fn rk4_step(s: GS, h: f32) -> GS {
    let k1 = deriv(s);
    let k2 = deriv(axpy(s, 0.5 * h, k1));
    let k3 = deriv(axpy(s, 0.5 * h, k2));
    let k4 = deriv(axpy(s, h, k3));
    let c = h / 6.0;
    return GS(
        s.l    + c * (k1.l    + 2.0 * k2.l    + 2.0 * k3.l    + k4.l),
        s.phi  + c * (k1.phi  + 2.0 * k2.phi  + 2.0 * k3.phi  + k4.phi),
        s.vl   + c * (k1.vl   + 2.0 * k2.vl   + 2.0 * k3.vl   + k4.vl),
        s.vphi + c * (k1.vphi + 2.0 * k2.vphi + 2.0 * k3.vphi + k4.vphi),
    );
}

/// Angular position on the sphere after sweeping `phi` in the geodesic plane.
fn u_at(p: vec3<f32>, t: vec3<f32>, phi: f32) -> vec3<f32> {
    return p * cos(phi) + t * sin(phi);
}

/// The plane's tangential basis vector, rotated along with it.
fn t_at(p: vec3<f32>, t: vec3<f32>, phi: f32) -> vec3<f32> {
    return -p * sin(phi) + t * cos(phi);
}

fn any_perp(u: vec3<f32>) -> vec3<f32> {
    var a = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(u.z) >= 0.9) { a = vec3<f32>(1.0, 0.0, 0.0); }
    return normalize(a - u * dot(a, u));
}

/// First-order proper offset of the manifold point (l, u) from (l0, u0), as a
/// vector in map axes: the radial part along u0 plus the tangential part along
/// the great circle towards u, scaled to a proper length by the radius midway
/// between them. The two are orthogonal, so its length is a proper distance.
///
/// The tangential part uses the arc, not the chord: the chord vanishes at the
/// antipode of u0 as well as at u0 itself, which would read as a second,
/// spurious copy of the model for rays that have wound half way round.
fn offset_from(l0: f32, u0: vec3<f32>, l: f32, u: vec3<f32>) -> vec3<f32> {
    let r_mid = sqrt(0.25 * (l + l0) * (l + l0) + U.a * U.a);
    let c = dot(u, u0);
    let across = u - u0 * c;
    let s = length(across);
    var tangential = vec3<f32>(0.0);
    if (s > 1e-9) {
        tangential = across * (r_mid * atan2(s, c) / s);
    } else if (c <= 0.0) {
        tangential = any_perp(u0) * (r_mid * PI);
    }
    return u0 * (l - l0) + tangential;
}

/// Manifold point -> the model's own coordinates: into the body frame, then
/// wound back by the spin so the mesh itself never has to move.
fn to_model(l: f32, u: vec3<f32>) -> vec3<f32> {
    let o = offset_from(U.body_l, U.body_b1, l, u);
    let b = vec3<f32>(dot(o, U.body_b1), dot(o, U.body_b2), dot(o, U.body_b3));
    return vec3<f32>(b.x * U.spin_c + b.y * U.spin_s, -b.x * U.spin_s + b.y * U.spin_c, b.z);
}

// ------------------------------ mesh traversal ------------------------------

struct Hit { t: f32, tri: u32, bu: f32, bv: f32 }

/// Moller-Trumbore, double sided: the helmet has open edges and shells, so
/// rejecting back faces would punch holes in it.
fn tri_hit(idx: u32, o: vec3<f32>, d: vec3<f32>, tmax: f32) -> Hit {
    var h: Hit;
    h.tri = NO_HIT;
    h.t = tmax;
    let tr = tris[idx];
    let e1 = tr.p1 - tr.p0;
    let e2 = tr.p2 - tr.p0;
    let pv = cross(d, e2);
    let det = dot(e1, pv);
    if (abs(det) < 1e-16) { return h; }
    let inv_det = 1.0 / det;
    let tv = o - tr.p0;
    let bu = dot(tv, pv) * inv_det;
    if (bu < 0.0 || bu > 1.0) { return h; }
    let qv = cross(tv, e1);
    let bv = dot(d, qv) * inv_det;
    if (bv < 0.0 || bu + bv > 1.0) { return h; }
    let t = dot(e2, qv) * inv_det;
    if (t <= 1e-7 || t >= tmax) { return h; }
    h.t = t;
    h.tri = idx;
    h.bu = bu;
    h.bv = bv;
    return h;
}

/// Entry distance of the slab test, or FAR if the segment misses the box.
fn slab(bmin: vec3<f32>, bmax: vec3<f32>, o: vec3<f32>, inv: vec3<f32>, tmax: f32) -> f32 {
    let t0 = (bmin - o) * inv;
    let t1 = (bmax - o) * inv;
    let tn = min(t0, t1);
    let tf = max(t0, t1);
    let tenter = max(max(tn.x, tn.y), max(tn.z, 0.0));
    let texit = min(min(tf.x, tf.y), min(tf.z, tmax));
    if (tenter <= texit) { return tenter; }
    return FAR;
}

/// Nearest triangle crossed by the segment o -> o + d, as a parameter in
/// [0, 1]. Children are visited near-first so the far one is usually culled by
/// the hit found in the near one.
fn trace_segment(o: vec3<f32>, d: vec3<f32>) -> Hit {
    var best: Hit;
    best.t = 1.0; // nothing past the end of this step counts
    best.tri = NO_HIT;

    // A zero component would make (bmin - o) * inv indeterminate; nudging it
    // just makes that axis' slab infinitely wide, which is the right answer.
    let sd = select(d, vec3<f32>(1e-20), abs(d) < vec3<f32>(1e-20));
    let inv = vec3<f32>(1.0) / sd;

    var stack: array<u32, 32>;
    var sp = 0u;
    var node = 0u;
    loop {
        let n = nodes[node];
        if (n.count > 0u) {
            for (var i = 0u; i < n.count; i = i + 1u) {
                let h = tri_hit(n.left_first + i, o, d, best.t);
                if (h.tri != NO_HIT) { best = h; }
            }
            if (sp == 0u) { break; }
            sp = sp - 1u;
            node = stack[sp];
            continue;
        }
        var c1 = n.left_first;
        var c2 = n.left_first + 1u;
        var d1 = slab(nodes[c1].bmin, nodes[c1].bmax, o, inv, best.t);
        var d2 = slab(nodes[c2].bmin, nodes[c2].bmax, o, inv, best.t);
        if (d1 > d2) {
            let td = d1; d1 = d2; d2 = td;
            let tc = c1; c1 = c2; c2 = tc;
        }
        if (d1 >= FAR) {
            if (sp == 0u) { break; }
            sp = sp - 1u;
            node = stack[sp];
        } else {
            node = c1;
            if (d2 < FAR && sp < 31u) {
                stack[sp] = c2;
                sp = sp + 1u;
            }
        }
    }
    return best;
}

// -------------------------------- shading ----------------------------------

/// Sky of the universe the ray escaped into. Map axes put "up" on +z and the
/// cube maps are authored with +y up; the cyclic swap is the rotation that
/// lines them up. The textures are sRGB, so the fetch returns linear light.
fn sky(n: vec3<f32>, side: f32) -> vec3<f32> {
    let dir = vec3<f32>(n.y, n.z, n.x);
    var c: vec3<f32>;
    if (side > 0.0) {
        c = textureSampleLevel(sky_a, sky_samp, dir, 0.0).rgb;
    } else {
        c = textureSampleLevel(sky_b, sky_samp, dir, 0.0).rgb;
    }
    return c * U.sky_gain;
}

/// Base colour times a Lambert key light and a flat ambient term.
///
/// The light is fixed in the model's own frame, so it stays lit the same way
/// all the way round -- a distant light source would be a fiction here, with
/// two universes and no time coordinate to propagate it along.
fn shade(h: Hit, d: vec3<f32>, dist: f32) -> vec3<f32> {
    let tr = tris[h.tri];
    let w = 1.0 - h.bu - h.bv;
    var n = normalize(tr.n0 * w + tr.n1 * h.bu + tr.n2 * h.bv);
    // Double-sided geometry: face the normal back towards the incoming ray.
    if (dot(n, d) > 0.0) { n = -n; }
    let uv = tr.uv0 * w + tr.uv1 * h.bu + tr.uv2 * h.bv;

    // No screen-space derivatives in a ray tracer, so the mip level comes from
    // how wide the ray's cone has spread by the time it arrives: one sample's
    // angular size times the proper distance travelled, converted to texels.
    var lod = 0.0;
    if (tr.uv_density > 0.0) {
        lod = log2(max(dist * U.pixel_angle * tr.uv_density * U.tex_size, 1.0));
    }
    let albedo = textureSampleLevel(model_tex, model_samp, uv, tr.mat, lod).rgb;

    // The model turns about its own z, so its normals sit in a frame rotated
    // by -spin from the body frame the light is defined in; rotate back.
    let nb = vec3<f32>(n.x * U.spin_c - n.y * U.spin_s, n.x * U.spin_s + n.y * U.spin_c, n.z);
    let k = 0.25 + 1.15 * max(dot(nb, U.light), 0.0);
    return albedo * k;
}

// ------------------------------- ray casting -------------------------------

fn ray_color(d: vec3<f32>) -> vec3<f32> {
    let p_hat = U.cam_u; // radial unit vector at the camera
    let vl0 = dot(d, U.cam_u); // radial velocity component
    let dt = d - U.cam_u * vl0; // tangential part
    let tlen = length(dt);
    var t_hat: vec3<f32>;
    var vph0: f32;
    if (tlen < 1e-9) {
        // Purely radial ray; the geodesic plane is arbitrary, take any normal.
        t_hat = any_perp(U.cam_u);
        vph0 = 0.0;
    } else {
        t_hat = dt / tlen;
        vph0 = tlen;
    }

    var s = GS(U.cam_l, 0.0, vl0, vph0 / sqrt(U.cam_l * U.cam_l + U.a * U.a));
    var arc = 0.0;

    for (var step = 0u; step < U.max_steps; step = step + 1u) {
        let r = sqrt(s.l * s.l + U.a * U.a);
        if (abs(s.l) > U.l_escape && s.l * s.vl > 0.0) {
            let side = select(-1.0, 1.0, s.l >= 0.0);
            // Rotate the plane basis by the total swept angle, then take the
            // asymptotic straight-line direction in the escape universe
            // (radius grows like |l|, hence the sign on the radial part).
            let p_f = u_at(p_hat, t_hat, s.phi);
            let t_f = t_at(p_hat, t_hat, s.phi);
            let n = normalize(p_f * (s.vl * side) + t_f * (r * s.vphi));
            return sky(n, side);
        }

        // Step size grows with distance from the throat: the geometry (and the
        // bending) only has structure on scales ~ sqrt(l^2 + a^2).
        var h = U.h0 * r;

        // Is the model close enough to be worth a full segment test? The
        // radial reject is a single comparison and throws out almost every
        // ray; including h in the band means no step can jump clean across it
        // (the ray has unit speed, so one step advances arc length h).
        let u_now = u_at(p_hat, t_hat, s.phi);
        let band = 6.0 * U.body_r + h;
        var near = abs(s.l - U.body_l) < band;
        if (near) {
            near = length(offset_from(U.body_l, U.body_b1, s.l, u_now)) < band;
        }
        if (near) {
            // Keep the chord short against the model's features, so that
            // replacing the curved step by a straight segment stays accurate.
            h = min(h, 0.25 * U.body_r);
        }

        let next = rk4_step(s, h);
        if (near) {
            // The segment test needs no bisection refinement: unlike the
            // sphere's inside/outside test, it lands exactly on the triangle.
            let a0 = to_model(s.l, u_now);
            let a1 = to_model(next.l, u_at(p_hat, t_hat, next.phi));
            let seg = a1 - a0;
            let hit = trace_segment(a0, seg);
            if (hit.tri != NO_HIT) {
                return shade(hit, normalize(seg), arc + hit.t * h);
            }
        }
        s = next;
        arc = arc + h;
    }
    return vec3<f32>(0.0); // photon ring: never escapes
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let t = vec3<f32>(1.0) - exp(-max(c, vec3<f32>(0.0))); // soft exposure roll-off
    return clamp(pow(t, vec3<f32>(1.0 / 2.2)), vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= U.width || gid.y >= U.height) { return; }
    let ns = max(U.ssaa, 1u);
    var acc = vec3<f32>(0.0);
    for (var sj = 0u; sj < ns; sj = sj + 1u) {
        for (var si = 0u; si < ns; si = si + 1u) {
            let uu = (f32(gid.x) + (f32(si) + 0.5) / f32(ns)) / f32(U.width);
            let vv = (f32(gid.y) + (f32(sj) + 0.5) / f32(ns)) / f32(U.height);
            let px = (2.0 * uu - 1.0) * U.half_fov;
            let py = (1.0 - 2.0 * vv) * U.half_fov * U.aspect;
            let d = normalize(U.cam_fwd + U.cam_right * px + U.cam_up * py);
            acc = acc + ray_color(d);
        }
    }
    // Tone mapped and gamma encoded here, exactly as the CPU tracer did, so
    // the storage texture already holds display-ready bytes: the headless path
    // writes them straight to a PNG and the window blits them untouched.
    let c = tonemap(acc / f32(ns * ns));
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(c, 1.0));
}
