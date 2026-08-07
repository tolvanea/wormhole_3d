// Puts the traced image on screen.
//
// The compute pass already tone mapped and gamma encoded, so this must not
// encode again: the surface is configured with a non-sRGB format precisely so
// that what the window shows is byte-for-byte what the headless renderer
// writes to PNG.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One oversized triangle covering the viewport, so there is no vertex buffer
// and no seam down the middle of the screen.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: VsOut;
    out.uv = uv;
    out.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Clip space runs +y up, the traced image runs +y down.
    out.pos.y = -out.pos.y;
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src, samp, in.uv);
}
