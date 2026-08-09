//! Loading the chase model from glTF, and building the BVH the shader walks.
//!
//! The model replaces the checkered sphere the CPU tracer used. Everything
//! about the wormhole stays the same; only the intersection test changes, from
//! "is this point within R of the centre" to "does this segment cross a
//! triangle". The README's sketch of that swap is what this implements.
//!
//! Coordinates. The model lives in its own little flat Euclidean box. The
//! shader brings each ray segment into that box with `offset_from`, which is
//! valid because the model is small compared with the throat radius, so its
//! neighbourhood is flat to first order. glTF is Y-up and the sample models
//! face +Z; the model's local frame here is (outward radial, across, up), so
//! the axes are remapped
//!
//!     local x = glTF z   (front, i.e. back towards the chasing camera)
//!     local y = glTF x
//!     local z = glTF y   (up, which is the map's +z on the flight plane)
//!
//! a cyclic permutation, hence orientation-preserving -- the model is not
//! mirrored.

use crate::assets;
use crate::scene::MODEL_R;
use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};

/// One triangle, ready for the GPU.
///
/// Laid out so that WGSL's `vec3<f32>` alignment (16-byte aligned, 12 bytes
/// wide) leaves exactly a `u32` slot behind each position and normal; the
/// first of those carries the material id and the rest are padding. 128 bytes
/// per triangle, so the whole helmet is about 12 MB.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuTri {
    pub p0: [f32; 3],
    pub mat: u32,
    pub p1: [f32; 3],
    /// Texels per unit area, as sqrt(uv area / world area). A ray tracer has
    /// no screen-space derivatives to pick a mip level from, so the shader
    /// reconstructs one from this and the width of the ray's cone at the hit.
    pub uv_density: f32,
    pub p2: [f32; 3],
    pub _pad2: u32,
    pub n0: [f32; 3],
    pub _pad3: u32,
    pub n1: [f32; 3],
    pub _pad4: u32,
    pub n2: [f32; 3],
    pub _pad5: u32,
    pub uv0: [f32; 2],
    pub uv1: [f32; 2],
    pub uv2: [f32; 2],
    pub _pad6: [f32; 2],
}

/// A flattened BVH node. `count > 0` marks a leaf, whose triangles are the
/// `count` entries starting at `left_first`; otherwise `left_first` is the
/// index of the left child and the right child is the one after it, since the
/// builder allocates the pair together.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuNode {
    pub bmin: [f32; 3],
    pub left_first: u32,
    pub bmax: [f32; 3],
    pub count: u32,
}

/// glTF's three ways of treating the alpha channel.
pub const ALPHA_OPAQUE: u32 = 0;
pub const ALPHA_MASK: u32 = 1;
pub const ALPHA_BLEND: u32 = 2;

/// Everything the shader needs to know about one material.
///
/// Triangles carry a material index rather than a texture layer, which is what
/// lets a material have no texture at all: the pony's teeth, tongue, eyelashes
/// and glowing pip are pure `baseColorFactor`, and before this they were being
/// silently textured with whatever happened to be in layer 0.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GpuMaterial {
    /// baseColorFactor, RGBA. Multiplies the texture where there is one.
    pub base_color: [f32; 4],
    /// emissiveFactor already multiplied by KHR_materials_emissive_strength.
    pub emissive: [f32; 3],
    pub alpha_cutoff: f32,
    /// Layer in the texture array, or -1 for "no texture, use the factor".
    pub base_tex: i32,
    pub emissive_tex: i32,
    pub alpha_mode: u32,
    pub _pad: u32,
}

/// One image with its mip chain, mip 0 first.
pub struct Tex {
    pub mips: Vec<Vec<u8>>,
}

/// The loaded model: GPU-ready triangles, BVH, materials and textures.
pub struct Model {
    pub tris: Vec<GpuTri>,
    pub nodes: Vec<GpuNode>,
    pub materials: Vec<GpuMaterial>,
    /// One layer per unique image, all at `tex_size` with the same number of
    /// mip levels. Base colour and emissive share the array.
    pub textures: Vec<Tex>,
    pub tex_size: u32,
    pub mip_levels: u32,
}

// ------------------------------- glTF loading -------------------------------

impl Model {
    /// Load the first scene of a `.gltf` file, scaled and re-axised so that it
    /// sits centred in the model's local frame with bounding radius `MODEL_R`.
    pub fn load(path: &Path) -> Result<Model> {
        // Check the path ourselves first: the gltf crate reports a missing
        // file as a bare io error with no mention of which file.
        let json = assets::read(path, "the glTF model")?;
        let doc = gltf::Gltf::from_slice(&json)
            .with_context(|| format!("cannot parse the glTF model: {}", path.display()))?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));

        // Buffers. The sample model keeps its data in an external .bin, so
        // resolve URIs relative to the .gltf; embedded base64 is handled too.
        let mut buffers: Vec<Vec<u8>> = Vec::new();
        for buf in doc.buffers() {
            let data = match buf.source() {
                gltf::buffer::Source::Bin => doc.blob.clone().ok_or_else(|| {
                    anyhow!(
                        "the glTF file declares a binary chunk but does not contain one: {}",
                        path.display()
                    )
                })?,
                // Named by the glTF, so the interesting thing to report is
                // which file asked for it -- the user did not choose this path.
                gltf::buffer::Source::Uri(uri) => read_uri(dir, uri).with_context(|| {
                    format!(
                        "{} references a buffer that could not be read",
                        path.display()
                    )
                })?,
            };
            buffers.push(data);
        }

        // Materials, and the unique images they point at. Base colour and
        // emissive images share one array, since several materials reuse the
        // same file and the shader indexes both the same way.
        let mut image_paths: Vec<PathBuf> = Vec::new();
        let mut intern = |src: Option<gltf::texture::Info>| -> i32 {
            let Some(info) = src else { return -1 };
            match info.texture().source().source() {
                gltf::image::Source::Uri { uri, .. } => {
                    let p = dir.join(percent_decode(uri));
                    match image_paths.iter().position(|q| *q == p) {
                        Some(i) => i as i32,
                        None => {
                            image_paths.push(p);
                            image_paths.len() as i32 - 1
                        }
                    }
                }
                // An image packed into the .bin would need decoding out of the
                // buffer view; no model here uses it, so treat it as untextured
                // rather than pretending some other layer belongs to it.
                gltf::image::Source::View { .. } => -1,
            }
        };

        let mut materials: Vec<GpuMaterial> = Vec::new();
        for mat in doc.materials() {
            let pbr = mat.pbr_metallic_roughness();
            let alpha_mode = match mat.alpha_mode() {
                gltf::material::AlphaMode::Opaque => ALPHA_OPAQUE,
                gltf::material::AlphaMode::Mask => ALPHA_MASK,
                gltf::material::AlphaMode::Blend => ALPHA_BLEND,
            };
            // KHR_materials_emissive_strength scales the factor; folding it in
            // here means the shader never has to know the extension exists.
            let strength = mat.emissive_strength().unwrap_or(1.0);
            let e = mat.emissive_factor();
            materials.push(GpuMaterial {
                base_color: pbr.base_color_factor(),
                emissive: [e[0] * strength, e[1] * strength, e[2] * strength],
                alpha_cutoff: mat.alpha_cutoff().unwrap_or(0.5),
                base_tex: intern(pbr.base_color_texture()),
                emissive_tex: intern(mat.emissive_texture()),
                alpha_mode,
                _pad: 0,
            });
        }
        // A primitive may have no material at all, in which case glTF says to
        // use the default: white, opaque, untextured. Kept last so existing
        // indices are untouched.
        let default_material = materials.len() as u32;
        materials.push(GpuMaterial {
            base_color: [1.0, 1.0, 1.0, 1.0],
            emissive: [0.0; 3],
            alpha_cutoff: 0.5,
            base_tex: -1,
            emissive_tex: -1,
            alpha_mode: ALPHA_OPAQUE,
            _pad: 0,
        });

        // Geometry. Node transforms are walked so that a model with a rigged
        // hierarchy still lands in the right place; FlightHelmet's are all
        // identity, but nothing here depends on that.
        let mut tris: Vec<GpuTri> = Vec::new();
        let scene = doc
            .default_scene()
            .or_else(|| doc.scenes().next())
            .ok_or_else(|| anyhow!("the glTF file contains no scenes: {}", path.display()))?;
        for node in scene.nodes() {
            visit_node(
                &node,
                nalgebra::Matrix4::identity(),
                &buffers,
                default_material,
                &mut tris,
            );
        }
        if tris.is_empty() {
            bail!(
                "the glTF model contains no triangles: {}\n\n  \
                 its meshes may use point or line primitives, which this \
                 renderer does not draw",
                path.display()
            );
        }

        // Centre on the bounding box and scale the bounding sphere to MODEL_R,
        // so the model occupies the same slot in frame the old sphere did and
        // the flight path needs no retuning.
        let (lo, hi) = bounds_of(&tris);
        let centre = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        let mut far: f32 = 0.0;
        for t in &tris {
            for p in [t.p0, t.p1, t.p2] {
                let d = (0..3).map(|k| (p[k] - centre[k]).powi(2)).sum::<f32>();
                far = far.max(d);
            }
        }
        let scale = MODEL_R as f32 / far.sqrt().max(1e-9);
        for t in &mut tris {
            for p in [&mut t.p0, &mut t.p1, &mut t.p2] {
                for k in 0..3 {
                    p[k] = (p[k] - centre[k]) * scale;
                }
            }
        }

        // Axis remap, applied after scaling so the two stay independent.
        for t in &mut tris {
            for v in [
                &mut t.p0, &mut t.p1, &mut t.p2, &mut t.n0, &mut t.n1, &mut t.n2,
            ] {
                *v = [v[2], v[0], v[1]];
            }
        }

        // Texel density is measured after scaling, since it relates UV area to
        // world area and the scale changes the latter.
        for t in &mut tris {
            t.uv_density = texel_density(t);
        }

        let (textures, tex_size) = load_textures(&image_paths)?;
        let mip_levels = textures[0].mips.len() as u32;
        let nodes = build_bvh(&mut tris);

        eprintln!(
            "model: {} triangles, {} BVH nodes, {} material(s), {} texture layer(s) \
             at {}x{} ({} mips)",
            tris.len(),
            nodes.len(),
            materials.len(),
            textures.len(),
            tex_size,
            tex_size,
            mip_levels
        );
        report_materials(&doc, &materials, &textures);

        Ok(Model {
            tris,
            nodes,
            materials,
            textures,
            tex_size,
            mip_levels,
        })
    }
}

/// Say which materials are see-through or glowing, and -- for the blended ones
/// -- how much alpha their texture actually carries.
///
/// A material marked BLEND whose texture is solid alpha renders exactly like an
/// opaque one, which looks like the transparency support is broken when it is
/// really the asset saying "fully opaque". Worth one line at startup to tell
/// those two cases apart.
fn report_materials(doc: &gltf::Gltf, materials: &[GpuMaterial], textures: &[Tex]) {
    let names: Vec<String> = doc
        .materials()
        .map(|m| m.name().unwrap_or("<unnamed>").to_string())
        .collect();
    for (i, m) in materials.iter().enumerate() {
        let name = names.get(i).map(String::as_str).unwrap_or("<default>");
        if m.alpha_mode != ALPHA_OPAQUE {
            let kind = if m.alpha_mode == ALPHA_MASK {
                format!("masked at {:.2}", m.alpha_cutoff)
            } else {
                "blended".to_string()
            };
            // Mean alpha over the smallest mip that still has some area, which
            // is a cheap stand-in for "how transparent is this on average".
            let coverage = if m.base_tex >= 0 {
                textures
                    .get(m.base_tex as usize)
                    .and_then(|t| t.mips.iter().rev().nth(4))
                    .map(|mip| {
                        let a: u32 = mip.iter().skip(3).step_by(4).map(|v| *v as u32).sum();
                        a as f32 / (255.0 * (mip.len() / 4) as f32)
                    })
                    .unwrap_or(1.0)
                    * m.base_color[3]
            } else {
                m.base_color[3]
            };
            eprintln!(
                "  material {i} \"{name}\": {kind}, mean alpha {coverage:.2}{}",
                if coverage > 0.99 {
                    "  (already opaque -- nothing to see through)"
                } else {
                    ""
                }
            );
        }
        if m.emissive.iter().any(|&v| v > 0.0) {
            eprintln!(
                "  material {i} \"{name}\": emissive {:.2},{:.2},{:.2}",
                m.emissive[0], m.emissive[1], m.emissive[2]
            );
        }
    }
}

fn read_uri(dir: &Path, uri: &str) -> Result<Vec<u8>> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let b64 = rest
            .split(";base64,")
            .nth(1)
            .ok_or_else(|| anyhow!("unsupported data URI in the glTF (not base64)"))?;
        return base64_decode(b64);
    }
    // Percent-decoding matters: glTF URIs escape spaces in file names.
    let path = dir.join(percent_decode(uri));
    assets::read(&path, &format!("the glTF buffer \"{uri}\""))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%'
            && i + 2 < b.len()
            && let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn base64_decode(s: &str) -> Result<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lut = [255u8; 256];
    for (i, c) in TABLE.iter().enumerate() {
        lut[*c as usize] = i as u8;
    }
    let (mut acc, mut bits, mut out) = (0u32, 0u32, Vec::new());
    for c in s.bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = lut[c as usize];
        if v == 255 {
            bail!("bad base64 in the glTF data URI (unexpected character {:?})", c as char);
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

/// Walk a node and its children, accumulating world-space triangles.
fn visit_node(
    node: &gltf::Node,
    parent: nalgebra::Matrix4<f32>,
    buffers: &[Vec<u8>],
    default_material: u32,
    out: &mut Vec<GpuTri>,
) {
    let local = nalgebra::Matrix4::from_column_slice(
        &node
            .transform()
            .matrix()
            .iter()
            .flat_map(|c| c.iter().copied())
            .collect::<Vec<f32>>(),
    );
    let xform = parent * local;
    // Normals transform by the inverse transpose; for the rigid, uniformly
    // scaled transforms glTF scenes normally carry this is the same rotation,
    // but a non-uniform scale would shear them otherwise.
    let normal_xform = xform
        .fixed_view::<3, 3>(0, 0)
        .try_inverse()
        .map(|m| m.transpose())
        .unwrap_or_else(nalgebra::Matrix3::identity);

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue; // strips/fans are not used by the sample models
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|v| v.as_slice()));
            let Some(positions) = reader.read_positions() else {
                continue;
            };
            let positions: Vec<[f32; 3]> = positions.collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> =
                reader.read_tex_coords(0).map(|t| t.into_f32().collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            // The index into our material table; primitives with no material
            // of their own get glTF's default.
            let mat = prim
                .material()
                .index()
                .map(|i| i as u32)
                .unwrap_or(default_material);

            for tri in indices.chunks_exact(3) {
                let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
                let p = |i: usize| {
                    let v = positions[i];
                    let w = xform * nalgebra::Vector4::new(v[0], v[1], v[2], 1.0);
                    [w.x, w.y, w.z]
                };
                // Without shading normals, fall back to the face normal, which
                // renders faceted but correct.
                let face = face_normal(p(a), p(b), p(c));
                let n = |i: usize| match &normals {
                    Some(ns) => {
                        let v = ns[i];
                        let w = normal_xform * nalgebra::Vector3::new(v[0], v[1], v[2]);
                        let len = w.norm();
                        if len > 1e-12 {
                            [w.x / len, w.y / len, w.z / len]
                        } else {
                            face
                        }
                    }
                    None => face,
                };
                let uv = |i: usize| uvs.as_ref().map(|t| t[i]).unwrap_or([0.0, 0.0]);
                out.push(GpuTri {
                    p0: p(a),
                    mat,
                    p1: p(b),
                    p2: p(c),
                    n0: n(a),
                    n1: n(b),
                    n2: n(c),
                    uv0: uv(a),
                    uv1: uv(b),
                    uv2: uv(c),
                    ..Default::default()
                });
            }
        }
    }
    for child in node.children() {
        visit_node(&child, xform, buffers, default_material, out);
    }
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 1e-20 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn bounds_of(tris: &[GpuTri]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for t in tris {
        for p in [t.p0, t.p1, t.p2] {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    (lo, hi)
}

/// sqrt(UV area / world area) of a triangle: how many texture widths the
/// surface covers per unit of length, which is what turns a world-space cone
/// width at the hit into a mip level.
fn texel_density(t: &GpuTri) -> f32 {
    let e1 = [
        t.p1[0] - t.p0[0],
        t.p1[1] - t.p0[1],
        t.p1[2] - t.p0[2],
    ];
    let e2 = [
        t.p2[0] - t.p0[0],
        t.p2[1] - t.p0[1],
        t.p2[2] - t.p0[2],
    ];
    let cross = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];
    let world = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let duv1 = [t.uv1[0] - t.uv0[0], t.uv1[1] - t.uv0[1]];
    let duv2 = [t.uv2[0] - t.uv0[0], t.uv2[1] - t.uv0[1]];
    let uv = (duv1[0] * duv2[1] - duv1[1] * duv2[0]).abs();
    // Degenerate triangles (zero area either way) get density 0, which the
    // shader reads as "use the coarsest sensible level" rather than NaN.
    if world > 1e-20 && uv > 0.0 {
        (uv / world).sqrt()
    } else {
        0.0
    }
}

/// Decode the images, bring them all to a common size (a texture array needs
/// uniform dimensions, and these rarely agree), and build box-filtered mip
/// chains.
///
/// Mips are not optional here. A ray tracer samples one point per ray, so a
/// 2048-square texture on a model a few hundred pixels across would alias into
/// a shimmering mess -- especially under lensing, which stretches the model
/// unevenly across the frame.
///
/// The common size is the next power of two at or above the largest source,
/// capped, rather than a fixed 2048: the pony's images are mostly 1024 with one
/// 2000, and forcing every layer up to 2048 would triple the texture memory to
/// gain nothing.
fn load_textures(paths: &[PathBuf]) -> Result<(Vec<Tex>, u32)> {
    const MAX_SIZE: u32 = 2048;
    if paths.is_empty() {
        // A model with no textures still needs one layer to bind: plain white,
        // so the base colour factor shows through unmodified.
        const SIZE: u32 = 4;
        let white = vec![255u8; (SIZE * SIZE * 4) as usize];
        return Ok((
            vec![Tex {
                mips: mip_chain(white, SIZE),
            }],
            SIZE,
        ));
    }

    let mut images = Vec::with_capacity(paths.len());
    let mut widest = 1;
    for p in paths {
        let what = format!(
            "the texture \"{}\"",
            p.file_name().unwrap_or(p.as_os_str()).to_string_lossy()
        );
        let img = assets::open_image(p, &what)?.to_rgba8();
        widest = widest.max(img.width()).max(img.height());
        images.push(img);
    }
    let size = widest.next_power_of_two().min(MAX_SIZE);

    let mut out = Vec::with_capacity(images.len());
    for img in images {
        let img = if img.width() != size || img.height() != size {
            image::imageops::resize(&img, size, size, image::imageops::FilterType::CatmullRom)
        } else {
            img
        };
        out.push(Tex {
            mips: mip_chain(img.into_raw(), size),
        });
    }
    Ok((out, size))
}

/// Successive 2x2 box reductions down to 1x1.
///
/// The averaging is done on the raw sRGB bytes rather than in linear light.
/// That is the same (slightly dark) convention the GPU's own mip generation
/// uses, and matching it matters more here than being right in the abstract:
/// the alternative would make the mips disagree with the hardware filtering
/// that blends between them.
fn mip_chain(mip0: Vec<u8>, size: u32) -> Vec<Vec<u8>> {
    let mut mips = vec![mip0];
    let mut w = size;
    while w > 1 {
        let src = mips.last().unwrap();
        let hw = w / 2;
        let mut dst = vec![0u8; (hw * hw * 4) as usize];
        for y in 0..hw {
            for x in 0..hw {
                for c in 0..4 {
                    let at = |sx: u32, sy: u32| src[(((sy * w) + sx) * 4 + c) as usize] as u32;
                    let sum = at(2 * x, 2 * y)
                        + at(2 * x + 1, 2 * y)
                        + at(2 * x, 2 * y + 1)
                        + at(2 * x + 1, 2 * y + 1);
                    dst[(((y * hw) + x) * 4 + c) as usize] = ((sum + 2) / 4) as u8;
                }
            }
        }
        mips.push(dst);
        w = hw;
    }
    mips
}

// ---------------------------------- BVH ------------------------------------

const BINS: usize = 12;
const MAX_LEAF: usize = 4;

/// Binned-SAH BVH over the triangles, reordering `tris` so each leaf's
/// primitives are contiguous. Children are allocated in pairs, so one index
/// per interior node locates both.
fn build_bvh(tris: &mut Vec<GpuTri>) -> Vec<GpuNode> {
    let n = tris.len();
    let centroids: Vec<[f32; 3]> = tris
        .iter()
        .map(|t| {
            [
                (t.p0[0] + t.p1[0] + t.p2[0]) / 3.0,
                (t.p0[1] + t.p1[1] + t.p2[1]) / 3.0,
                (t.p0[2] + t.p1[2] + t.p2[2]) / 3.0,
            ]
        })
        .collect();
    let bounds: Vec<([f32; 3], [f32; 3])> = tris
        .iter()
        .map(|t| {
            let mut lo = t.p0;
            let mut hi = t.p0;
            for p in [t.p1, t.p2] {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            (lo, hi)
        })
        .collect();

    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut nodes: Vec<GpuNode> = Vec::with_capacity(2 * n / MAX_LEAF + 1);
    nodes.push(GpuNode::default());
    subdivide(&mut nodes, 0, &mut order, 0, n, &centroids, &bounds);

    // Apply the permutation the build settled on.
    let reordered: Vec<GpuTri> = order.iter().map(|&i| tris[i as usize]).collect();
    *tris = reordered;
    nodes
}

fn aabb_of(idx: &[u32], bounds: &[([f32; 3], [f32; 3])]) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for &i in idx {
        let (l, h) = bounds[i as usize];
        for k in 0..3 {
            lo[k] = lo[k].min(l[k]);
            hi[k] = hi[k].max(h[k]);
        }
    }
    (lo, hi)
}

fn half_area(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let d = [
        (hi[0] - lo[0]).max(0.0),
        (hi[1] - lo[1]).max(0.0),
        (hi[2] - lo[2]).max(0.0),
    ];
    d[0] * d[1] + d[1] * d[2] + d[2] * d[0]
}

fn subdivide(
    nodes: &mut Vec<GpuNode>,
    node_idx: usize,
    order: &mut [u32],
    start: usize,
    count: usize,
    centroids: &[[f32; 3]],
    bounds: &[([f32; 3], [f32; 3])],
) {
    let (lo, hi) = aabb_of(&order[start..start + count], bounds);
    nodes[node_idx].bmin = lo;
    nodes[node_idx].bmax = hi;

    // Written whenever the node is not worth splitting, by any of the several
    // ways that can turn out.
    macro_rules! make_leaf {
        () => {{
            nodes[node_idx].left_first = start as u32;
            nodes[node_idx].count = count as u32;
            return;
        }};
    }
    if count <= MAX_LEAF {
        make_leaf!();
    }

    // Split along the widest extent of the centroids, binned by SAH.
    let (mut clo, mut chi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for &i in order[start..start + count].iter() {
        for k in 0..3 {
            clo[k] = clo[k].min(centroids[i as usize][k]);
            chi[k] = chi[k].max(centroids[i as usize][k]);
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| (chi[a] - clo[a]).total_cmp(&(chi[b] - clo[b])))
        .unwrap();
    let extent = chi[axis] - clo[axis];
    if extent < 1e-12 {
        make_leaf!(); // degenerate: all centroids coincide
    }

    let scale = BINS as f32 / extent;
    let mut bin_count = [0usize; BINS];
    let mut bin_lo = [[f32::INFINITY; 3]; BINS];
    let mut bin_hi = [[f32::NEG_INFINITY; 3]; BINS];
    let bin_of = |c: f32| (((c - clo[axis]) * scale) as usize).min(BINS - 1);
    for &i in order[start..start + count].iter() {
        let b = bin_of(centroids[i as usize][axis]);
        bin_count[b] += 1;
        let (l, h) = bounds[i as usize];
        for k in 0..3 {
            bin_lo[b][k] = bin_lo[b][k].min(l[k]);
            bin_hi[b][k] = bin_hi[b][k].max(h[k]);
        }
    }

    // Sweep once from each side to get the cost of every split plane.
    let mut left_area = [0.0f32; BINS - 1];
    let mut right_area = [0.0f32; BINS - 1];
    let mut left_n = [0usize; BINS - 1];
    let mut right_n = [0usize; BINS - 1];
    {
        let (mut l, mut h, mut c) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3], 0usize);
        for b in 0..BINS - 1 {
            c += bin_count[b];
            for k in 0..3 {
                l[k] = l[k].min(bin_lo[b][k]);
                h[k] = h[k].max(bin_hi[b][k]);
            }
            left_n[b] = c;
            left_area[b] = if c > 0 { half_area(l, h) } else { 0.0 };
        }
        let (mut l, mut h, mut c) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3], 0usize);
        for b in (1..BINS).rev() {
            c += bin_count[b];
            for k in 0..3 {
                l[k] = l[k].min(bin_lo[b][k]);
                h[k] = h[k].max(bin_hi[b][k]);
            }
            right_n[b - 1] = c;
            right_area[b - 1] = if c > 0 { half_area(l, h) } else { 0.0 };
        }
    }
    let mut best = (f32::INFINITY, 0usize);
    for b in 0..BINS - 1 {
        let cost = left_area[b] * left_n[b] as f32 + right_area[b] * right_n[b] as f32;
        if left_n[b] > 0 && right_n[b] > 0 && cost < best.0 {
            best = (cost, b);
        }
    }
    // Splitting has to beat leaving the node whole, or the traversal pays for
    // a node that buys it nothing.
    let leaf_cost = half_area(lo, hi) * count as f32;
    if best.0 >= leaf_cost {
        make_leaf!();
    }

    let split = best.1;
    let mid = partition(&mut order[start..start + count], |i| {
        bin_of(centroids[i as usize][axis]) <= split
    });
    if mid == 0 || mid == count {
        make_leaf!();
    }

    // The two children are allocated side by side, so the parent only has to
    // remember where the pair starts.
    let left_idx = nodes.len();
    nodes.push(GpuNode::default());
    nodes.push(GpuNode::default());
    nodes[node_idx].left_first = left_idx as u32;
    nodes[node_idx].count = 0;
    subdivide(nodes, left_idx, order, start, mid, centroids, bounds);
    subdivide(
        nodes,
        left_idx + 1,
        order,
        start + mid,
        count - mid,
        centroids,
        bounds,
    );
}

/// In-place stable-enough partition; returns the number of elements kept left.
fn partition(slice: &mut [u32], pred: impl Fn(u32) -> bool) -> usize {
    let mut i = 0;
    let mut j = slice.len();
    while i < j {
        if pred(slice[i]) {
            i += 1;
        } else {
            j -= 1;
            slice.swap(i, j);
        }
    }
    i
}
