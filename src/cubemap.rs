//! Cutting the cross-form sky images into six cube faces for the GPU.
//!
//! The CPU tracer did its own face selection and bilinear fetch. On the GPU a
//! real cube texture does both in hardware, and the face order and projection
//! WebGPU uses is exactly the convention the old `CubeMap::sample` implemented
//! by hand -- so the faces go up in the standard order and the shader just
//! asks for a direction.

use crate::assets;
use anyhow::{Result, bail};
use std::path::Path;

/// Cell `(column, row)` of each face inside the 4x3 cross, in the WebGPU cube
/// layer order +X, -X, +Y, -Y, +Z, -Z. The cross sits on the left of the
/// image: its vertical arm is column 1, and the -Z face trails off to the right.
///
///     .   +Y  .   .
///     -X  +Z  +X  -Z
///     .   -Y  .   .
const FACE_CELLS: [(u32, u32); 6] = [(2, 1), (0, 1), (1, 0), (1, 2), (1, 1), (3, 1)];

pub struct Sky {
    pub size: u32,
    /// Six RGBA8 faces in cube layer order, ready to upload.
    pub faces: Vec<Vec<u8>>,
}

impl Sky {
    pub fn load(path: &Path, what: &str) -> Result<Sky> {
        let img = assets::open_image(path, what)?.to_rgba8();
        let (w, h) = (img.width(), img.height());
        if w % 4 != 0 || h % 3 != 0 || w / 4 != h / 3 {
            bail!(
                "{what} is not a cube map cross\n\n  {}\n\n  \
                 expected a 4x3 grid of square faces, but {w}x{h} is not one \
                 ({w}/4 = {:.1}, {h}/3 = {:.1})",
                path.display(),
                w as f64 / 4.0,
                h as f64 / 3.0,
            );
        }
        let size = w / 4;
        let faces = FACE_CELLS
            .iter()
            .map(|&(col, row)| {
                let (ox, oy) = (col * size, row * size);
                let mut px = Vec::with_capacity((size * size * 4) as usize);
                for y in 0..size {
                    for x in 0..size {
                        px.extend_from_slice(&img.get_pixel(ox + x, oy + y).0);
                    }
                }
                px
            })
            .collect();
        Ok(Sky { size, faces })
    }
}
