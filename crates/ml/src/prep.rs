//! Pre- and post-processing shared by every backend: antialiased resampling, normalisation to a
//! `1×3×S×S` NCHW tensor, and mapping raw network output back to a full-resolution matte.

use rayon::prelude::*;

use crate::registry::Post;
use crate::{Matte, MlError, Result, RgbImage};

/// Largest image side the ML layer accepts (guards allocations on hostile input).
pub const MAX_SIDE: usize = 30_000;

/// Precomputed taps for one output coordinate of a separable triangle (bilinear) filter whose
/// support widens when downscaling (antialiasing, like Pillow's BILINEAR).
fn taps(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f32 / dst as f32;
    let support = scale.max(1.0);
    (0..dst)
        .map(|i| {
            let center = (i as f32 + 0.5) * scale;
            let lo = ((center - support).floor().max(0.0)) as usize;
            let hi = ((center + support).ceil() as usize).min(src);
            let mut w: Vec<f32> = (lo..hi).map(|j| (1.0 - ((j as f32 + 0.5 - center) / support).abs()).max(0.0)).collect();
            let sum: f32 = w.iter().sum();
            if sum > 0.0 {
                w.iter_mut().for_each(|v| *v /= sum);
            } else if let Some(first) = w.first_mut() {
                *first = 1.0;
            }
            (lo, w)
        })
        .collect()
}

/// Resamples a `C`-channel interleaved image `(sw, sh)` → `(dw, dh)`.
pub fn resize<const C: usize>(src: &[[f32; C]], sw: usize, sh: usize, dw: usize, dh: usize) -> Result<Vec<[f32; C]>> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 || sw.checked_mul(sh) != Some(src.len()) || dw > MAX_SIDE || dh > MAX_SIDE {
        return Err(MlError::BadInput(format!("resize {sw}x{sh} -> {dw}x{dh} over {} pixels", src.len())));
    }
    if (sw, sh) == (dw, dh) {
        return Ok(src.to_vec());
    }
    let (tx, ty) = (taps(sw, dw), taps(sh, dh));
    // Horizontal pass: sh rows × dw.
    let mut mid = vec![[0.0f32; C]; dw * sh];
    mid.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let s = &src[y * sw..(y + 1) * sw];
        for (o, (lo, w)) in row.iter_mut().zip(&tx) {
            let mut acc = [0.0f32; C];
            for (k, wk) in w.iter().enumerate() {
                if let Some(p) = s.get(lo + k) {
                    for c in 0..C {
                        acc[c] += p[c] * wk;
                    }
                }
            }
            *o = acc;
        }
    });
    // Vertical pass.
    let mut out = vec![[0.0f32; C]; dw * dh];
    out.par_chunks_mut(dw).enumerate().for_each(|(y, row)| {
        let Some((lo, w)) = ty.get(y) else { return };
        for (x, o) in row.iter_mut().enumerate() {
            let mut acc = [0.0f32; C];
            for (k, wk) in w.iter().enumerate() {
                if let Some(p) = mid.get((lo + k) * dw + x) {
                    for c in 0..C {
                        acc[c] += p[c] * wk;
                    }
                }
            }
            *o = acc;
        }
    });
    Ok(out)
}

/// Resizes `img` to `side × side` and normalises it into a planar `1×3×side×side` tensor.
pub fn to_tensor(img: &RgbImage, side: usize, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>> {
    let small = resize(&img.px, img.width, img.height, side, side)?;
    let plane = side * side;
    let mut t = vec![0.0f32; plane * 3];
    for (i, p) in small.iter().enumerate() {
        for c in 0..3 {
            let s = if std[c].abs() < 1e-6 { 1.0 } else { std[c] };
            t[c * plane + i] = (p[c].clamp(0.0, 1.0) - mean[c]) / s;
        }
    }
    Ok(t)
}

/// Maps raw network output (`side × side`) to a `0..=1` matte at the image's size.
pub fn to_matte(raw: &[f32], side: usize, post: Post, width: usize, height: usize) -> Result<Matte> {
    if raw.len() != side * side {
        return Err(MlError::Runtime(format!("model returned {} values, expected {}", raw.len(), side * side)));
    }
    let mut a: Vec<[f32; 1]> = match post {
        Post::Sigmoid => raw.iter().map(|v| [1.0 / (1.0 + (-v).exp())]).collect(),
        Post::Clamp => raw.iter().map(|v| [v.clamp(0.0, 1.0)]).collect(),
        Post::MinMax => {
            let (lo, hi) = raw.iter().filter(|v| v.is_finite()).fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
            let span = (hi - lo).max(1e-6);
            raw.iter().map(|v| [((v - lo) / span).clamp(0.0, 1.0)]).collect()
        }
    };
    a.iter_mut().for_each(|v| {
        if !v[0].is_finite() {
            v[0] = 0.0;
        }
    });
    let full = resize(&a, side, side, width, height)?;
    Ok(Matte { width, height, alpha: full.into_iter().map(|v| v[0].clamp(0.0, 1.0)).collect() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_preserves_constants_and_sizes() {
        let src = vec![[0.25f32, 0.5, 0.75]; 37 * 23];
        for (w, h) in [(10, 10), (100, 61), (37, 23), (1, 1)] {
            let out = resize(&src, 37, 23, w, h).unwrap();
            assert_eq!(out.len(), w * h);
            assert!(out.iter().all(|p| (p[0] - 0.25).abs() < 1e-5 && (p[2] - 0.75).abs() < 1e-5));
        }
        assert!(resize(&src, 36, 23, 5, 5).is_err());
        assert!(resize(&src, 37, 23, 0, 5).is_err());
        assert!(resize(&src, 37, 23, MAX_SIDE + 1, 5).is_err());
    }

    #[test]
    fn tensor_is_planar_and_normalised() {
        let img = RgbImage::new(4, 2, vec![[1.0, 0.5, 0.0]; 8]).unwrap();
        let t = to_tensor(&img, 8, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unwrap();
        assert_eq!(t.len(), 3 * 64);
        assert!((t[0] - 1.0).abs() < 1e-5 && t[64].abs() < 1e-5 && (t[128] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn matte_post_processing() {
        let raw = vec![-20.0, 20.0, -20.0, 20.0];
        let m = to_matte(&raw, 2, Post::Sigmoid, 2, 2).unwrap();
        assert!(m.alpha[0] < 0.01 && m.alpha[1] > 0.99);
        let m = to_matte(&[2.0, 4.0, 2.0, 4.0], 2, Post::MinMax, 4, 4).unwrap();
        assert_eq!(m.alpha.len(), 16);
        assert!(m.alpha.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(to_matte(&[f32::NAN; 4], 2, Post::MinMax, 2, 2).is_ok());
        assert!(to_matte(&[0.0; 3], 2, Post::Clamp, 2, 2).is_err());
    }
}
