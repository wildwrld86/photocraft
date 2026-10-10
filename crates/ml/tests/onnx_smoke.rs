//! End-to-end check of the local ONNX path with the smallest registry model (u2netp, 4.6 MB,
//! Apache-2.0), downloaded on first use into `PHOTOCRAFT_MODELS_DIR` / the user cache.
//! Ignored by default (network); run with
//! `cargo test -p photocraft-ml --features onnx --test onnx_smoke -- --ignored`.
#![cfg(all(feature = "onnx", not(target_arch = "wasm32")))]

use photocraft_ml::{NoProgress, Provider, RgbImage, backend};

#[test]
#[ignore = "downloads a 4.6 MB model"]
fn u2netp_separates_a_disc_from_a_flat_background() {
    let (w, h) = (240usize, 180usize);
    let px = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 - 120.0, (i / w) as f32 - 90.0);
            if x * x + y * y < 55.0 * 55.0 { [0.9, 0.25, 0.1] } else { [0.93, 0.93, 0.95] }
        })
        .collect();
    let img = RgbImage::new(w, h, px).unwrap();
    let b = backend(&Provider::Local { model: "u2netp".into(), allow_download: true }).unwrap();
    let m = b.matte(&img, &NoProgress).unwrap();
    assert_eq!((m.width, m.height, m.alpha.len()), (w, h, w * h));
    let center = m.alpha[90 * w + 120];
    let corner = m.alpha[5 * w + 5];
    assert!(center > 0.8 && corner < 0.2, "center {center} corner {corner}");
}
