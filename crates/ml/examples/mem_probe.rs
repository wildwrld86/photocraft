//! Peak-memory probe for one local inference: `cargo run --release -p photocraft-ml --features onnx
//! --example mem_probe -- <model> [width height]` with `PHOTOCRAFT_MODELS_DIR` set to a cache
//! holding the model. Measure the process peak RSS from outside (e.g. `/usr/bin/time -v`); the
//! `PHOTOCRAFT_ML_*` tuning variables in `onnx.rs` select the session settings.
use photocraft_ml::{NoProgress, Provider, RgbImage, backend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let model = args.first().cloned().unwrap_or_else(|| "birefnet-lite".into());
    let w: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(480);
    let h: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(640);
    let px = (0..w * h).map(|i| [(i % w) as f32 / w as f32, (i / w) as f32 / h as f32, 0.5]).collect();
    let img = RgbImage::new(w, h, px)?;
    let b = backend(&Provider::Local { model, allow_download: false, models_dir: None })?;
    let t = std::time::Instant::now();
    let m = b.matte(&img, &NoProgress)?;
    let mean = m.alpha.iter().sum::<f32>() / m.alpha.len().max(1) as f32;
    println!("backend={} {}x{} {:.2}s mean_alpha={mean:.3} note={:?}", b.id(), m.width, m.height, t.elapsed().as_secs_f64(), b.note());
    Ok(())
}
