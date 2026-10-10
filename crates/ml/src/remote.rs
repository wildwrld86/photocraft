//! Remote matting over HTTP, behind the same [`MattingBackend`] trait as local models.
//!
//! Wire contract (deliberately minimal so a self-hosted server, a serverless function or a thin
//! adapter for a commercial API can implement it): `POST <url>` with an `image/png` body (the RGB
//! image) and `Authorization: Bearer <key>` when a key is set. The response is a PNG of the same
//! size: either an RGBA cutout (its alpha is the matte) or a grayscale / RGB mask (luma).

use std::io::{Cursor, Read};

use image::{DynamicImage, ImageFormat, RgbImage as ImgRgb};

use crate::{Matte, MattingBackend, MlError, Progress, Result, RgbImage};

/// Largest response accepted (bytes).
const MAX_RESPONSE: u64 = 256 << 20;

pub struct RemoteBackend {
    url: String,
    key: Option<String>,
}

impl RemoteBackend {
    pub fn new(url: &str, key: Option<String>) -> Self {
        Self { url: url.to_string(), key }
    }
}

fn err(e: impl std::fmt::Display) -> MlError {
    MlError::Remote(e.to_string())
}

/// Encodes `img` as an 8-bit PNG.
pub fn encode_png(img: &RgbImage) -> Result<Vec<u8>> {
    let w = u32::try_from(img.width).map_err(err)?;
    let h = u32::try_from(img.height).map_err(err)?;
    let bytes: Vec<u8> = img.px.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
    let buf = ImgRgb::from_raw(w, h, bytes).ok_or_else(|| err("bad image buffer"))?;
    let mut out = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(buf).write_to(&mut out, ImageFormat::Png).map_err(err)?;
    Ok(out.into_inner())
}

/// Decodes a response PNG into a matte, resampling it if the server returned another size.
pub fn decode_matte(bytes: &[u8], width: usize, height: usize) -> Result<Matte> {
    let img = image::load_from_memory_with_format(bytes, ImageFormat::Png).map_err(err)?;
    let (w, h) = (img.width() as usize, img.height() as usize);
    let alpha: Vec<[f32; 1]> = if img.color().has_alpha() {
        img.to_rgba8().pixels().map(|p| [p.0[3] as f32 / 255.0]).collect()
    } else {
        img.to_luma8().pixels().map(|p| [p.0[0] as f32 / 255.0]).collect()
    };
    let full = crate::prep::resize(&alpha, w, h, width, height)?;
    Ok(Matte { width, height, alpha: full.into_iter().map(|v| v[0].clamp(0.0, 1.0)).collect() })
}

impl MattingBackend for RemoteBackend {
    fn id(&self) -> String {
        "remote".into()
    }

    fn matte(&self, img: &RgbImage, progress: &dyn Progress) -> Result<Matte> {
        progress.progress(0.2, "Uploading image");
        let body = encode_png(img)?;
        let agent: ureq::Agent = ureq::Agent::config_builder().user_agent(crate::USER_AGENT).build().into();
        let mut req = agent.post(&self.url).header("Content-Type", "image/png").header("Accept", "image/png");
        if let Some(k) = &self.key {
            req = req.header("Authorization", &format!("Bearer {k}"));
        }
        let resp = req.send(&body[..]).map_err(err)?;
        if progress.cancelled() {
            return Err(MlError::Cancelled);
        }
        progress.progress(0.8, "Reading mask");
        let mut bytes = Vec::new();
        resp.into_body().into_reader().take(MAX_RESPONSE).read_to_end(&mut bytes).map_err(err)?;
        decode_matte(&bytes, img.width, img.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_round_trip_reads_alpha_or_luma() {
        let img = RgbImage::new(3, 2, vec![[1.0, 1.0, 1.0], [0.0; 3], [1.0; 3], [0.0; 3], [1.0; 3], [0.0; 3]]).unwrap();
        let png = encode_png(&img).unwrap();
        let m = decode_matte(&png, 3, 2).unwrap();
        assert!(m.alpha[0] > 0.99 && m.alpha[1] < 0.01);
        assert!(decode_matte(b"not a png", 3, 2).is_err());
    }
}
