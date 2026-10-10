//! Model registry. Every entry is pinned (URL at a fixed revision, byte size, SHA-256) and carries
//! its license, so only commercially usable weights can be listed here. Non-commercial models
//! (BRIA RMBG-1.4 / RMBG-2.0: CC BY-NC 4.0 / bria-rmbg-1.4) are deliberately absent.

use crate::{MlError, Result};

/// How the raw network output maps to a `0..=1` matte.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Post {
    /// Logits: apply a sigmoid.
    Sigmoid,
    /// Arbitrary range: min–max normalise per image (U²-Net / IS-Net convention).
    MinMax,
    /// Already `0..=1`; clamp.
    Clamp,
}

/// A pinned, licensed ONNX matting model.
#[derive(Debug)]
pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    pub license: &'static str,
    pub license_url: &'static str,
    /// Square network input side in pixels.
    pub input: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub post: Post,
    /// Measured peak resident memory of one CPU inference (MB, ONNX Runtime, arena off).
    pub peak_mb: u32,
}

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Default model: BiRefNet-lite (MIT), best quality per second on CPU among the commercially
/// usable candidates benchmarked in the fork's ai-bg report.
pub const DEFAULT_MODEL: &str = "birefnet-lite";

pub const MODELS: &[ModelSpec] = &[
    ModelSpec {
        id: "birefnet-lite",
        label: "BiRefNet lite (Swin-T, general)",
        url: "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/de15b22ba131738a16dff04aab8bdf8dc32e3ac1/onnx/model.onnx",
        sha256: "5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333",
        bytes: 224_005_088,
        license: "MIT",
        license_url: "https://huggingface.co/ZhengPeng7/BiRefNet_lite",
        input: 1024,
        mean: IMAGENET_MEAN,
        std: IMAGENET_STD,
        post: Post::Sigmoid,
        peak_mb: 6000,
    },
    ModelSpec {
        id: "birefnet",
        label: "BiRefNet (Swin-L, general, highest quality)",
        url: "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/534d3c82d3bb8b2f0867db6dfbc3a525b8e42f67/onnx/model.onnx",
        sha256: "58f621f00f5d756097615970a88a791584600dcf7c45b18a0a6267535a1ebd3c",
        bytes: 972_666_916,
        license: "MIT",
        license_url: "https://huggingface.co/ZhengPeng7/BiRefNet",
        input: 1024,
        mean: IMAGENET_MEAN,
        std: IMAGENET_STD,
        post: Post::Sigmoid,
        // Not measured: one CPU run exceeded a 7 GB cap on the 16 GB test box. Conservative guess.
        peak_mb: 12000,
    },
    ModelSpec {
        id: "isnet",
        label: "IS-Net general use (DIS)",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        sha256: "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        bytes: 178_648_008,
        license: "Apache-2.0",
        license_url: "https://github.com/xuebinqin/DIS/blob/main/LICENSE.md",
        input: 1024,
        mean: [0.5, 0.5, 0.5],
        std: [1.0, 1.0, 1.0],
        post: Post::MinMax,
        peak_mb: 900,
    },
    ModelSpec {
        id: "u2netp",
        label: "U²-Net small (fast, low quality)",
        url: "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        sha256: "309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8",
        bytes: 4_574_861,
        license: "Apache-2.0",
        license_url: "https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE",
        input: 320,
        mean: IMAGENET_MEAN,
        std: IMAGENET_STD,
        post: Post::MinMax,
        peak_mb: 400,
    },
    ModelSpec {
        id: "modnet",
        label: "MODNet (portraits only)",
        url: "https://huggingface.co/Xenova/modnet/resolve/fa2fa546052fba4c08921230a26cc69a333fca12/onnx/model.onnx",
        sha256: "07c308cf0fc7e6e8b2065a12ed7fc07e1de8febb7dc7839d7b7f15dd66584df9",
        bytes: 25_888_640,
        license: "Apache-2.0",
        license_url: "https://github.com/ZHKKKe/MODNet/blob/master/LICENSE",
        input: 512,
        mean: [0.5, 0.5, 0.5],
        std: [0.5, 0.5, 0.5],
        post: Post::Clamp,
        peak_mb: 300,
    },
];

/// Fallback when the default model would not fit in memory.
pub const LOW_MEMORY_MODEL: &str = "isnet";

/// Resolves `"auto"` (or no model): the default model, unless the machine reports less free
/// memory than it needs (plus 1 GB headroom), in which case the low-memory model. A machine that
/// cannot report free memory gets the default.
pub fn resolve(id: Option<&str>) -> &str {
    match id {
        Some(m) if m != "auto" => m,
        _ => match (find(DEFAULT_MODEL), available_mb()) {
            (Ok(d), Some(free)) if free < u64::from(d.peak_mb) + 1024 => LOW_MEMORY_MODEL,
            _ => DEFAULT_MODEL,
        },
    }
}

/// Available physical memory in MB (Linux `MemAvailable`; `None` elsewhere).
pub fn available_mb() -> Option<u64> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024)
}

/// Looks a model up by id.
pub fn find(id: &str) -> Result<&'static ModelSpec> {
    MODELS.iter().find(|m| m.id == id).ok_or_else(|| MlError::UnknownModel(id.to_string(), ids().join(", ")))
}

/// All model ids.
pub fn ids() -> Vec<&'static str> {
    MODELS.iter().map(|m| m.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_pinned_and_commercially_licensed() {
        assert!(find(DEFAULT_MODEL).is_ok());
        for m in MODELS {
            assert_eq!(m.sha256.len(), 64, "{}", m.id);
            assert!(m.sha256.chars().all(|c| c.is_ascii_hexdigit()), "{}", m.id);
            assert!(m.url.starts_with("https://"), "{}", m.id);
            assert!(["MIT", "Apache-2.0"].contains(&m.license), "{}: {}", m.id, m.license);
            assert!(m.input >= 32 && m.bytes > 0);
        }
        let mut ids = ids();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), MODELS.len());
    }

    #[test]
    fn auto_resolves_to_a_registered_model() {
        assert!(find(resolve(None)).is_ok());
        assert!(find(resolve(Some("auto"))).is_ok());
        assert_eq!(resolve(Some("u2netp")), "u2netp");
        assert!(find(LOW_MEMORY_MODEL).is_ok());
    }
}
