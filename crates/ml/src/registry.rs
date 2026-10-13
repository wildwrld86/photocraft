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
        // ~6 GB peak with the arena off (round-1 benchmark). In 0.1.1 it passed 5.6 GB with every
        // setting tried (2 or 8 threads, prepacking off, arena on), for a 480×640 or 1920×1527
        // image alike, so it can't run on an 8 GB machine.
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
        // 0.1.1 settings (arena and memory pattern off, ≤8 threads): 766 MB peak RSS for a
        // 480×640 image on the 16 GB test box; 904 MB with the arena and memory pattern on.
        peak_mb: 800,
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

/// Machines with less physical memory than this (MB) get the low-memory model from Auto, whatever
/// is free at the moment: an "8 GB" laptop reports 7–7.9 GB (less with memory reserved for an
/// integrated GPU), and BiRefNet's peak is several GB at any image size, because every image is
/// resized to the model's fixed 1024×1024 input first.
pub const LOW_TOTAL_MB: u64 = 10 * 1024;

/// Free memory to keep on top of a model's measured peak before running it.
pub const HEADROOM_MB: u64 = 1024;

/// Resolves `"auto"` (or no model): the default model, unless the machine is small or short on
/// free memory (see [`auto_choice`]) or an earlier run crashed ([`crate::guard`]).
pub fn resolve(id: Option<&str>) -> &str {
    match id {
        Some(m) if m != "auto" => m,
        _ => auto_choice(total_mb(), available_mb(), crashed_before()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn crashed_before() -> bool {
    crate::guard::low_memory_flag()
}
#[cfg(target_arch = "wasm32")]
fn crashed_before() -> bool {
    false
}

/// Auto's choice from the machine's total and available memory (MB, `None` = unknown) and
/// whether a run already crashed here.
pub fn auto_choice(total_mb: Option<u64>, available_mb: Option<u64>, crashed_before: bool) -> &'static str {
    let Ok(d) = find(DEFAULT_MODEL) else { return LOW_MEMORY_MODEL };
    let small = total_mb.is_some_and(|t| t < LOW_TOTAL_MB);
    let short = available_mb.is_some_and(|a| !fits(d, a));
    if small || short || crashed_before { LOW_MEMORY_MODEL } else { DEFAULT_MODEL }
}

/// Whether `spec` should run with `available_mb` free (its measured peak plus headroom).
pub fn fits(spec: &ModelSpec, available_mb: u64) -> bool {
    available_mb >= u64::from(spec.peak_mb) + HEADROOM_MB
}

/// The model to retry with when `id` runs out of memory: the low-memory model for the BiRefNet
/// family, nothing for models that are already small.
pub fn fallback_for(id: &str) -> Option<&'static ModelSpec> {
    let spec = find(id).ok()?;
    let low = find(LOW_MEMORY_MODEL).ok()?;
    (spec.peak_mb > low.peak_mb * 2).then_some(low)
}

/// Available physical memory in MB: `MemAvailable` on Linux, `GlobalMemoryStatusEx` on Windows,
/// free + inactive + purgeable pages on macOS (both through `sysinfo`). `None` elsewhere.
pub fn available_mb() -> Option<u64> {
    memory().map(|m| m.1)
}

/// Total physical memory in MB (`None` where unknown).
pub fn total_mb() -> Option<u64> {
    memory().map(|m| m.0)
}

/// `(total, available)` in MB.
#[cfg(target_os = "linux")]
fn memory() -> Option<(u64, u64)> {
    parse_meminfo(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

#[cfg(any(target_os = "windows", target_os = "macos"))]
fn memory() -> Option<(u64, u64)> {
    use sysinfo::{MemoryRefreshKind, RefreshKind, System};
    let s = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()));
    let mb = |b: u64| b / (1024 * 1024);
    (s.total_memory() > 0).then(|| (mb(s.total_memory()), mb(s.available_memory())))
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn memory() -> Option<(u64, u64)> {
    None
}

/// `(MemTotal, MemAvailable)` in MB from `/proc/meminfo` text.
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let kb = |key: &str| -> Option<u64> { text.lines().find(|l| l.starts_with(key))?.split_whitespace().nth(1)?.parse().ok() };
    Some((kb("MemTotal:")? / 1024, kb("MemAvailable:")? / 1024))
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
    fn auto_picks_the_low_memory_model_on_small_or_busy_machines() {
        // Daniel's 8 GB Vivobook: about 7.3 GB total, a few GB free.
        assert_eq!(auto_choice(Some(7_400), Some(3_000), false), LOW_MEMORY_MODEL);
        assert_eq!(auto_choice(Some(7_900), Some(7_500), false), LOW_MEMORY_MODEL);
        // 16 GB with plenty free: the default; with too little free: low memory.
        assert_eq!(auto_choice(Some(16_000), Some(12_000), false), DEFAULT_MODEL);
        let need = u64::from(find(DEFAULT_MODEL).unwrap().peak_mb) + HEADROOM_MB;
        assert_eq!(auto_choice(Some(16_000), Some(need - 1), false), LOW_MEMORY_MODEL);
        assert_eq!(auto_choice(Some(16_000), Some(need), false), DEFAULT_MODEL);
        // Unknown memory: the default, unless a run already crashed here.
        assert_eq!(auto_choice(None, None, false), DEFAULT_MODEL);
        assert_eq!(auto_choice(None, None, true), LOW_MEMORY_MODEL);
        assert_eq!(auto_choice(Some(64_000), Some(60_000), true), LOW_MEMORY_MODEL);
    }

    #[test]
    fn birefnet_falls_back_to_isnet_and_small_models_do_not() {
        assert_eq!(fallback_for("birefnet-lite").map(|m| m.id), Some(LOW_MEMORY_MODEL));
        assert_eq!(fallback_for("birefnet").map(|m| m.id), Some(LOW_MEMORY_MODEL));
        assert!(fallback_for("isnet").is_none() && fallback_for("u2netp").is_none() && fallback_for("nope").is_none());
    }

    #[test]
    fn meminfo_parses_and_this_machine_reports_memory() {
        let t = "MemTotal:       16397536 kB\nMemFree:  100 kB\nMemAvailable:    6667264 kB\n";
        assert_eq!(parse_meminfo(t), Some((16013, 6511)));
        assert_eq!(parse_meminfo("MemTotal: 1 kB\n"), None);
        if cfg!(any(target_os = "linux", target_os = "windows", target_os = "macos")) {
            let (t, a) = (total_mb().unwrap(), available_mb().unwrap());
            assert!(t > 0 && a <= t, "{t} {a}");
        }
    }

    #[test]
    fn auto_resolves_to_a_registered_model() {
        assert!(find(resolve(None)).is_ok());
        assert!(find(resolve(Some("auto"))).is_ok());
        assert_eq!(resolve(Some("u2netp")), "u2netp");
        assert!(find(LOW_MEMORY_MODEL).is_ok());
    }
}
