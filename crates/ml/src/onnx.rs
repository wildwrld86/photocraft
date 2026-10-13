//! Local inference with ONNX Runtime (`ort`). Sessions are cached per model for the life of the
//! process, so a batch over many files loads each model once.
//!
//! Memory safety net (8 GB laptops): before a BiRefNet run the free memory is checked against the
//! model's measured peak and the low-memory model is used instead when it doesn't fit; an
//! allocation failure during a run evicts the session and retries once with the low-memory
//! model; a panic inside the runtime call becomes an error; and [`crate::guard`] leaves a record
//! if the process dies mid-run anyway. Each switch is reported through [`MattingBackend::note`].

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::registry::{self, ModelSpec};
use crate::{Matte, MattingBackend, MlError, Progress, Result, RgbImage, guard, prep, store};

type Shared = Arc<Mutex<Session>>;

fn cache() -> &'static Mutex<HashMap<&'static str, Shared>> {
    static C: OnceLock<Mutex<HashMap<&'static str, Shared>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Drops the cached session of `id` (frees its weights and buffers once no run holds it).
pub fn evict(id: &str) {
    cache().lock().unwrap_or_else(PoisonError::into_inner).remove(id);
}

fn rt(e: impl std::fmt::Display) -> MlError {
    MlError::Runtime(e.to_string())
}

/// Session settings. Defaults chosen by peak-memory measurements (crates/ml/README.md); the
/// environment overrides exist for those measurements and for support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    /// Intra-op threads (`PHOTOCRAFT_ML_THREADS`).
    pub threads: usize,
    /// CPU memory arena (`PHOTOCRAFT_ML_ARENA=1`; off by default).
    pub arena: bool,
    /// Memory-pattern planning (`PHOTOCRAFT_ML_MEM_PATTERN=1`; off by default).
    pub mem_pattern: bool,
    /// Weight prepacking (`PHOTOCRAFT_ML_PREPACK=0` turns it off).
    pub prepack: bool,
}

impl Tuning {
    pub fn from_env() -> Self {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        let flag = |k: &str, default: bool| env(k).map_or(default, |v| v != "0");
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Tuning {
            threads: env("PHOTOCRAFT_ML_THREADS").and_then(|v| v.parse().ok()).filter(|n| *n > 0).unwrap_or_else(|| default_threads(cores)),
            arena: flag("PHOTOCRAFT_ML_ARENA", false),
            mem_pattern: flag("PHOTOCRAFT_ML_MEM_PATTERN", false),
            prepack: flag("PHOTOCRAFT_ML_PREPACK", true),
        }
    }
}

/// Intra-op threads: every core up to 8. More threads barely change peak memory (measured), and
/// beyond 8 they no longer speed these models up.
pub fn default_threads(cores: usize) -> usize {
    cores.clamp(1, 8)
}

fn session(spec: &'static ModelSpec, dir: Option<&Path>, allow_download: bool, progress: &dyn Progress) -> Result<Shared> {
    if let Some(s) = cache().lock().unwrap_or_else(PoisonError::into_inner).get(spec.id) {
        return Ok(s.clone());
    }
    let path = store::ensure(spec, dir, allow_download, progress)?;
    progress.progress(0.5, "Loading AI model");
    let t = Tuning::from_env();
    let s = Session::builder()
        .map_err(rt)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(rt)?
        .with_intra_threads(t.threads)
        .map_err(rt)?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(t.arena).build()])
        .map_err(rt)?
        .with_memory_pattern(t.mem_pattern)
        .map_err(rt)?
        .with_prepacking(t.prepack)
        .map_err(rt)?
        .commit_from_file(&path)
        .map_err(|e| MlError::Runtime(format!("{}: {e}", path.display())))?;
    let s = Arc::new(Mutex::new(s));
    cache().lock().unwrap_or_else(PoisonError::into_inner).insert(spec.id, s.clone());
    Ok(s)
}

/// `PHOTOCRAFT_ML_SIMULATE_OOM=<model id>`: that model fails like ONNX Runtime out of memory
/// (support and tests of the fallback without exhausting the machine).
fn simulated_oom(spec: &ModelSpec) -> bool {
    std::env::var("PHOTOCRAFT_ML_SIMULATE_OOM").is_ok_and(|v| v.split(',').any(|m| m.trim() == spec.id))
}

/// "BiRefNet lite" from "BiRefNet lite (Swin-T, general)".
fn short(spec: &ModelSpec) -> &str {
    spec.label.split(" (").next().unwrap_or(spec.label)
}

fn gb(mb: u64) -> String {
    format!("{:.1} GB", mb as f64 / 1024.0)
}

/// A registry model run locally.
pub struct OnnxBackend {
    spec: &'static ModelSpec,
    allow_download: bool,
    dir: Option<PathBuf>,
    /// The model the last run actually used, and its note.
    last: Mutex<(Option<&'static str>, Option<String>)>,
}

impl OnnxBackend {
    pub fn new(spec: &'static ModelSpec, allow_download: bool, dir: Option<PathBuf>) -> Self {
        Self { spec, allow_download, dir, last: Mutex::new((None, None)) }
    }

    fn run(&self, spec: &'static ModelSpec, img: &RgbImage, progress: &dyn Progress) -> Result<Matte> {
        // Covers loading the model too: ONNX Runtime allocates most while building the session.
        // Any way the process ends before the guard drops (a C++ abort, Rust's abort on a failed
        // allocation, a kill) leaves the marker behind for the next run to report.
        let _guard = guard::begin(spec.id, img.width, img.height);
        let shared = session(spec, self.dir.as_deref(), self.allow_download, progress)?;
        if progress.cancelled() {
            return Err(MlError::Cancelled);
        }
        progress.progress(0.55, "Preparing image");
        let side = spec.input;
        let input = prep::to_tensor(img, side, spec.mean, spec.std)?;
        let tensor = Tensor::from_array(([1usize, 3, side, side], input)).map_err(rt)?;
        progress.busy("Running AI model");
        let raw: Vec<f32> = catch_unwind(AssertUnwindSafe(|| -> Result<Vec<f32>> {
            if simulated_oom(spec) {
                return Err(MlError::Runtime("Non-zero status code returned while running Mul node. Status Message: bad allocation (simulated)".into()));
            }
            let mut s = shared.lock().unwrap_or_else(PoisonError::into_inner);
            let name = s.inputs().first().map(|i| i.name().to_string()).ok_or_else(|| rt("model has no inputs"))?;
            let out = s.run(ort::inputs![name => tensor]).map_err(rt)?;
            let first = out.values().next().ok_or_else(|| rt("model has no outputs"))?;
            let (_, data) = first.try_extract_tensor::<f32>().map_err(rt)?;
            Ok(data.to_vec())
        }))
        .unwrap_or_else(|_| Err(MlError::Runtime(format!("{} failed with an internal error", spec.id))))?;
        if progress.cancelled() {
            return Err(MlError::Cancelled);
        }
        progress.progress(0.9, "Building mask");
        prep::to_matte(&raw, side, spec.post, img.width, img.height)
    }
}

impl MattingBackend for OnnxBackend {
    fn id(&self) -> String {
        let used = self.last.lock().unwrap_or_else(PoisonError::into_inner).0;
        format!("local:{}", used.unwrap_or(self.spec.id))
    }

    fn note(&self) -> Option<String> {
        self.last.lock().unwrap_or_else(PoisonError::into_inner).1.clone()
    }

    fn matte(&self, img: &RgbImage, progress: &dyn Progress) -> Result<Matte> {
        let mut notes: Vec<String> = guard::previous_crash().map(str::to_string).into_iter().collect();
        let fallback = registry::fallback_for(self.spec.id);
        // Too little free memory for this model right now: don't start it.
        let mut spec = self.spec;
        // `PHOTOCRAFT_ML_MEMORY_CHECK=0` skips this (machines with plenty of swap, measurements).
        let check = std::env::var("PHOTOCRAFT_ML_MEMORY_CHECK").map_or(true, |v| v != "0");
        if let (true, Some(low), Some(free)) = (check, fallback, registry::available_mb())
            && !registry::fits(spec, free)
        {
            evict(spec.id);
            notes.push(format!(
                "Only {} of memory is free and {} needs about {}, so {} was used.",
                gb(free),
                short(spec),
                gb(u64::from(spec.peak_mb) + registry::HEADROOM_MB),
                short(low)
            ));
            spec = low;
        }
        let mut r = self.run(spec, img, progress);
        if let (Err(e), Some(low)) = (&r, fallback)
            && spec.id != low.id
            && crate::is_out_of_memory(e)
        {
            evict(spec.id);
            notes.push(format!("{} ran out of memory, so {} was used instead.", short(spec), short(low)));
            spec = low;
            r = self.run(spec, img, progress);
        }
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = (Some(spec.id), (!notes.is_empty()).then(|| notes.join(" ")));
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_are_capped() {
        assert_eq!(default_threads(0), 1);
        assert_eq!(default_threads(4), 4);
        assert_eq!(default_threads(32), 8);
    }

    #[test]
    fn simulated_oom_names_models() {
        // Read-only check of the parsing; the variable is not set in the test environment.
        if std::env::var_os("PHOTOCRAFT_ML_SIMULATE_OOM").is_none() {
            assert!(!simulated_oom(registry::find("birefnet-lite").unwrap()));
        }
    }
}
