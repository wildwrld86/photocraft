//! Local inference with ONNX Runtime (`ort`). Sessions are cached per model for the life of the
//! process, so a batch over many files loads each model once.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use crate::registry::ModelSpec;
use crate::{Matte, MattingBackend, MlError, Progress, Result, RgbImage, prep, store};

type Shared = Arc<Mutex<Session>>;

fn cache() -> &'static Mutex<HashMap<&'static str, Shared>> {
    static C: OnceLock<Mutex<HashMap<&'static str, Shared>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

fn rt(e: impl std::fmt::Display) -> MlError {
    MlError::Runtime(e.to_string())
}

fn session(spec: &'static ModelSpec, dir: Option<&Path>, allow_download: bool, progress: &dyn Progress) -> Result<Shared> {
    if let Some(s) = cache().lock().unwrap_or_else(PoisonError::into_inner).get(spec.id) {
        return Ok(s.clone());
    }
    let path = store::ensure(spec, dir, allow_download, progress)?;
    progress.progress(0.5, "Loading AI model");
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let s = Session::builder()
        .map_err(rt)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(rt)?
        .with_intra_threads(threads)
        .map_err(rt)?
        // No CPU arena and no memory-pattern planning: BiRefNet's peak drops from ~8 GB to ~6 GB
        // in the reference benchmark, and memory is returned between images in a batch.
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])
        .map_err(rt)?
        .with_memory_pattern(false)
        .map_err(rt)?
        .commit_from_file(&path)
        .map_err(|e| MlError::Runtime(format!("{}: {e}", path.display())))?;
    let s = Arc::new(Mutex::new(s));
    cache().lock().unwrap_or_else(PoisonError::into_inner).insert(spec.id, s.clone());
    Ok(s)
}

/// A registry model run locally.
pub struct OnnxBackend {
    spec: &'static ModelSpec,
    allow_download: bool,
    dir: Option<PathBuf>,
}

impl OnnxBackend {
    pub fn new(spec: &'static ModelSpec, allow_download: bool, dir: Option<PathBuf>) -> Self {
        Self { spec, allow_download, dir }
    }
}

impl MattingBackend for OnnxBackend {
    fn id(&self) -> String {
        format!("local:{}", self.spec.id)
    }

    fn matte(&self, img: &RgbImage, progress: &dyn Progress) -> Result<Matte> {
        let spec = self.spec;
        let shared = session(spec, self.dir.as_deref(), self.allow_download, progress)?;
        if progress.cancelled() {
            return Err(MlError::Cancelled);
        }
        progress.progress(0.6, "Running AI model");
        let side = spec.input;
        let input = prep::to_tensor(img, side, spec.mean, spec.std)?;
        let tensor = Tensor::from_array(([1usize, 3, side, side], input)).map_err(rt)?;
        let raw: Vec<f32> = {
            let mut s = shared.lock().unwrap_or_else(PoisonError::into_inner);
            let name = s.inputs().first().map(|i| i.name().to_string()).ok_or_else(|| rt("model has no inputs"))?;
            let out = s.run(ort::inputs![name => tensor]).map_err(rt)?;
            let first = out.values().next().ok_or_else(|| rt("model has no outputs"))?;
            let (_, data) = first.try_extract_tensor::<f32>().map_err(rt)?;
            data.to_vec()
        };
        if progress.cancelled() {
            return Err(MlError::Cancelled);
        }
        progress.progress(0.9, "Building mask");
        prep::to_matte(&raw, side, spec.post, img.width, img.height)
    }
}
