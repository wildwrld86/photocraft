//! PhotoCraft ML layer (`photocraft-ml`, L4).
//!
//! Background removal / subject matting behind one trait, [`MattingBackend`], with two
//! implementations:
//!
//! * **Local** ([`onnx`], feature `onnx`): ONNX Runtime through the `ort` crate. Models are not
//!   vendored: [`registry::MODELS`] pins each one to a URL, byte size and SHA-256, and
//!   [`store`] downloads it into the user's cache directory on first use and verifies it.
//! * **Remote** ([`remote`], feature `remote`): POST the image to an HTTP endpoint (self-hosted or
//!   a paid API) and read a matte back, for the web build or machines without the model.
//!
//! The crate has no workspace dependencies: it takes plain interleaved RGB (`0..=1`) and returns a
//! matte of the same size, so the engine owns all document, colour and history concerns. Without
//! features (and always on wasm32) [`backend`] returns [`MlError::Unsupported`], never panics.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(all(feature = "onnx", not(target_arch = "wasm32"), not(all(target_os = "macos", target_arch = "x86_64"))))]
pub mod onnx;
#[cfg(not(target_arch = "wasm32"))]
pub mod paths;
pub mod prep;
pub mod registry;
#[cfg(all(feature = "remote", not(target_arch = "wasm32")))]
pub mod remote;
#[cfg(not(target_arch = "wasm32"))]
pub mod secrets;
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
pub mod store;

pub use registry::{DEFAULT_MODEL, MODELS, ModelSpec};

/// Errors from the ML layer. All are recoverable; none is a panic.
#[derive(Debug, thiserror::Error)]
pub enum MlError {
    #[error("{0}")]
    Unsupported(String),
    #[error("unknown model \"{0}\" (available: {1})")]
    UnknownModel(String, String),
    #[error("model \"{0}\" is not downloaded and downloads are disabled ({1})")]
    NotDownloaded(String, String),
    #[error("model download failed: {0}")]
    Download(String),
    #[error("inference failed: {0}")]
    Runtime(String),
    #[error("remote matting failed: {0}")]
    Remote(String),
    #[error("bad input: {0}")]
    BadInput(String),
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, MlError>;

/// Interleaved RGB image, values `0..=1`, row-major.
#[derive(Clone, Debug, Default)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub px: Vec<[f32; 3]>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, px: Vec<[f32; 3]>) -> Result<Self> {
        if width == 0 || height == 0 || width.checked_mul(height) != Some(px.len()) {
            return Err(MlError::BadInput(format!("{width}x{height} image with {} pixels", px.len())));
        }
        Ok(Self { width, height, px })
    }
}

/// A single-channel alpha matte (`0..=1`, row-major), the same size as the input image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Matte {
    pub width: usize,
    pub height: usize,
    pub alpha: Vec<f32>,
}

/// Progress / cancellation hooks passed down from the engine's job context.
pub trait Progress: Sync {
    fn progress(&self, _fraction: f32, _label: &str) {}
    fn cancelled(&self) -> bool {
        false
    }
}

/// No-op progress.
pub struct NoProgress;
impl Progress for NoProgress {}

/// Something that turns an RGB image into a foreground alpha matte.
pub trait MattingBackend: Send + Sync {
    /// Short id for logs and command results (e.g. `local:birefnet-lite`).
    fn id(&self) -> String;
    /// Computes the matte at the input's resolution.
    fn matte(&self, img: &RgbImage, progress: &dyn Progress) -> Result<Matte>;
}

/// Which backend to use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Provider {
    /// Local ONNX Runtime with a registry model id, cached in `models_dir` (default cache when
    /// `None`).
    Local { model: String, allow_download: bool, models_dir: Option<std::path::PathBuf> },
    /// Remote HTTP endpoint.
    Remote { url: String, api_key: Option<String> },
}

/// Options a caller (command params, preferences) can set; `None` means "the default".
#[derive(Clone, Debug, Default)]
pub struct Options<'a> {
    /// `"local"` (default) or `"remote"`.
    pub provider: Option<&'a str>,
    /// Registry id or `"auto"`.
    pub model: Option<&'a str>,
    pub allow_download: bool,
    pub url: Option<&'a str>,
    pub models_dir: Option<&'a str>,
    /// Send the stored cloud key (only to an endpoint the user configured).
    pub send_key: bool,
}

impl Provider {
    /// Builds a provider. The remote URL falls back to `PHOTOCRAFT_ML_REMOTE_URL`; the key comes
    /// from [`secrets::cloud_key`] (`PHOTOCRAFT_ML_REMOTE_KEY` or the owner-only key file).
    /// Downloads can be disabled globally with `PHOTOCRAFT_ML_OFFLINE=1`.
    pub fn from(o: &Options<'_>) -> Result<Self> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        match o.provider.unwrap_or("local") {
            "local" => {
                let offline = env("PHOTOCRAFT_ML_OFFLINE").is_some_and(|v| v != "0");
                Ok(Provider::Local {
                    model: registry::resolve(o.model).to_string(),
                    allow_download: o.allow_download && !offline,
                    models_dir: o.models_dir.filter(|d| !d.trim().is_empty()).map(std::path::PathBuf::from),
                })
            }
            "remote" => {
                let url = o.url.filter(|u| !u.trim().is_empty()).map(str::to_string).or_else(|| env("PHOTOCRAFT_ML_REMOTE_URL")).ok_or_else(|| {
                    MlError::Unsupported(
                        "the cloud provider needs a URL (Preferences › AI Models › Cloud URL, the \"url\" param or PHOTOCRAFT_ML_REMOTE_URL)".into(),
                    )
                })?;
                Ok(Provider::Remote { url, api_key: if o.send_key { cloud_key() } else { None } })
            }
            other => Err(MlError::Unsupported(format!("unknown provider \"{other}\" (local|remote)"))),
        }
    }

    /// Shorthand for [`Provider::from`] with the common options.
    pub fn from_options(provider: Option<&str>, model: Option<&str>, allow_download: bool, url: Option<&str>) -> Result<Self> {
        Self::from(&Options { provider, model, allow_download, url, models_dir: None, send_key: true })
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cloud_key() -> Option<String> {
    secrets::cloud_key()
}
#[cfg(target_arch = "wasm32")]
fn cloud_key() -> Option<String> {
    None
}

/// What the UI needs before running a local model: is it downloaded, how big, which license.
#[derive(Clone, Debug)]
pub struct ModelStatus {
    pub spec: &'static ModelSpec,
    pub downloaded: bool,
    pub path: Option<std::path::PathBuf>,
}

/// Status of a local model in `models_dir` (default cache when `None`).
pub fn model_status(model: Option<&str>, models_dir: Option<&str>) -> Result<ModelStatus> {
    let spec = registry::find(registry::resolve(model))?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let dir = paths::models_dir(models_dir.filter(|d| !d.trim().is_empty()).map(std::path::Path::new))?;
        Ok(ModelStatus { spec, downloaded: paths::is_downloaded(&dir, spec), path: Some(paths::model_path_in(&dir, spec)) })
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = models_dir;
        Ok(ModelStatus { spec, downloaded: false, path: None })
    }
}

/// Returns a backend for `provider`, or [`MlError::Unsupported`] when this build lacks it.
pub fn backend(provider: &Provider) -> Result<Box<dyn MattingBackend>> {
    match provider {
        Provider::Local { model, allow_download, models_dir } => {
            let spec = registry::find(model)?;
            local_backend(spec, *allow_download, models_dir.clone())
        }
        Provider::Remote { url, api_key } => remote_backend(url, api_key.clone()),
    }
}

#[cfg(all(feature = "onnx", not(target_arch = "wasm32"), not(all(target_os = "macos", target_arch = "x86_64"))))]
fn local_backend(spec: &'static ModelSpec, allow_download: bool, dir: Option<std::path::PathBuf>) -> Result<Box<dyn MattingBackend>> {
    Ok(Box::new(onnx::OnnxBackend::new(spec, allow_download, dir)))
}
#[cfg(not(all(feature = "onnx", not(target_arch = "wasm32"), not(all(target_os = "macos", target_arch = "x86_64")))))]
fn local_backend(spec: &'static ModelSpec, _allow_download: bool, _dir: Option<std::path::PathBuf>) -> Result<Box<dyn MattingBackend>> {
    Err(MlError::Unsupported(format!(
        "local AI model \"{}\" is unavailable: this build has no ONNX runtime (build with the `ml-onnx` feature, or use provider \"remote\")",
        spec.id
    )))
}

#[cfg(all(feature = "remote", not(target_arch = "wasm32")))]
fn remote_backend(url: &str, key: Option<String>) -> Result<Box<dyn MattingBackend>> {
    Ok(Box::new(remote::RemoteBackend::new(url, key)))
}
#[cfg(not(all(feature = "remote", not(target_arch = "wasm32"))))]
fn remote_backend(_url: &str, _key: Option<String>) -> Result<Box<dyn MattingBackend>> {
    Err(MlError::Unsupported("remote AI provider is unavailable in this build (build with the `ml-remote` feature)".into()))
}

/// Whether this build can run local models.
pub const HAS_LOCAL: bool = cfg!(all(feature = "onnx", not(target_arch = "wasm32"), not(all(target_os = "macos", target_arch = "x86_64"))));
/// Whether this build can call a remote provider.
pub const HAS_REMOTE: bool = cfg!(all(feature = "remote", not(target_arch = "wasm32")));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_options() {
        assert!(matches!(Provider::from_options(None, None, true, None), Ok(Provider::Local { .. })));
        assert!(Provider::from_options(Some("magic"), None, true, None).is_err());
        let p = Provider::from_options(Some("remote"), None, true, Some("http://127.0.0.1:1/x")).unwrap();
        assert!(matches!(p, Provider::Remote { .. }));
    }

    #[test]
    fn unknown_model_is_an_error() {
        let p = Provider::Local { model: "rmbg-2.0".into(), allow_download: false, models_dir: None };
        assert!(matches!(backend(&p), Err(MlError::UnknownModel(..))));
    }

    /// ONNX Runtime is pinned through ort-sys (see ORT_PINS.md): a lockfile bump must update the
    /// documented archive hashes and the NOTICE entry too.
    const PINNED_ORT_SYS: (&str, &str) = ("2.0.0-rc.13", "cf211e3776eea6aec988552fa118dd746d70e1b1e5e244058d1c98015f3e5872");

    #[test]
    fn ort_runtime_is_pinned() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let lock = std::fs::read_to_string(root.join("Cargo.lock")).unwrap();
        let block = lock.split("[[package]]").find(|b| b.contains("name = \"ort-sys\"")).unwrap();
        assert!(block.contains(&format!("version = \"{}\"", PINNED_ORT_SYS.0)), "{block}");
        assert!(block.contains(&format!("checksum = \"{}\"", PINNED_ORT_SYS.1)), "{block}");
        let pins = std::fs::read_to_string(root.join("crates/ml/ORT_PINS.md")).unwrap();
        assert!(pins.contains(PINNED_ORT_SYS.1) && pins.contains(PINNED_ORT_SYS.0));
        let notice = std::fs::read_to_string(root.join("NOTICE")).unwrap();
        assert!(notice.contains("ONNX Runtime 1.28.0"));
    }

    #[test]
    fn status_reports_size_license_and_custom_folder() {
        let st = model_status(Some("isnet"), Some("/nonexistent/photocraft-models")).unwrap();
        assert_eq!(st.spec.id, "isnet");
        assert!(!st.downloaded);
        assert_eq!(st.spec.license, "Apache-2.0");
        assert!(st.path.unwrap().starts_with("/nonexistent/photocraft-models"));
        assert!(model_status(Some("nope"), None).is_err());
        let p = Provider::from(&Options { model: Some("u2netp"), models_dir: Some("/tmp/x"), allow_download: true, ..Default::default() }).unwrap();
        assert!(matches!(p, Provider::Local { models_dir: Some(_), .. }));
    }

    #[test]
    fn rgb_image_validates_its_size() {
        assert!(RgbImage::new(0, 3, vec![]).is_err());
        assert!(RgbImage::new(2, 2, vec![[0.0; 3]; 3]).is_err());
        assert!(RgbImage::new(2, 2, vec![[0.0; 3]; 4]).is_ok());
    }
}
