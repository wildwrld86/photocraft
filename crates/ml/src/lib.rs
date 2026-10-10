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

pub mod prep;
pub mod registry;
#[cfg(all(feature = "download", not(target_arch = "wasm32")))]
pub mod store;
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub mod onnx;
#[cfg(all(feature = "remote", not(target_arch = "wasm32")))]
pub mod remote;

pub use registry::{ModelSpec, MODELS, DEFAULT_MODEL};

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
    /// Local ONNX Runtime with a registry model id.
    Local { model: String, allow_download: bool },
    /// Remote HTTP endpoint.
    Remote { url: String, api_key: Option<String> },
}

impl Provider {
    /// Builds a provider from command-style options. `provider` is `"local"` (default) or
    /// `"remote"`; the remote URL and key fall back to `PHOTOCRAFT_ML_REMOTE_URL` /
    /// `PHOTOCRAFT_ML_REMOTE_KEY`. Downloads can be disabled globally with `PHOTOCRAFT_ML_OFFLINE=1`.
    pub fn from_options(provider: Option<&str>, model: Option<&str>, allow_download: bool, url: Option<&str>) -> Result<Self> {
        let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
        match provider.unwrap_or("local") {
            "local" => {
                let offline = env("PHOTOCRAFT_ML_OFFLINE").is_some_and(|v| v != "0");
                Ok(Provider::Local { model: registry::resolve(model).to_string(), allow_download: allow_download && !offline })
            }
            "remote" => {
                let url = url.map(str::to_string).or_else(|| env("PHOTOCRAFT_ML_REMOTE_URL")).ok_or_else(|| {
                    MlError::Unsupported("the remote provider needs a \"url\" param or PHOTOCRAFT_ML_REMOTE_URL".into())
                })?;
                Ok(Provider::Remote { url, api_key: env("PHOTOCRAFT_ML_REMOTE_KEY") })
            }
            other => Err(MlError::Unsupported(format!("unknown provider \"{other}\" (local|remote)"))),
        }
    }
}

/// Returns a backend for `provider`, or [`MlError::Unsupported`] when this build lacks it.
pub fn backend(provider: &Provider) -> Result<Box<dyn MattingBackend>> {
    match provider {
        Provider::Local { model, allow_download } => {
            let spec = registry::find(model)?;
            local_backend(spec, *allow_download)
        }
        Provider::Remote { url, api_key } => remote_backend(url, api_key.clone()),
    }
}

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
fn local_backend(spec: &'static ModelSpec, allow_download: bool) -> Result<Box<dyn MattingBackend>> {
    Ok(Box::new(onnx::OnnxBackend::new(spec, allow_download)))
}
#[cfg(not(all(feature = "onnx", not(target_arch = "wasm32"))))]
fn local_backend(spec: &'static ModelSpec, _allow_download: bool) -> Result<Box<dyn MattingBackend>> {
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
pub const HAS_LOCAL: bool = cfg!(all(feature = "onnx", not(target_arch = "wasm32")));
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
        let p = Provider::Local { model: "rmbg-2.0".into(), allow_download: false };
        assert!(matches!(backend(&p), Err(MlError::UnknownModel(..))));
    }

    #[test]
    fn rgb_image_validates_its_size() {
        assert!(RgbImage::new(0, 3, vec![]).is_err());
        assert!(RgbImage::new(2, 2, vec![[0.0; 3]; 3]).is_err());
        assert!(RgbImage::new(2, 2, vec![[0.0; 3]; 4]).is_ok());
    }
}
