//! Where the ML layer keeps things on disk (std only, no feature needed): the model cache and the
//! cloud API key file. Native only; the wasm build has no file system here.

use std::path::{Path, PathBuf};

use crate::registry::ModelSpec;
use crate::{MlError, Result};

/// The app's folder name under the platform config and cache dirs. Matches the desktop app's own
/// data folder (`apps/photocraft/src/app_dirs.rs`): `ForkItUp` on macOS and Windows, `forkitup`
/// on Linux and the BSDs.
pub const APP_DIR: &str = if cfg!(any(target_os = "macos", target_os = "windows")) { "ForkItUp" } else { "forkitup" };

fn env(k: &str) -> Option<PathBuf> {
    std::env::var_os(k).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// Default model cache directory: `PHOTOCRAFT_MODELS_DIR`, else the platform cache dir
/// (`$XDG_CACHE_HOME`/`~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on
/// Windows) + [`APP_DIR`]`/models`.
pub fn default_models_dir() -> Result<PathBuf> {
    if let Some(d) = env("PHOTOCRAFT_MODELS_DIR") {
        return Ok(d);
    }
    let base = if cfg!(target_os = "windows") {
        env("LOCALAPPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Caches"))
    } else {
        env("XDG_CACHE_HOME").or_else(|| env("HOME").map(|h| h.join(".cache")))
    };
    base.map(|b| b.join(APP_DIR).join("models")).ok_or_else(|| MlError::Download("no cache directory (set PHOTOCRAFT_MODELS_DIR)".into()))
}

/// The model cache to use: `custom` when set (Preferences › AI Models › Cache folder), else the
/// default.
pub fn models_dir(custom: Option<&Path>) -> Result<PathBuf> {
    match custom {
        Some(d) if !d.as_os_str().is_empty() => Ok(d.to_path_buf()),
        _ => default_models_dir(),
    }
}

/// Path where `spec` is (or will be) stored in `dir`.
pub fn model_path_in(dir: &Path, spec: &ModelSpec) -> PathBuf {
    dir.join(format!("{}-{}.onnx", spec.id, spec.sha256.get(..12).unwrap_or(spec.sha256)))
}

/// Is `spec` already downloaded (complete, by size) in `dir`?
pub fn is_downloaded(dir: &Path, spec: &ModelSpec) -> bool {
    std::fs::metadata(model_path_in(dir, spec)).is_ok_and(|m| m.len() == spec.bytes)
}

/// Per-user config directory for Fork It Up's AI settings that must not live in the preferences
/// JSON: `PHOTOCRAFT_CONFIG_DIR`, else `$XDG_CONFIG_HOME`/`~/.config` (Linux),
/// `~/Library/Application Support` (macOS), `%APPDATA%` (Windows) + [`APP_DIR`], the same folder
/// the desktop app keeps its preferences in (outside portable mode).
pub fn config_dir() -> Result<PathBuf> {
    if let Some(d) = env("PHOTOCRAFT_CONFIG_DIR") {
        return Ok(d);
    }
    let base = if cfg!(target_os = "windows") {
        env("APPDATA")
    } else if cfg!(target_os = "macos") {
        env("HOME").map(|h| h.join("Library/Application Support"))
    } else {
        env("XDG_CONFIG_HOME").or_else(|| env("HOME").map(|h| h.join(".config")))
    };
    base.map(|b| b.join(APP_DIR)).ok_or_else(|| MlError::Unsupported("no config directory (set PHOTOCRAFT_CONFIG_DIR)".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn folders_follow_the_fork_name() {
        assert!(APP_DIR.eq_ignore_ascii_case("forkitup"));
        if std::env::var_os("PHOTOCRAFT_CONFIG_DIR").is_none()
            && let Ok(d) = config_dir()
        {
            assert!(d.ends_with(APP_DIR), "{}", d.display());
        }
        if std::env::var_os("PHOTOCRAFT_MODELS_DIR").is_none()
            && let Ok(d) = default_models_dir()
        {
            assert!(d.ends_with(Path::new(APP_DIR).join("models")), "{}", d.display());
        }
    }
}
