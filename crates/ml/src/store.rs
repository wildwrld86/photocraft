//! On-demand model store: models live in the user's cache directory, are fetched over HTTPS on
//! first use, written to a `.part` file, verified (size + SHA-256) and only then renamed into
//! place. Nothing is vendored in the repository or the installer.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::registry::ModelSpec;
use crate::{MlError, Progress, Result, paths};

/// Returns the local path of `spec`, downloading and verifying it first if needed.
pub fn ensure(spec: &ModelSpec, dir: Option<&Path>, allow_download: bool, progress: &dyn Progress) -> Result<PathBuf> {
    let path = paths::model_path_in(&paths::models_dir(dir)?, spec);
    if fs::metadata(&path).is_ok_and(|m| m.len() == spec.bytes) {
        return Ok(path);
    }
    if !allow_download {
        return Err(MlError::NotDownloaded(spec.id.into(), format!("expected at {}", path.display())));
    }
    let dir = path.parent().ok_or_else(|| MlError::Download("bad model path".into()))?;
    fs::create_dir_all(dir).map_err(|e| MlError::Download(format!("{}: {e}", dir.display())))?;
    let part = path.with_extension("onnx.part");
    let r = download(spec, &part, progress);
    if r.is_err() {
        let _ = fs::remove_file(&part);
    }
    r?;
    fs::rename(&part, &path).map_err(|e| MlError::Download(format!("rename: {e}")))?;
    Ok(path)
}

fn download(spec: &ModelSpec, part: &Path, progress: &dyn Progress) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder().user_agent("Photocraft-dev").build().into();
    let resp = agent.get(spec.url).call().map_err(|e| MlError::Download(format!("{}: {e}", spec.url)))?;
    let mut reader = resp.into_body().into_reader();
    let mut file = fs::File::create(part).map_err(|e| MlError::Download(format!("{}: {e}", part.display())))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut total: u64 = 0;
    loop {
        if progress.cancelled() {
            return Err(MlError::Cancelled);
        }
        let n = reader.read(&mut buf).map_err(|e| MlError::Download(e.to_string()))?;
        if n == 0 {
            break;
        }
        let chunk = buf.get(..n).ok_or_else(|| MlError::Download("read overflow".into()))?;
        total = total.saturating_add(n as u64);
        if total > spec.bytes {
            return Err(MlError::Download(format!("{} is larger than the pinned {} bytes", spec.id, spec.bytes)));
        }
        hasher.update(chunk);
        file.write_all(chunk).map_err(|e| MlError::Download(e.to_string()))?;
        progress.progress((total as f32 / spec.bytes.max(1) as f32) * 0.5, "Downloading AI model");
    }
    file.flush().map_err(|e| MlError::Download(e.to_string()))?;
    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if total != spec.bytes || digest != spec.sha256 {
        return Err(MlError::Download(format!("{}: checksum mismatch ({total} bytes, sha256 {digest})", spec.id)));
    }
    Ok(())
}
