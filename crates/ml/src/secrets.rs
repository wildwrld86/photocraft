//! The cloud provider's API key.
//!
//! PhotoCraft has no OS keychain integration (the app stores no other secrets: the control-channel
//! token is passed on the command line or in a file). So the key is kept **out of the preferences
//! JSON** (which `prefs.get`, exports and agents can read) in its own file,
//! `<config dir>/ai-cloud-key`, created with owner-only permissions (0600) on Unix. On Windows it
//! inherits the per-user `%APPDATA%` ACL. `PHOTOCRAFT_ML_REMOTE_KEY` overrides the file. Moving it
//! into the macOS Keychain / Windows Credential Manager / Secret Service is a follow-up.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::{MlError, Result, paths};

const FILE: &str = "ai-cloud-key";
const MAX_KEY: usize = 4096;

/// Path of the key file.
pub fn cloud_key_path() -> Result<PathBuf> {
    Ok(paths::config_dir()?.join(FILE))
}

/// The key: the environment override, else the stored file (trimmed), else `None`.
pub fn cloud_key() -> Option<String> {
    if let Some(k) = std::env::var("PHOTOCRAFT_ML_REMOTE_KEY").ok().filter(|k| !k.is_empty()) {
        return Some(k);
    }
    read_key(&cloud_key_path().ok()?)
}

fn read_key(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let k = text.trim();
    (!k.is_empty()).then(|| k.to_string())
}

/// Is a key stored (or set in the environment)?
pub fn has_cloud_key() -> bool {
    cloud_key().is_some()
}

/// Stores `key` (owner-only file). An empty key clears it.
pub fn set_cloud_key(key: &str) -> Result<()> {
    write_key(&cloud_key_path()?, key)
}

fn write_key(path: &Path, key: &str) -> Result<()> {
    let key = key.trim();
    if key.is_empty() {
        return remove_key(path);
    }
    if key.len() > MAX_KEY || key.chars().any(char::is_control) {
        return Err(MlError::BadInput("the API key is too long or contains control characters".into()));
    }
    let dir = path.parent().ok_or_else(|| MlError::BadInput("bad config path".into()))?;
    fs::create_dir_all(dir).map_err(|e| MlError::BadInput(format!("{}: {e}", dir.display())))?;
    let tmp = path.with_extension("tmp");
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut f = opts.open(&tmp).map_err(|e| MlError::BadInput(format!("{}: {e}", tmp.display())))?;
    f.write_all(key.as_bytes()).and_then(|_| f.sync_all()).map_err(|e| MlError::BadInput(e.to_string()))?;
    drop(f);
    fs::rename(&tmp, path).map_err(|e| MlError::BadInput(e.to_string()))?;
    Ok(())
}

/// Removes the stored key (no error if there is none).
pub fn clear_cloud_key() -> Result<()> {
    remove_key(&cloud_key_path()?)
}

fn remove_key(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(MlError::BadInput(format!("{}: {e}", path.display()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_file_round_trip_is_owner_only() {
        let dir = std::env::temp_dir().join(format!("pc-ml-key-{}", std::process::id()));
        let path = dir.join(FILE);
        write_key(&path, "  sk-test-123 \n").unwrap();
        assert_eq!(read_key(&path).as_deref(), Some("sk-test-123"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
        assert!(write_key(&path, "bad\u{7}key").is_err());
        write_key(&path, "").unwrap();
        assert_eq!(read_key(&path), None);
        remove_key(&path).unwrap();
        let _ = fs::remove_dir_all(dir);
    }
}
