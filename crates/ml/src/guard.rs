//! A breadcrumb around local inference, so a run that takes the whole process down leaves a
//! record and the next run avoids repeating it.
//!
//! ONNX Runtime is C++: an allocation failure inside it usually comes back as an error ("bad
//! allocation"), but some failures abort the process, which no Rust code can catch. So before a
//! model runs, [`begin`] writes `ai-run.inprogress` into the config dir ([`paths::config_dir`])
//! naming the model, the image size and the memory the machine reported; the returned guard
//! deletes it when the run ends (success or error). If the file is still there when the next
//! process starts an AI run, that run never finished: [`previous_crash`] appends the record to
//! `logs/ai-crash.log` (Windows: `%APPDATA%\ForkItUp\logs\ai-crash.log`; macOS:
//! `~/Library/Application Support/ForkItUp/logs/`; Linux: `~/.config/forkitup/logs/`), and sets
//! the `ai-low-memory` flag, which makes Auto choose the low-memory model from then on.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{paths, registry};

const MARKER: &str = "ai-run.inprogress";
const FLAG: &str = "ai-low-memory";
/// The crash log, relative to the config dir (next to the app's own log files).
pub const CRASH_LOG: &str = "logs/ai-crash.log";

fn dir() -> Option<PathBuf> {
    paths::config_dir().ok()
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Deletes the marker when dropped.
pub struct RunGuard(Option<PathBuf>);

impl Drop for RunGuard {
    fn drop(&mut self) {
        if let Some(p) = self.0.take() {
            let _ = std::fs::remove_file(p);
        }
    }
}

/// The marker's text for one run.
pub fn record(model: &str, width: usize, height: usize, total_mb: Option<u64>, available_mb: Option<u64>) -> String {
    let mb = |v: Option<u64>| v.map_or_else(|| "unknown".to_string(), |v| format!("{v} MB"));
    format!(
        "unix_time={} version={} os={}-{} model={model} image={width}x{height} total_memory={} available_memory={}",
        now(),
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH,
        mb(total_mb),
        mb(available_mb)
    )
}

/// Writes the in-progress marker for a run of `model` on a `width`×`height` image.
pub fn begin(model: &str, width: usize, height: usize) -> RunGuard {
    let Some(d) = dir() else { return RunGuard(None) };
    let _ = std::fs::create_dir_all(&d);
    let p = d.join(MARKER);
    let text = record(model, width, height, registry::total_mb(), registry::available_mb());
    match std::fs::write(&p, text) {
        Ok(()) => RunGuard(Some(p)),
        Err(_) => RunGuard(None),
    }
}

/// If an earlier process left its marker (it ended mid-run), log it once per process and return
/// a short user-facing note. Sets the low-memory flag.
pub fn previous_crash() -> Option<&'static str> {
    static ONCE: OnceLock<Option<String>> = OnceLock::new();
    ONCE.get_or_init(|| {
        let d = dir()?;
        let marker = d.join(MARKER);
        let text = std::fs::read_to_string(&marker).ok()?;
        let _ = std::fs::remove_file(&marker);
        let log = d.join(CRASH_LOG);
        if let Some(parent) = log.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let line = format!("unix_time={} an AI run did not finish (the app closed during inference, most likely out of memory): {}\n", now(), text.trim());
        let _ = std::fs::OpenOptions::new().create(true).append(true).open(&log).and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
        let _ = std::fs::write(d.join(FLAG), "set after an AI run did not finish; delete to let Auto pick the default model again\n");
        Some(format!(
            "The last AI run closed the app before it finished (most likely out of memory), so Auto now uses the low-memory model. Details: {}",
            log.display()
        ))
    })
    .as_deref()
}

/// Whether an earlier run crashed on this machine (see the module docs).
pub fn low_memory_flag() -> bool {
    dir().is_some_and(|d| d.join(FLAG).is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_names_the_model_image_and_memory() {
        let r = record("birefnet-lite", 480, 640, Some(7800), None);
        assert!(r.contains("model=birefnet-lite") && r.contains("image=480x640"), "{r}");
        assert!(r.contains("total_memory=7800 MB") && r.contains("available_memory=unknown"), "{r}");
    }
}
