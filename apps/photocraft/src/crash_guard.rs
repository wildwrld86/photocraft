//! Last-resort crash guard (AGENTS.md, Never crash).
//!
//! Code must return errors, never panic. This is the safety net for a panic that escapes anyway:
//! the hook logs it, and [`guard`] turns a panic during file import/export into an error the UI
//! shows, so the open documents stay in place. Commands get the same treatment in
//! `Session::execute`.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// The crash log, relative to the settings directory: `%APPDATA%\ForkItUp\logs\crash.log` on
/// Windows, `~/Library/Application Support/ForkItUp/logs/crash.log` on macOS,
/// `~/.config/forkitup/logs/crash.log` on Linux (or next to the exe in portable mode). Panics are
/// appended here as well as to the session log in the same folder. AI runs that
/// end the process are recorded in `logs/ai-crash.log` (`photocraft_ml::guard`).
pub const CRASH_LOG: &str = "logs/crash.log";

/// Log every panic (with its location) before the default hook prints it, and append it to
/// [`CRASH_LOG`].
pub fn install_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log::error!("Fork It Up internal error: {info}");
        if let Some(dir) = crate::services::config_dir() {
            append_crash(&dir.join(CRASH_LOG), &crash_line(&info.to_string()));
        }
        default(info);
    }));
}

fn crash_line(what: &str) -> String {
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("unix_time={t} version={} internal error: {}\n", photocraft_engine::build_info::long_version(), what.replace('\n', " "))
}

/// Best effort: a crash log that can't be written must not cause a second panic.
fn append_crash(path: &std::path::Path, line: &str) {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = f.write_all(line.as_bytes());
    }
}

/// Run `f`; a panic inside it becomes an `Err` naming `what`.
pub fn guard<T>(what: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err(format!("{what} failed with an internal error (logged); your open documents are unchanged")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crash_lines_are_appended() {
        let dir = std::env::temp_dir().join(format!("fiu-crash-{}", std::process::id()));
        let path = dir.join(CRASH_LOG);
        append_crash(&path, &crash_line("boom\nat main.rs:1"));
        append_crash(&path, &crash_line("second"));
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 2, "{text}");
        assert!(text.contains("internal error: boom at main.rs:1"), "{text}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn guard_turns_a_panic_into_an_error() {
        let r: Result<(), String> = guard("Open", || panic!("boom"));
        assert!(r.unwrap_err().contains("Open failed"));
        assert_eq!(guard("Open", || Ok(3)), Ok(3));
        assert_eq!(guard::<()>("Open", || Err("bad file".into())), Err("bad file".into()));
    }
}
