//! First use of an AI command (Remove Background (AI), Select Subject (AI)): before a local model
//! is downloaded, say what will be downloaded (model, size, license, where it is kept) and ask.
//! "Download" runs the command with `allowDownload: true`; the download and the model run then show
//! in the usual job progress dialog. Agents and actions (`automation_input`), calls that pass
//! `allowDownload` explicitly, the cloud provider, offline mode and cached models skip the prompt.
use crate::{PhotocraftApp, state::DialogKind};
use serde_json::{Map, Value, json};

const MARK: &str = "__aiDownload";
/// The commands that may need a model download.
pub const COMMANDS: &[(&str, &str)] = &[("layer.removeBackgroundAI", "Remove Background (AI)"), ("select.subjectAI", "Select Subject (AI)")];

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// `224 MB`-style size (decimal megabytes, as download pages show them).
pub fn size_label(bytes: u64) -> String {
    if bytes >= 1_000_000_000 { format!("{:.2} GB", bytes as f64 / 1e9) } else { format!("{:.0} MB", (bytes as f64 / 1e6).max(1.0)) }
}

/// The download the command would need, or `None` when it can run straight away.
fn pending_download(app: &mut PhotocraftApp, command: &str, params: &Value) -> Option<Value> {
    if app.automation_input || params.get("allowDownload").is_some() || !COMMANDS.iter().any(|(id, _)| *id == command) {
        return None;
    }
    let mut q = Map::new();
    for k in ["model", "provider", "url"] {
        if let Some(v) = params.get(k) {
            q.insert(k.into(), v.clone());
        }
    }
    let st = app.session.execute("ai.modelStatus", Value::Object(q)).ok()?;
    let local = st.get("provider").and_then(Value::as_str) == Some("local");
    let ready = st.get("downloaded").and_then(Value::as_bool) != Some(false);
    let usable = st.get("available").and_then(Value::as_bool) == Some(true);
    let offline = st.get("offline").and_then(Value::as_bool) == Some(true);
    (local && usable && !ready && !offline).then_some(st)
}

pub fn intercept(app: &mut PhotocraftApp, command: &str, params: &Value) -> Option<Result<Value, String>> {
    let st = pending_download(app, command, params)?;
    if app.ui.dialogs.iter().any(|d| owns(&d.fields)) {
        return Some(Ok(json!({"downloadPending": true})));
    }
    let label = COMMANDS.iter().find(|(id, _)| *id == command).map_or("AI", |(_, l)| *l);
    let fields = json!({MARK: true, "__command": command, "__label": label, "params": params,
        "model": st["label"], "bytes": st["bytes"], "license": st["license"], "licenseUrl": st["licenseUrl"], "path": st["location"]});
    let id = app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default());
    Some(Ok(json!({"dialog": id, "downloadPending": true})))
}

pub fn body(ui: &mut egui::Ui, fields: &Map<String, Value>) {
    let t = crate::theme::Tokens::get(ui.ctx());
    let s = |k: &str| fields.get(k).and_then(Value::as_str).unwrap_or_default();
    let size = size_label(fields.get("bytes").and_then(Value::as_u64).unwrap_or(0));
    ui.set_max_width(460.0);
    ui.label(crate::i18n::fmt(tl!("This uses the AI model {model}, which isn't on this computer yet."), &[("model", s("model"))]));
    ui.add_space(6.0);
    ui.label(crate::i18n::fmt(tl!("Download size: {size}"), &[("size", &size)]));
    ui.horizontal(|ui| {
        ui.label(crate::i18n::fmt(tl!("License: {license}"), &[("license", s("license"))]));
        if !s("licenseUrl").is_empty() {
            ui.hyperlink_to(tl!("View license"), s("licenseUrl"));
        }
    });
    if !s("path").is_empty() {
        ui.label(egui::RichText::new(crate::i18n::fmt(tl!("Saved to: {path}"), &[("path", s("path"))])).color(t.text_dim).size(11.5));
    }
    ui.add_space(6.0);
    ui.label(egui::RichText::new(tl!("The model runs on this computer; your images are not uploaded.")).color(t.text_dim));
    ui.label(egui::RichText::new(tl!("Choose the model, the cache folder or offline mode in Preferences › AI Models.")).color(t.text_faint).size(11.5));
}

pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let command = fields.get("__command").and_then(Value::as_str).ok_or(tl!("Missing command"))?;
    if !COMMANDS.iter().any(|(id, _)| *id == command) {
        return Err(tl!("Invalid command").into());
    }
    let mut params = fields.get("params").and_then(Value::as_object).cloned().unwrap_or_default();
    params.insert("allowDownload".into(), json!(true));
    app.run_command(command, Value::Object(params))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app_with_cache(dir: &str) -> PhotocraftApp {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width":8,"height":8})).unwrap();
        s.execute("prefs.set", json!({"path": "ai.cacheFolder", "value": dir})).unwrap();
        PhotocraftApp::new(s, crate::Services::default())
    }

    #[test]
    fn sizes_read_like_download_pages() {
        assert_eq!(size_label(224_005_088), "224 MB");
        assert_eq!(size_label(972_666_916), "973 MB");
        assert_eq!(size_label(1_500_000_000), "1.50 GB");
        assert_eq!(size_label(10), "1 MB");
    }

    #[test]
    fn first_use_asks_before_downloading_and_shows_size_and_license() {
        let mut app = app_with_cache("/nonexistent/pc-ai-prompt");
        let available = app.session.execute("ai.modelStatus", json!({})).unwrap()["available"] == json!(true);
        let r = app.run("layer.removeBackgroundAI", json!({"model": "isnet"}));
        if !available {
            // A build without the ONNX runtime has nothing to download: no prompt, the command
            // reports that it is unavailable.
            assert!(r.is_err() && app.ui.dialogs.is_empty(), "{r:?}");
            return;
        }
        let r = r.unwrap();
        assert_eq!(r["downloadPending"], true);
        let f = &app.ui.dialogs[0].fields;
        assert!(owns(f));
        assert_eq!(f["model"], "IS-Net general use (DIS)");
        assert_eq!(f["license"], "Apache-2.0");
        assert_eq!(f["bytes"], 178_648_008u64);
        assert!(f["path"].as_str().unwrap().starts_with("/nonexistent/pc-ai-prompt"));
        // A second click doesn't stack another prompt.
        assert_eq!(app.run("layer.removeBackgroundAI", json!({"model": "isnet"})).unwrap()["downloadPending"], true);
        assert_eq!(app.ui.dialogs.len(), 1);
    }

    #[test]
    fn automation_explicit_flags_and_offline_skip_the_prompt() {
        let mut app = app_with_cache("/nonexistent/pc-ai-prompt");
        assert!(pending_download(&mut app, "layer.removeBackgroundAI", &json!({"allowDownload": false})).is_none());
        assert!(pending_download(&mut app, "layer.removeBackground", &json!({})).is_none());
        assert!(pending_download(&mut app, "select.subjectAI", &json!({"provider": "remote", "url": "http://127.0.0.1:9"})).is_none());
        app.automation_input = true;
        assert!(pending_download(&mut app, "select.subjectAI", &json!({})).is_none());
        app.automation_input = false;
        app.session.execute("prefs.set", json!({"path": "ai.offlineMode", "value": true})).unwrap();
        assert!(pending_download(&mut app, "select.subjectAI", &json!({})).is_none());
    }
}
