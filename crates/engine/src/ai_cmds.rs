//! AI (ML) commands backed by `photocraft-ml`: model-based Remove Background and Select Subject.
//!
//! The pixels to analyse are read exactly like the classical `layer.removeBackground` /
//! `select.subject` (active layer, or the composite with `sampleAllLayers`), handed to a
//! [`photocraft_ml::MattingBackend`] (local ONNX model downloaded on first use, or a remote HTTP
//! provider), and the returned soft matte becomes a layer mask or a selection in one history step.
//! Builds without the `ml-onnx` / `ml-remote` features (and the wasm build) keep the commands
//! registered but return an actionable "unsupported" error, so menus, MCP and batch stay stable.

use photocraft_algo::matting::{self, RefineParams};
use photocraft_algo::selection::{Region, SelectionMode, combine_region};
use photocraft_doc::{Document, Layer, LayerContent, LayerId, LayerMask};
use photocraft_geom::Rect;
use photocraft_ml::{MattingBackend, Provider};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::jobs::JobCtx;
use crate::prefs::AiPrefs;
use crate::smartselect_cmds::with_doc_sampler;
use crate::{EngineError, Result, Session};

const REMOVE_BG: &str = "layer.removeBackgroundAI";
const SELECT_SUBJECT: &str = "select.subjectAI";
const STATUS: &str = "ai.modelStatus";

/// Longest side of the image handed to the model. Every bundled model runs at ≤1024 px, so a
/// 2048 px proxy loses nothing and keeps 50 MP documents cheap to read.
const PROXY_SIDE: usize = 2048;

/// Optional edge refinement on top of the model matte (off by default: the matte is already soft).
const REFINE: RefineParams = RefineParams { radius: 2.0, smart_radius: true, smooth: 0.0, feather: 0.0, contrast: 0.0, shift_edge: 0.0 };

struct Progress<'a>(&'a JobCtx);

impl photocraft_ml::Progress for Progress<'_> {
    fn progress(&self, fraction: f32, label: &str) {
        self.0.progress(fraction, label);
    }
    fn cancelled(&self) -> bool {
        self.0.cancelled()
    }
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn ml_err(e: photocraft_ml::MlError) -> EngineError {
    match e {
        photocraft_ml::MlError::Cancelled => EngineError::Cancelled,
        e => EngineError::Other(e.to_string()),
    }
}

/// Options shared by both commands.
struct Opts {
    backend: Box<dyn MattingBackend>,
    sample_all: bool,
    refine: bool,
    threshold: Option<f32>,
}

/// Provider options from the params, defaulting to Edit › Preferences › AI Models.
fn provider(cmd: &str, p: &Value, ai: &AiPrefs) -> Result<Provider> {
    let s = |k: &str| p.get(k).and_then(Value::as_str);
    let provider = s("provider").unwrap_or(if ai.use_cloud_provider { "remote" } else { "local" });
    // The stored API key only goes to the endpoint the user configured (Preferences or
    // PHOTOCRAFT_ML_REMOTE_URL), never to a `url` an agent or action passes in.
    let configured = |u: &str| u == ai.cloud_url || std::env::var("PHOTOCRAFT_ML_REMOTE_URL").is_ok_and(|e| e == u);
    let o = photocraft_ml::Options {
        send_key: s("url").is_none_or(|u| u.is_empty() || configured(u)),
        provider: Some(provider),
        model: Some(s("model").unwrap_or(ai.model.registry_id())),
        allow_download: p.get("allowDownload").and_then(Value::as_bool).unwrap_or(!ai.offline_mode),
        url: s("url").or(Some(ai.cloud_url.as_str())),
        models_dir: Some(ai.cache_folder.as_str()),
    };
    Provider::from(&o).map_err(|e| bad(cmd, e.to_string()))
}

fn opts(cmd: &str, p: &Value, ai: &AiPrefs, sample_all_default: bool) -> Result<Opts> {
    let threshold = match p.get("threshold") {
        None | Some(Value::Null) => None,
        Some(v) => {
            let t = v.as_f64().ok_or_else(|| bad(cmd, "threshold must be a number in 0..1"))? as f32;
            if !(0.0..=1.0).contains(&t) {
                return Err(bad(cmd, "threshold must be in 0..1"));
            }
            Some(t)
        }
    };
    let backend = make_backend(&provider(cmd, p, ai)?)?;
    Ok(Opts {
        backend,
        sample_all: p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(sample_all_default),
        refine: p.get("refine").and_then(Value::as_bool).unwrap_or(false),
        threshold,
    })
}

#[cfg(not(test))]
fn make_backend(p: &Provider) -> Result<Box<dyn MattingBackend>> {
    photocraft_ml::backend(p).map_err(ml_err)
}

/// Tests swap in a deterministic backend (model `"test:red"`: foreground = reddish pixels) so the
/// command plumbing is covered without a multi-hundred-MB model.
#[cfg(test)]
fn make_backend(p: &Provider) -> Result<Box<dyn MattingBackend>> {
    match p {
        Provider::Local { model, .. } if model == "test:red" => Ok(Box::new(tests::RedBackend)),
        _ => photocraft_ml::backend(p).map_err(ml_err),
    }
}

/// Matte values below this are background, above `1 - MATTE_SNAP` foreground. Models leave a
/// faint 1–3/255 haze over the background (IS-Net especially), invisible on its own but enough to
/// stop Image › Trim from finding the product's bounds; and a near-1 interior should be opaque.
const MATTE_SNAP: f32 = 0.02;

/// One matte sample (0..1) as a mask byte: a hard cut at `threshold` when given, else the soft
/// matte with the haze and near-opaque values snapped.
fn mask_value(a: f32, threshold: Option<f32>) -> u8 {
    let v = match threshold {
        Some(t) => (a >= t) as u8 as f32,
        None if a < MATTE_SNAP => 0.0,
        None if a > 1.0 - MATTE_SNAP => 1.0,
        None => a,
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Runs the model over the document and returns the matte as a document-space region, plus the
/// backend id.
fn find_subject(doc: &Document, id: Option<LayerId>, o: &Opts, ctx: &JobCtx) -> Result<Region> {
    ctx.progress(0.05, "Reading pixels");
    let (bounds, img) = with_doc_sampler(doc, id, o.sample_all, |smp, d| {
        let b = d.bounds();
        let side = (b.width().max(b.height())) as usize;
        let step = side.div_ceil(PROXY_SIDE).max(1);
        (b, smp.rgb_scaled(b, step))
    });
    if bounds.width() == 0 || bounds.height() == 0 {
        return Err(EngineError::Other("the document is empty".into()));
    }
    let input = photocraft_ml::RgbImage::new(img.w, img.h, img.px).map_err(ml_err)?;
    let matte = o.backend.matte(&input, &Progress(ctx)).map_err(ml_err)?;
    if ctx.cancelled() {
        return Err(EngineError::Cancelled);
    }
    let (w, h) = (bounds.width() as usize, bounds.height() as usize);
    let alpha: Vec<[f32; 1]> = matte.alpha.iter().map(|a| [*a]).collect();
    let full = photocraft_ml::prep::resize(&alpha, matte.width, matte.height, w, h).map_err(ml_err)?;
    let mask: Vec<u8> = full.iter().map(|a| mask_value(a[0], o.threshold)).collect();
    if mask.iter().all(|v| *v < 8) {
        return Err(EngineError::Other("no subject found".into()));
    }
    let mut region = Region { bbox: bounds, mask };
    if o.refine {
        ctx.progress(0.92, "Refining edge");
        let refined =
            with_doc_sampler(doc, id, o.sample_all, |smp, d| matting::refine_mask(smp, &matting::region_reader(&region), region.bbox, d.bounds(), &REFINE));
        region = refined.unwrap_or(region);
    }
    Ok(region)
}

fn bounds_json(r: Rect) -> [i32; 4] {
    [r.x0, r.y0, r.width() as i32, r.height() as i32]
}

/// Remove Background (AI) needs an unlocked pixel layer.
fn check(doc: &Document, l: &Layer) -> std::result::Result<(), String> {
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("the layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    let locks = doc.effective_locks(l.id);
    if locks.pixels || locks.all {
        return Err(format!("Could not complete your request because the layer \"{}\" is locked", l.name));
    }
    Ok(())
}

/// Builds without any backend (no `ml-onnx` / `ml-remote`, or wasm) keep the commands but grey
/// them out in menus and quick actions.
fn available() -> std::result::Result<(), String> {
    if cfg!(test) || photocraft_ml::HAS_LOCAL || photocraft_ml::HAS_REMOTE { Ok(()) } else { Err("AI models are not available in this build".into()) }
}

fn remove_bg_enabled(s: &Session) -> std::result::Result<(), String> {
    available()?;
    let d = s.active().ok_or("no document open")?;
    check(&d.doc, crate::active_layer_of(s)?)
}

fn remove_bg(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    check(doc, doc.layer(id).ok_or(EngineError::NoLayer(id))?).map_err(EngineError::Other)?;
    let o = opts(REMOVE_BG, p, &s.prefs().ai, false)?;
    let backend_id = o.backend.id();
    crate::jobs::edit_job(
        s,
        "Remove Background",
        move |doc, _, ctx| {
            let region = find_subject(doc, Some(id), &o, ctx)?;
            let coverage = region.mask.iter().map(|v| *v as f64).sum::<f64>() / (255.0 * region.mask.len().max(1) as f64);
            crate::extra_cmds::background_to_layer_for_mask(doc, id);
            let b = region.bbox;
            doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(LayerMask { surface: matting::region_surface(&region), ..LayerMask::reveal_all() });
            doc.selection = None;
            Ok((b, coverage))
        },
        move |(b, coverage)| json!({ "layer": id.0, "bounds": bounds_json(b), "coverage": (coverage * 1000.0).round() / 1000.0, "backend": backend_id }),
    )
}

fn select_enabled(s: &Session) -> std::result::Result<(), String> {
    available()?;
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn select_subject(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let active = d.active_layer;
    let m = p.get("mode").and_then(Value::as_str).unwrap_or("replace");
    let mode = SelectionMode::parse(if m == "new" { "replace" } else { m });
    let o = opts(SELECT_SUBJECT, p, &s.prefs().ai, true)?;
    let backend_id = o.backend.id();
    crate::jobs::edit_job(
        s,
        "Select Subject",
        move |doc, _, ctx| {
            let region = find_subject(doc, active, &o, ctx)?;
            let b = region.bbox;
            doc.selection = combine_region(doc.selection.as_ref(), Some(&region), mode);
            Ok(b)
        },
        move |b| json!({ "selected": true, "changed": true, "bounds": bounds_json(b), "backend": backend_id }),
    )
}

/// `ai.modelStatus`: what running an AI command would need, for the first-use download prompt and
/// agents: the resolved provider and model, its download size and license, and whether it is
/// cached. Read-only (never downloads).
fn model_status(s: &mut Session, p: &Value) -> Result<Value> {
    let ai = s.prefs().ai.clone();
    let cloud = p.get("provider").and_then(Value::as_str).map_or(ai.use_cloud_provider, |v| v == "remote");
    if cloud {
        let url = p.get("url").and_then(Value::as_str).filter(|u| !u.is_empty()).unwrap_or(ai.cloud_url.as_str());
        return Ok(json!({ "provider": "remote", "url": url, "available": photocraft_ml::HAS_REMOTE }));
    }
    let model = p.get("model").and_then(Value::as_str).unwrap_or(ai.model.registry_id());
    let st = photocraft_ml::model_status(Some(model), Some(&ai.cache_folder)).map_err(|e| bad(STATUS, e.to_string()))?;
    let m = st.spec;
    Ok(json!({
        "provider": "local",
        "available": photocraft_ml::HAS_LOCAL,
        "model": m.id,
        "label": m.label,
        "bytes": m.bytes,
        "license": m.license,
        "licenseUrl": m.license_url,
        "downloaded": st.downloaded,
        "location": st.path.map(|p| p.display().to_string()),
        "offline": ai.offline_mode,
    }))
}

/// Options shared by both commands' params docs.
macro_rules! common_params {
    () => {
        r##""model":"auto|birefnet-lite|birefnet|isnet|u2netp|modnet"?,"provider":"local|remote"?,"url":str?,"allowDownload":bool?,"threshold":0..1?,"refine":bool=false"##
    };
}

/// AI command specs.
pub fn specs() -> Vec<CommandSpec> {
    let rb = concat!(
        r##"{"layer":id?,"sampleAllLayers":bool=false,"##,
        common_params!(),
        r##"} → {layer,bounds,coverage,backend} (adds a layer mask from an AI matte; the model downloads on first use; the Background becomes a normal layer)"##
    );
    let ss = concat!(
        r##"{"sampleAllLayers":bool=true,"mode":"replace|add|subtract|intersect"="replace","##,
        common_params!(),
        r##"} → {selected,changed,bounds,backend}"##
    );
    vec![
        CommandSpec {
            id: REMOVE_BG,
            label: "Remove Background (AI)",
            menu: &[],
            shortcut: None,
            params: rb,
            enabled: remove_bg_enabled,
            run: remove_bg,
            journal: true,
        },
        CommandSpec {
            id: SELECT_SUBJECT,
            label: "Subject (AI)",
            menu: &[],
            shortcut: None,
            params: ss,
            enabled: select_enabled,
            run: select_subject,
            journal: true,
        },
        CommandSpec {
            id: STATUS,
            label: "AI Model Status",
            menu: &[],
            shortcut: None,
            params: r##"{"model":"auto|birefnet-lite|birefnet|isnet|u2netp|modnet"?,"provider":"local|remote"?,"url":str?} → {provider,available,model,label,bytes,license,licenseUrl,downloaded,location,offline} (defaults from Preferences › AI Models; never downloads)"##,
            enabled: |_| Ok(()),
            run: model_status,
            journal: false,
        },
    ]
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use photocraft_ml::{Matte, MlError};

    /// Foreground = pixels whose red clearly dominates.
    pub(crate) struct RedBackend;

    impl MattingBackend for RedBackend {
        fn id(&self) -> String {
            "test:red".into()
        }
        fn matte(&self, img: &photocraft_ml::RgbImage, _: &dyn photocraft_ml::Progress) -> std::result::Result<Matte, MlError> {
            Ok(Matte { width: img.width, height: img.height, alpha: img.px.iter().map(|p| if p[0] > p[2] + 0.2 { 1.0 } else { 0.0 }).collect() })
        }
    }

    fn session_with(w: u32, h: u32, depth: u32, px: impl Fn(u32, u32) -> [f32; 3]) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::with_capacity((w * h) as usize * n);
            for y in 0..h {
                for x in 0..w {
                    let c = px(x, y);
                    data.extend_from_slice(&c[..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    fn inside(x: i32, y: i32) -> bool {
        let (dx, dy) = (x as f32 + 0.5 - 100.0, y as f32 + 0.5 - 80.0);
        dx * dx + dy * dy <= 45.0 * 45.0
    }

    fn disc(depth: u32) -> Session {
        session_with(200, 160, depth, |x, y| if inside(x as i32, y as i32) { [0.9, 0.2, 0.1] } else { [0.2, 0.4, 0.8] })
    }

    fn active(s: &Session) -> LayerId {
        s.active().unwrap().active_layer.unwrap()
    }

    fn mask_at(s: &Session, x: i32, y: i32) -> f32 {
        s.active().unwrap().doc.layer(active(s)).unwrap().mask.as_ref().map_or(0.0, |m| m.value(x, y))
    }

    #[test]
    fn ai_remove_background_masks_the_subject_at_every_depth_and_undoes() {
        for depth in [8, 16, 32] {
            let mut s = disc(depth);
            let id = active(&s);
            let steps = s.active().unwrap().history.past_len();
            let r = s.execute(REMOVE_BG, json!({"model": "test:red"})).unwrap();
            assert_eq!(r["layer"], id.0);
            assert_eq!(r["backend"], "test:red");
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "depth {depth}");
            assert!(mask_at(&s, 100, 80) > 0.9 && mask_at(&s, 2, 2) < 0.1, "depth {depth}");
            let (mut i, mut u) = (0, 0);
            for y in 0..160 {
                for x in 0..200 {
                    let (a, b) = (mask_at(&s, x, y) >= 0.5, inside(x, y));
                    i += (a && b) as i32;
                    u += (a || b) as i32;
                }
            }
            assert!(i as f32 / u as f32 > 0.95, "depth {depth}: IoU {}", i as f32 / u as f32);
            assert_eq!(s.active().unwrap().doc.layer(id).unwrap().name, "Layer 0");
            s.execute("edit.undo", json!({})).unwrap();
            assert!(s.active().unwrap().doc.layer(id).unwrap().mask.is_none());
        }
    }

    #[test]
    fn ai_select_subject_sets_the_selection() {
        let mut s = disc(8);
        let r = s.execute(SELECT_SUBJECT, json!({"model": "test:red", "threshold": 0.5})).unwrap();
        assert_eq!(r["selected"], true);
        assert!(s.active().unwrap().doc.selection.is_some());
        // The layer itself is untouched.
        assert!(s.active().unwrap().doc.layer(active(&s)).unwrap().mask.is_none());
    }

    #[test]
    fn matte_haze_is_dropped_and_edges_kept() {
        assert_eq!(mask_value(0.004, None), 0);
        assert_eq!(mask_value(0.015, None), 0);
        assert_eq!(mask_value(0.5, None), 128);
        assert_eq!(mask_value(0.985, None), 255);
        assert_eq!(mask_value(f32::NAN, None), 0);
        assert_eq!(mask_value(0.4, Some(0.5)), 0);
        assert_eq!(mask_value(0.6, Some(0.5)), 255);
    }

    #[test]
    fn model_status_follows_preferences() {
        let mut s = Session::new();
        let r = s.execute(STATUS, json!({})).unwrap();
        assert_eq!(r["provider"], "local");
        assert!(["birefnet-lite", "isnet"].contains(&r["model"].as_str().unwrap()));
        s.execute("prefs.set", json!({"path": "ai.model", "value": "isnet"})).unwrap();
        s.execute("prefs.set", json!({"path": "ai.cacheFolder", "value": "/nonexistent/pc-models"})).unwrap();
        let r = s.execute(STATUS, json!({})).unwrap();
        assert_eq!(r["model"], "isnet");
        assert_eq!(r["license"], "Apache-2.0");
        assert_eq!(r["bytes"], 178_648_008u64);
        assert_eq!(r["downloaded"], false);
        assert!(r["location"].as_str().unwrap().starts_with("/nonexistent/pc-models"));
        assert_eq!(s.execute(STATUS, json!({"model": "u2netp"})).unwrap()["model"], "u2netp");
        assert!(s.execute(STATUS, json!({"model": "rmbg-2.0"})).is_err());
        s.execute("prefs.set", json!({"path": "ai.useCloudProvider", "value": true})).unwrap();
        s.execute("prefs.set", json!({"path": "ai.cloudUrl", "value": "http://127.0.0.1:9/matte"})).unwrap();
        let r = s.execute(STATUS, json!({})).unwrap();
        assert_eq!((r["provider"].as_str(), r["url"].as_str()), (Some("remote"), Some("http://127.0.0.1:9/matte")));
    }

    #[test]
    fn offline_preference_blocks_downloads() {
        let mut s = disc(8);
        s.execute("prefs.set", json!({"path": "ai.offlineMode", "value": true})).unwrap();
        s.execute("prefs.set", json!({"path": "ai.cacheFolder", "value": "/nonexistent/pc-models"})).unwrap();
        let e = s.execute(REMOVE_BG, json!({"model": "u2netp"})).unwrap_err().to_string();
        // Without the ONNX runtime the command fails earlier, with "no ONNX runtime".
        assert!(e.contains(if photocraft_ml::HAS_LOCAL { "download" } else { "ONNX" }), "{e}");
    }

    #[test]
    fn ai_commands_fail_gracefully() {
        // No subject: nothing changes.
        let mut s = session_with(32, 32, 8, |_, _| [0.5, 0.5, 0.5]);
        let before = s.active().unwrap().revision;
        assert!(s.execute(REMOVE_BG, json!({"model": "test:red"})).is_err());
        assert_eq!(s.active().unwrap().revision, before);
        // Bad params are errors, not panics.
        let mut s = disc(8);
        for p in [
            json!({"model": "rmbg-2.0"}),
            json!({"model": "test:red", "threshold": 3}),
            json!({"model": "test:red", "threshold": "x"}),
            json!({"provider": "magic"}),
            json!({"model": "test:red", "layer": 999_999}),
        ] {
            assert!(s.execute(REMOVE_BG, p.clone()).is_err(), "{p}");
        }
        // Without the ml features (the default test build) the real models report "unsupported";
        // with them and downloads disabled they report "not downloaded". Either way: an error.
        if !photocraft_ml::HAS_LOCAL {
            let e = s.execute(REMOVE_BG, json!({})).unwrap_err().to_string();
            assert!(e.contains("ONNX"), "{e}");
        }
        assert!(Session::new().execute(REMOVE_BG, json!({})).is_err());
        assert!(Session::new().execute(SELECT_SUBJECT, json!({})).is_err());
        // Locked and non-pixel layers are refused.
        let mut s = disc(8);
        s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
        assert!(!s.is_enabled(REMOVE_BG));
        assert!(s.execute(REMOVE_BG, json!({"model": "test:red"})).is_err());
    }
}
