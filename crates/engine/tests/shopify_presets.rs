//! The Shopify listing presets in `examples/actions/` (AI cutout centred on a 2048×2048 square,
//! product ≈85% of the side): every step is a live command and the geometry is right. The AI step
//! itself needs a model, so here a drawn product on a transparent layer stands in for its output
//! (`removeBackgroundAI` + `layerMask.apply` leave exactly that); the AI step runs in the CLI batch.

use photocraft_engine::Session;
use serde_json::{Value, json};
use std::path::Path;

fn preset(name: &str) -> Vec<(String, Value)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/actions").join(name);
    let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    photocraft_engine::automate_cmds::parse_action(v.get("actions").unwrap_or(&v), "test").unwrap()
}

/// A 400×300 transparent document with a 200×100 red "product" at (50, 40).
fn cutout() -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 400, "height": 300, "background": "transparent"})).unwrap();
    s.execute("select.rect", json!({"x": 50, "y": 40, "width": 200, "height": 100})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s
}

fn run(s: &mut Session, steps: &[(String, Value)]) {
    assert_eq!(steps[0].0, "layer.removeBackgroundAI");
    assert_eq!(steps[1].0, "layer.layerMask.apply");
    for (id, p) in &steps[2..] {
        s.execute(id, p.clone()).unwrap_or_else(|e| panic!("{id}: {e}"));
    }
}

fn composite_rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let st = s.active().unwrap();
    let mut out = [0.0; 4];
    // Top-most opaque-ish layer at the pixel (the white result is flattened to one layer).
    for l in st.doc.layers.iter().rev() {
        if let Some(surf) = l.surface() {
            let px = surf.rgba(x, y);
            if px[3] > 0.0 {
                out = px;
                break;
            }
        }
    }
    out
}

fn red_extent(s: &Session) -> (i32, i32, i32, i32) {
    let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
    for y in 0..2048 {
        let p = composite_rgba(s, 1024, y);
        if p[0] > 0.9 && p[1] < 0.1 && p[3] > 0.9 {
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
    }
    for x in 0..2048 {
        let p = composite_rgba(s, x, 1024);
        if p[0] > 0.9 && p[1] < 0.1 && p[3] > 0.9 {
            x0 = x0.min(x);
            x1 = x1.max(x);
        }
    }
    (x0, y0, x1, y1)
}

#[test]
fn white_square_preset_centres_the_product_at_85_percent() {
    let steps = preset("shopify-ai-white-2048.json");
    let mut s = cutout();
    run(&mut s, &steps);
    let b = s.active().unwrap().doc.bounds();
    assert_eq!((b.width(), b.height()), (2048, 2048));
    assert_eq!(s.active().unwrap().doc.layers.len(), 1, "flattened");
    for (x, y) in [(0, 0), (2047, 0), (0, 2047), (2047, 2047), (1024, 100)] {
        let p = composite_rgba(&s, x, y);
        assert!(p.iter().all(|c| (c - 1.0).abs() < 1e-3), "({x},{y}) not pure white: {p:?}");
    }
    let (x0, y0, x1, y1) = red_extent(&s);
    let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
    assert!((w - 1741).abs() <= 2, "product width {w}");
    assert!((h - 870).abs() <= 3, "product height {h}");
    assert!(((x0 + x1) / 2 - 1024).abs() <= 2 && ((y0 + y1) / 2 - 1024).abs() <= 2, "centred: {x0},{y0}..{x1},{y1}");
}

#[test]
fn transparent_preset_keeps_the_background_clear() {
    let steps = preset("shopify-ai-transparent-2048.json");
    let mut s = cutout();
    run(&mut s, &steps);
    let b = s.active().unwrap().doc.bounds();
    assert_eq!((b.width(), b.height()), (2048, 2048));
    for (x, y) in [(0, 0), (2047, 2047), (1024, 100)] {
        assert_eq!(composite_rgba(&s, x, y)[3], 0.0, "({x},{y}) should be transparent");
    }
    let (x0, _, x1, _) = red_extent(&s);
    assert!((x1 - x0 + 1 - 1741).abs() <= 2);
}

#[test]
fn preset_steps_are_live_commands() {
    for name in ["shopify-ai-white-2048.json", "shopify-ai-transparent-2048.json"] {
        for (id, _) in preset(name) {
            assert!(photocraft_engine::commands::command_specs().iter().any(|c| c.id == id), "{name}: {id}");
        }
    }
}
