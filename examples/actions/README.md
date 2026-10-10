# Action presets

Run with `photocraft-cli batch --actions <file> --in <folder> --out <folder> --format png|jpg`, or
load them in the app (File › Automate › Batch).

| File | What it makes |
|---|---|
| `shopify-ai-white-2048.json` | Shopify main product image: AI background removal (`layer.removeBackgroundAI`, BiRefNet-lite by default), the cutout trimmed and scaled with Lanczos so its longer side is 1741 px (85% of 2048), centred on a 2048×2048 pure-white (#ffffff) square, flattened. Export as JPEG or PNG. |
| `shopify-ai-transparent-2048.json` | The same cutout and placement on a transparent 2048×2048 canvas. Export as PNG (keeps alpha) for collection tiles, theme sections or compositing. |

The AI step downloads its model on first use (224 MB for BiRefNet-lite, MIT; see Preferences ›
AI Models). Batch and CLI runs don't prompt; they download on first use. Set
`PHOTOCRAFT_ML_OFFLINE=1` to forbid downloads, or `PHOTOCRAFT_MODELS_DIR` to use a pre-filled
cache. To pin a model, edit the first step, e.g. `["layer.removeBackgroundAI",{"model":"isnet"}]`.
