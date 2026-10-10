# photocraft-ml

L4 ML layer: a model registry, on-demand model download and matting backends behind one trait.

```rust
let provider = Provider::from_options(Some("local"), Some("auto"), /*allow_download*/ true, None)?;
let backend = photocraft_ml::backend(&provider)?;     // Box<dyn MattingBackend>
let matte = backend.matte(&rgb_image, &NoProgress)?;  // soft alpha, same size as the input
```

| Feature | What it adds | Targets |
|---|---|---|
| *(none)* | registry, pre/post-processing; `backend()` returns `MlError::Unsupported` | all, incl. wasm32 |
| `onnx` | local inference with ONNX Runtime (`ort`, statically linked) + first-use download | native |
| `remote` | HTTP provider: `POST image/png` → PNG (RGBA cutout or grayscale mask), `Authorization: Bearer` | native |
| `download` | HTTPS download with size + SHA-256 verification (pulled in by `onnx`) | native |

The engine exposes it as `layer.removeBackgroundAI` and `select.subjectAI` (`crates/engine/src/ai_cmds.rs`,
features `ml-onnx` / `ml-remote`); the CLI enables both by default (feature `ai`), so the commands also
reach `run`, `batch`, droplets, `serve` and the MCP server.

## Models

Not vendored. Each registry entry pins a URL at a fixed revision, the byte size and SHA-256, and is downloaded
to `PHOTOCRAFT_MODELS_DIR` or the platform cache dir (`~/.cache/photocraft/models`,
`~/Library/Caches/photocraft/models`, `%LOCALAPPDATA%\photocraft\models`) on first use.
`PHOTOCRAFT_ML_OFFLINE=1` or `"allowDownload": false` disables downloading.

| id | weights | license | ONNX size | CPU time / peak RAM (8 vCPU, 1024² input) |
|---|---|---|---|---|
| `birefnet-lite` (default) | BiRefNet, Swin-T, general | MIT | 224 MB | ~5–7 s / ~6 GB |
| `birefnet` | BiRefNet, Swin-L, general | MIT | 973 MB | > 7 GB RAM; GPU recommended |
| `isnet` (low-memory fallback) | IS-Net general use (DIS) | Apache-2.0 | 179 MB | ~0.7 s / ~0.9 GB |
| `u2netp` | U²-Net small | Apache-2.0 | 4.6 MB | ~0.2 s / ~0.4 GB |
| `modnet` | MODNet (portraits only) | Apache-2.0 | 26 MB | ~0.2 s / ~0.3 GB |

`"model": "auto"` (the default) picks `birefnet-lite`, or `isnet` when the machine reports less free memory
than BiRefNet-lite needs. BRIA RMBG-1.4 / RMBG-2.0 are **not** listed: their weights are non-commercial
(bria-rmbg-1.4 / CC BY-NC 4.0).

## Settings, cache folder and the cloud key

Edit › Preferences › AI Models (`ai.*` preferences) sets the defaults the commands use: model
(`auto` / BiRefNet-lite / IS-Net), offline mode, cache folder (empty = the default above), cloud
provider on/off and cloud URL. Command params still override them. Agents can't change `ai.*`
preferences over MCP/`serve` (the cache folder is a write location and the URL receives images).

The cloud API key is **not** a preference. PhotoCraft has no OS keychain integration, so
[`secrets`](src/secrets.rs) keeps it in `<config dir>/photocraft/ai-cloud-key` (`$XDG_CONFIG_HOME` or
`~/.config` on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows;
`PHOTOCRAFT_CONFIG_DIR` overrides). The file is written atomically, owner-only (0600) on Unix.
`PHOTOCRAFT_ML_REMOTE_KEY` overrides it. The key is only sent to the configured URL, never to a
`url` passed as a command parameter.

ONNX Runtime itself is pinned: see [ORT_PINS.md](ORT_PINS.md). Licence texts for the runtime and
the MIT model weights are in [`licenses/`](licenses/).
