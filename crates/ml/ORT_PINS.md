# ONNX Runtime pin

`photocraft-ml` (feature `onnx`) links **ONNX Runtime 1.28.0** statically through
[`ort`](https://github.com/pykeio/ort) / `ort-sys` **=2.0.0-rc.13** (both pinned with `=` in
`crates/ml/Cargo.toml`; Cargo.lock checksums below). At build time `ort-sys` downloads pyke's
prebuilt archive for the target into `~/.cache/ort.pyke.io/dfbin/<target>/<sha256>/` and rejects it
unless its SHA-256 matches the table compiled into that exact `ort-sys` release
(`build/download/dist.tsv`). Pinning the crate therefore pins the runtime bytes.

| Crate | Version | crates.io checksum (Cargo.lock) |
|---|---|---|
| ort | 2.0.0-rc.13 | `4336a1e2b38848325241c72889086886004e589b7c74f335e60a8e8db5138a0b` |
| ort-sys | 2.0.0-rc.13 | `cf211e3776eea6aec988552fa118dd746d70e1b1e5e244058d1c98015f3e5872` |

Runtime archives used by PhotoCraft's targets (from `ort-sys` 2.0.0-rc.13 `dist.tsv`):

| Target | ort features | URL | SHA-256 |
|---|---|---|---|
| x86_64-unknown-linux-gnu | (none) | https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/x86_64-unknown-linux-gnu.tar.lzma2 | `e454f710f8a49f53aa5b4ff51e3454ae1835777e431c6c35c5255ce6f205fd68` |
| aarch64-unknown-linux-gnu | (none) | https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/aarch64-unknown-linux-gnu.tar.lzma2 | `06a050ab9137ccb32421d0cb49e9ccf72d9e18ab0aeb8f8d038d1b5cc844b35a` |
| x86_64-pc-windows-msvc | directml | https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/x86_64-pc-windows-msvc+directml.tar.lzma2 | `f7c654b3729cb9e5ad2a36a0c38e5b48e63bf4eed22968931aed33a0ad0b527d` |
| aarch64-pc-windows-msvc | directml | https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/aarch64-pc-windows-msvc+directml.tar.lzma2 | `56df72970baeea08295db33f15f985cd74f4d4cd00e0099c44aba1504c5d140e` |
| aarch64-apple-darwin | coreml | https://cdn.pyke.io/0/pyke:ort-rs/ms@1.28.0/aarch64-apple-darwin+coreml.tar.lzma2 | `6934874e2e953576d9c1db47ff1af39c62c4f4220dbe6f988e131f72879674c7` |
| x86_64-apple-darwin | — | no prebuilt archive: built without the local backend (cloud provider only) | — |

Only the Linux x86_64 archive has been built and run here (the hash above is the directory
name in the local cache). Windows and macOS rows are taken from the pinned table and have not
been built yet.

## Air-gapped / mirrored builds

To avoid the CDN at build time (release CI, reproducible builds), mirror the archive above,
verify its SHA-256, extract it, and build with `ORT_LIB_PATH=<extracted dir>/lib`. The same
`ort-sys` then links that copy instead of downloading. `ORT_SKIP_DOWNLOAD=1` (or
`CARGO_NET_OFFLINE`) makes a build fail rather than reach the network.

## Bumping

1. Change both `=` versions in `crates/ml/Cargo.toml` and run `cargo update -p ort -p ort-sys`.
2. Copy the new rows from `ort-sys`'s `build/download/dist.tsv` into this file.
3. Update `PINNED_ORT_SYS` (version and checksum) in `crates/ml/src/lib.rs` tests, and the
   ONNX Runtime row in `NOTICE` / `ATTRIBUTION.md`.
