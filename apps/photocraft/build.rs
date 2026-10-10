//! Windows only: embed the app icon and version info (VERSIONINFO) into `photocraft.exe`.
//!
//! On every other target this does nothing. A missing resource compiler is a warning, so a
//! cross-compile from macOS or Linux still links, unless `PHOTOCRAFT_REQUIRE_WINRES=1` (set by the
//! release workflow) turns it into an error.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../assets/app-icon/photocraft.ico");
    println!("cargo:rerun-if-env-changed=PHOTOCRAFT_REQUIRE_WINRES");
    println!("cargo:rerun-if-changed=../../FORKITUP_VERSION");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    // Fork It Up's own version (FORKITUP_VERSION), not the upstream workspace version.
    let version = std::fs::read_to_string("../../FORKITUP_VERSION").unwrap_or_default().trim().to_owned();
    let mut res = winresource::WindowsResource::new();
    if !version.is_empty() {
        let mut n = version.split('-').next().unwrap_or("").split('.').map(|p| p.parse::<u64>().unwrap_or(0));
        let packed = (n.next().unwrap_or(0) << 48) | (n.next().unwrap_or(0) << 32) | (n.next().unwrap_or(0) << 16);
        res.set("ProductVersion", &version)
            .set("FileVersion", &version)
            .set_version_info(winresource::VersionInfo::PRODUCTVERSION, packed)
            .set_version_info(winresource::VersionInfo::FILEVERSION, packed);
    }
    res.set_icon("../../assets/app-icon/photocraft.ico")
        .set("ProductName", "Fork It Up")
        .set("FileDescription", "Fork It Up image editor")
        .set("CompanyName", "Fork It Up contributors")
        .set("LegalCopyright", "Copyright (c) the Fork It Up and PhotoCraft authors. MIT OR Apache-2.0.")
        .set("OriginalFilename", "photocraft.exe")
        .set("InternalName", "photocraft");
    if let Err(e) = res.compile() {
        if std::env::var_os("PHOTOCRAFT_REQUIRE_WINRES").is_some() {
            panic!("embedding Windows resources failed: {e}");
        }
        println!("cargo:warning=photocraft.exe built without icon/version resources: {e}");
    }
}
