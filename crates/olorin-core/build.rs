use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();

    let target = std::env::var("TARGET").unwrap_or_default();
    let lib_dir = if target.starts_with("aarch64") {
        workspace_root.join("kernels/prebuilt/arm")
    } else {
        workspace_root.join("kernels/prebuilt/x86")
    };
    println!("cargo:rerun-if-changed={}", lib_dir.display());

    let kernels = [
        ("byte_classifier", "libbyte_classifier.so"),
        ("json_scanner", "libjson_scanner.so"),
        ("command_router", "libcommand_router.so"),
        ("leak_scanner", "libleak_scanner.so"),
        ("sanitizer", "libsanitizer.so"),
        ("fused_safety", "libfused_safety.so"),
        ("search", "libsearch.so"),
        ("search_avx512", "libsearch_avx512.so"),
    ];

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir).join("embedded_kernels.rs");

    let mut hasher = DefaultHasher::new();
    let mut code = String::new();

    for (name, file) in &kernels {
        let path = lib_dir.join(file);
        if path.exists() {
            let abs = fs::canonicalize(&path)
                .unwrap_or_else(|e| panic!("cannot resolve {file}: {e}"));
            let bytes = fs::read(&abs).unwrap_or_else(|e| panic!("cannot read {file}: {e}"));
            bytes.hash(&mut hasher);
            code.push_str(&format!(
                "pub const {}: &[u8] = include_bytes!(\"{}\");\n",
                name.to_uppercase(),
                abs.display(),
            ));
        } else {
            // Kernel not yet built — emit empty slice so crate compiles.
            // Runtime init() will fail until kernels are compiled.
            code.push_str(&format!(
                "pub const {}: &[u8] = &[];\n",
                name.to_uppercase(),
            ));
        }
    }

    let hash = format!("{:012x}", hasher.finish());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    code.push_str(&format!(
        "\npub const VERSION: &str = \"v{version}-{hash}\";\n"
    ));

    fs::write(&out_path, code).unwrap();
}
