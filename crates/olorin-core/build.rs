use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_ea() -> PathBuf {
    if let Ok(out) = Command::new("which").arg("ea").output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                return PathBuf::from(p);
            }
        }
    }
    panic!("ea compiler not found in PATH — build eacompute first");
}

fn compile_kernel(
    ea: &Path,
    ea_file: &Path,
    so_path: &Path,
    is_arm: bool,
) {
    let mut cmd = Command::new(ea);
    cmd.arg(ea_file)
        .arg("--lib")
        .arg("--opt-level=3")
        .arg("-o")
        .arg(so_path);
    if is_arm {
        cmd.arg("--target-triple=aarch64-unknown-linux-gnu");
        cmd.arg("--dotprod");
        cmd.env("CC", "aarch64-linux-gnu-gcc");
    }
    let output = cmd.output()
        .unwrap_or_else(|e| panic!("failed to run ea on {}: {e}", ea_file.display()));
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        panic!("ea compile failed for {}:\n{}", ea_file.display(), err);
    }
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let kernels_dir = workspace_root.join("kernels/olorin");
    let eachacha_dir = workspace_root.join("kernels/eachacha");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let target = std::env::var("TARGET").unwrap_or_default();
    let is_arm = target.starts_with("aarch64");

    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    println!("cargo:rerun-if-changed={}", eachacha_dir.display());

    let ea = find_ea();

    // Compile chacha20 for vault crypto.
    let chacha_ea = eachacha_dir.join("chacha20.ea");
    let chacha_so = out_dir.join("libchacha20.so");
    compile_kernel(&ea, &chacha_ea, &chacha_so, is_arm);
    let chacha_abs = fs::canonicalize(&chacha_so).unwrap();
    println!("cargo:rustc-env=CHACHA_LIB_PATH={}", chacha_abs.display());

    // Compile olorin kernels.
    let kernels = [
        "byte_classifier",
        "json_scanner",
        "command_router",
        "leak_scanner",
        "sanitizer",
        "fused_safety",
        "search",
        "search_avx512",
        "turbo_rotate",
        "jl_project",
    ];

    let mut hasher = DefaultHasher::new();
    let mut code = String::new();

    for name in &kernels {
        let ea_file = kernels_dir.join(format!("{name}.ea"));
        let so_path = out_dir.join(format!("lib{name}.so"));
        compile_kernel(&ea, &ea_file, &so_path, is_arm);

        let abs = fs::canonicalize(&so_path)
            .unwrap_or_else(|e| panic!("cannot resolve {name}: {e}"));
        let bytes = fs::read(&abs)
            .unwrap_or_else(|e| panic!("cannot read {name}: {e}"));
        bytes.hash(&mut hasher);
        code.push_str(&format!(
            "pub const {}: &[u8] = include_bytes!(\"{}\");\n",
            name.to_uppercase(),
            abs.display(),
        ));
    }

    let hash = format!("{:012x}", hasher.finish());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    code.push_str(&format!(
        "\npub const VERSION: &str = \"v{version}-{hash}\";\n"
    ));

    let out_path = out_dir.join("embedded_kernels.rs");
    fs::write(&out_path, code).unwrap();
}
