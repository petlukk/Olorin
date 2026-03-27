use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_ea() -> Option<PathBuf> {
    if let Ok(out) = Command::new("which").arg("ea").output() {
        if out.status.success() {
            let p = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !p.is_empty() {
                return Some(PathBuf::from(p));
            }
        }
    }
    let known = PathBuf::from("/root/dev/eacompute/target/release/ea");
    if known.is_file() { return Some(known); }
    None
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();
    let kernels_dir = workspace_root.join("kernels/olorin");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let target = std::env::var("TARGET").unwrap_or_default();
    let is_arm = target.starts_with("aarch64");

    println!("cargo:rerun-if-changed={}", kernels_dir.display());

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

    let ea = find_ea();
    let mut hasher = DefaultHasher::new();
    let mut code = String::new();

    for name in &kernels {
        let so_path = out_dir.join(format!("lib{name}.so"));
        let mut compiled = false;

        if let Some(ref ea_path) = ea {
            let ea_file = kernels_dir.join(format!("{name}.ea"));
            if ea_file.exists() {
                let mut cmd = Command::new(ea_path);
                cmd.arg(&ea_file)
                    .arg("--lib")
                    .arg("--opt-level=3")
                    .arg("-o")
                    .arg(&so_path);
                if is_arm {
                    cmd.arg("--target-triple=aarch64-unknown-linux-gnu");
                    cmd.arg("--dotprod");
                    cmd.env("CC", "aarch64-linux-gnu-gcc");
                }
                if let Ok(out) = cmd.output() {
                    if out.status.success() {
                        compiled = true;
                    } else {
                        let err = String::from_utf8_lossy(&out.stderr);
                        println!(
                            "cargo:warning=failed to compile {name}: {}",
                            err.lines().next().unwrap_or("unknown")
                        );
                    }
                }
            }
        }

        if compiled {
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
        } else {
            println!("cargo:warning=kernel {name} not available — will fail at runtime");
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

    let out_path = out_dir.join("embedded_kernels.rs");
    fs::write(&out_path, code).unwrap();
}
