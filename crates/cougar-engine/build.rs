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
    let kernel_src_dir = workspace_root.join("kernels/cougar");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let target = std::env::var("TARGET").unwrap_or_default();
    let is_arm = target.starts_with("aarch64");

    println!("cargo:rerun-if-changed={}", kernel_src_dir.display());
    println!("cargo:rustc-link-lib=dl");

    let ea = find_ea();

    let kernels = [
        "bitnet_activate", "bitnet_fused_attn", "bitnet_i2s", "bitnet_i8dot",
        "bitnet_quant", "bitnet_rmsnorm", "bitnet_vecadd",
        "q4k_quant", "q4k_dot", "q6k_dot", "rope",
    ];

    let mut hasher = DefaultHasher::new();
    let mut code = String::new();
    let mut files_list = Vec::new();

    for name in &kernels {
        let ea_stem = if is_arm {
            let arm_name = format!("{name}_arm");
            if kernel_src_dir.join(format!("{arm_name}.ea")).exists() {
                arm_name
            } else {
                name.to_string()
            }
        } else {
            name.to_string()
        };

        let ea_file = kernel_src_dir.join(format!("{ea_stem}.ea"));
        let so_name = format!("lib{name}.so");
        let so_path = out_dir.join(&so_name);
        compile_kernel(&ea, &ea_file, &so_path, is_arm);

        let bytes = fs::read(&so_path)
            .unwrap_or_else(|e| panic!("cannot read {so_name}: {e}"));
        bytes.hash(&mut hasher);
        let const_name = name.to_uppercase();
        code.push_str(&format!(
            "pub const {const_name}: &[u8] = include_bytes!(\"{}\");\n",
            so_path.display(),
        ));
        files_list.push((so_name, const_name));
    }

    let hash = format!("{:012x}", hasher.finish());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();
    code.push_str(&format!(
        "\npub const VERSION: &str = \"v{version}-{hash}\";\n"
    ));

    code.push_str("\npub const FILES: &[(&str, &[u8])] = &[\n");
    for (file, const_name) in &files_list {
        code.push_str(&format!("    (\"{file}\", {const_name}),\n"));
    }
    code.push_str("];\n");

    let out_path = out_dir.join("embedded_kernels.rs");
    fs::write(&out_path, code).unwrap();
}
