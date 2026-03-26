use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir).parent().unwrap().parent().unwrap();

    let target = std::env::var("TARGET").unwrap_or_default();
    let is_arm = target.starts_with("aarch64");

    // Try prebuilt dir first, then fall back to compiling with ea
    let prebuilt_dir = if is_arm {
        workspace_root.join("kernels/prebuilt/arm")
    } else {
        workspace_root.join("kernels/prebuilt/x86")
    };

    let kernel_src_dir = workspace_root.join("kernels/cougar");
    let out_dir = std::env::var("OUT_DIR").unwrap();

    println!("cargo:rerun-if-changed={}", kernel_src_dir.display());
    println!("cargo:rustc-link-lib=dl");

    let kernels = [
        "bitnet_activate", "bitnet_fused_attn", "bitnet_i2s", "bitnet_i8dot",
        "bitnet_quant", "bitnet_rmsnorm", "bitnet_vecadd",
        "q4k_quant", "q4k_dot", "q6k_dot", "rope",
    ];

    // Find ea compiler
    let ea_cmd = find_ea_compiler();

    let mut hasher = DefaultHasher::new();
    let mut code = String::new();
    let mut files_list = Vec::new();

    for name in &kernels {
        let so_name = format!("lib{}.so", name);
        let out_so = Path::new(&out_dir).join(&so_name);

        // Strategy: prebuilt → compile → skip
        let resolved = if prebuilt_dir.join(&so_name).exists() {
            let src = prebuilt_dir.join(&so_name);
            fs::copy(&src, &out_so).ok();
            Some(out_so.clone())
        } else if let Some(ref ea) = ea_cmd {
            // Try to compile from .ea source
            let ea_stem = if is_arm {
                let arm_name = format!("{}_arm", name);
                if kernel_src_dir.join(format!("{}.ea", arm_name)).exists() {
                    arm_name
                } else {
                    name.to_string()
                }
            } else {
                name.to_string()
            };
            let ea_file = kernel_src_dir.join(format!("{}.ea", ea_stem));
            if ea_file.exists() {
                let mut cmd = std::process::Command::new(ea);
                cmd.arg(ea_file.to_str().unwrap())
                    .arg("--lib")
                    .arg("--opt-level=3")
                    .arg("-o")
                    .arg(out_so.to_str().unwrap());
                if is_arm {
                    cmd.arg("--target=aarch64");
                }
                match cmd.status() {
                    Ok(s) if s.success() => Some(out_so.clone()),
                    _ => {
                        println!("cargo:warning=failed to compile {}", ea_stem);
                        None
                    }
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref path) = resolved {
            if let Ok(bytes) = fs::read(path) {
                bytes.hash(&mut hasher);
                let const_name = name.to_uppercase();
                code.push_str(&format!(
                    "pub const {}: &[u8] = include_bytes!(\"{}\");\n",
                    const_name,
                    path.display(),
                ));
                files_list.push((so_name.clone(), const_name));
            }
        } else {
            println!("cargo:warning=kernel {} not available — will load at runtime", name);
            let const_name = name.to_uppercase();
            code.push_str(&format!("pub const {}: &[u8] = &[];\n", const_name));
            files_list.push((so_name.clone(), const_name));
        }
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

    let out_path = Path::new(&out_dir).join("embedded_kernels.rs");
    fs::write(&out_path, code).unwrap();
}

fn find_ea_compiler() -> Option<String> {
    // Check PATH
    if std::process::Command::new("ea").arg("--version").output().is_ok() {
        return Some("ea".into());
    }
    // Check known location
    let known = "/root/dev/eacompute/target/release/ea";
    if Path::new(known).exists() {
        return Some(known.into());
    }
    println!("cargo:warning=ea compiler not found, using prebuilt kernels or empty stubs");
    None
}
