// Unified kernel build.rs — compiles and embeds all Ea kernels.
//
// Kernel directories: cougar, olorin, eachacha, eakv, eastat
// Detects architecture, compiles .ea → .so. No fallbacks.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

struct KernelSource {
    dir: &'static str,
    stem: &'static str,
    arm_only: bool,
}

fn discover_kernels(kernels_root: &Path) -> Vec<KernelSource> {
    let dirs = ["cougar", "olorin", "eachacha", "eakv", "eastat"];
    let mut sources = Vec::new();
    for dir in &dirs {
        let dir_path = kernels_root.join(dir);
        if !dir_path.is_dir() {
            continue;
        }
        let mut entries: Vec<_> = fs::read_dir(&dir_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir_path.display()))
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "ea")
                    .unwrap_or(false)
            })
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let fname = entry.file_name();
            let fname_str = fname.to_string_lossy();
            let stem = fname_str.strip_suffix(".ea").unwrap().to_string();
            let arm_only = stem.ends_with("_arm");
            sources.push(KernelSource {
                dir: match *dir {
                    "cougar" => "cougar",
                    "olorin" => "olorin",
                    "eachacha" => "eachacha",
                    "eakv" => "eakv",
                    "eastat" => "eastat",
                    _ => unreachable!(),
                },
                stem: Box::leak(stem.into_boxed_str()),
                arm_only,
            });
        }
    }
    sources
}

fn find_ea() -> PathBuf {
    if let Ok(output) = Command::new("which").arg("ea").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return PathBuf::from(path);
            }
        }
    }
    panic!("ea compiler not found in PATH — build eacompute first");
}

fn compile_kernel(
    ea: &Path,
    src: &Path,
    output_stem: &str,
    out_dir: &Path,
    is_arm: bool,
) -> PathBuf {
    let so_path = out_dir.join(format!("lib{output_stem}.so"));
    let mut cmd = Command::new(ea);
    cmd.arg(src)
        .arg("--lib")
        .arg("--opt-level=3")
        .arg("-o")
        .arg(&so_path);
    if is_arm {
        cmd.arg("--target-triple=aarch64-unknown-linux-gnu");
        cmd.arg("--dotprod");
        cmd.env("CC", "aarch64-linux-gnu-gcc");
    }

    let output = cmd.output()
        .unwrap_or_else(|e| panic!("failed to run ea on {}: {e}", src.display()));
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("ea compile failed for {}:\n{}", src.display(), stderr);
    }
    so_path
}

fn output_stem(stem: &str) -> &str {
    stem.strip_suffix("_arm").unwrap_or(stem)
}

fn to_upper_snake(s: &str) -> String {
    s.to_uppercase()
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let workspace_root = Path::new(&manifest_dir).parent().unwrap();
    let kernels_root = workspace_root.join("kernels");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_arm = target.starts_with("aarch64");

    for dir in &["cougar", "olorin", "eachacha", "eakv", "eastat"] {
        println!(
            "cargo:rerun-if-changed={}",
            kernels_root.join(dir).display()
        );
    }

    let ea = find_ea();
    let sources = discover_kernels(&kernels_root);

    // Filter by architecture
    let filtered: Vec<&KernelSource> = sources
        .iter()
        .filter(|s| {
            if is_arm {
                if !s.arm_only {
                    let arm_variant = format!("{}_arm", s.stem);
                    let has_arm = sources.iter().any(|o| o.stem == arm_variant && o.dir == s.dir);
                    !has_arm
                } else {
                    true
                }
            } else {
                !s.arm_only
            }
        })
        .collect();

    let mut hasher = DefaultHasher::new();
    let mut consts: Vec<(String, String, PathBuf)> = Vec::new();

    for src in &filtered {
        let out_stem = output_stem(src.stem);
        let so_name = format!("lib{out_stem}.so");
        let ea_file = kernels_root
            .join(src.dir)
            .join(format!("{}.ea", src.stem));

        let so_path = compile_kernel(&ea, &ea_file, out_stem, &out_dir, is_arm);

        let abs = fs::canonicalize(&so_path)
            .unwrap_or_else(|e| panic!("cannot resolve {}: {e}", so_path.display()));
        let bytes = fs::read(&abs)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", abs.display()));
        bytes.hash(&mut hasher);

        let const_name = to_upper_snake(out_stem);
        consts.push((const_name, so_name, abs));
    }

    // Deduplicate: if the same output stem was produced by multiple dirs, keep first
    let mut seen = std::collections::HashSet::new();
    consts.retain(|(name, _, _)| seen.insert(name.clone()));

    // Generate embedded_kernels.rs
    let content_hash = format!("{:012x}", hasher.finish());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();

    let mut code = String::with_capacity(4096);
    code.push_str(&format!(
        "pub const VERSION: &str = \"v{version}-{content_hash}\";\n\n"
    ));

    for (const_name, _so_name, abs_path) in &consts {
        code.push_str(&format!(
            "pub const {const_name}: &[u8] = include_bytes!(\"{}\");\n",
            abs_path.display(),
        ));
    }

    code.push_str("\npub const FILES: &[(&str, &[u8])] = &[\n");
    for (const_name, so_name, _) in &consts {
        code.push_str(&format!("    (\"{so_name}\", {const_name}),\n"));
    }
    code.push_str("];\n");

    let out_path = out_dir.join("embedded_kernels.rs");
    fs::write(&out_path, &code)
        .unwrap_or_else(|e| panic!("cannot write embedded_kernels.rs: {e}"));
}
