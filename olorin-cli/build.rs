// Unified kernel build.rs — compiles and embeds all 49 Ea kernels.
//
// Kernel directories: cougar, olorin, eachacha, eakv, eastat
// Detects architecture, compiles .ea → .so, falls back to prebuilt .so files.

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// A kernel source: (subdirectory, filename without .ea)
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

fn find_ea_compiler() -> Option<PathBuf> {
    // Check PATH first
    if let Ok(output) = Command::new("which").arg("ea").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }
    // Check well-known location
    let known = PathBuf::from("/root/dev/eacompute/target/release/ea");
    if known.is_file() {
        return Some(known);
    }
    None
}

/// Compile a single .ea file, returning the path to the .so on success.
fn compile_kernel(
    ea: &Path,
    src: &Path,
    output_stem: &str,
    out_dir: &Path,
    is_arm: bool,
) -> Option<PathBuf> {
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

    match cmd.output() {
        Ok(output) if output.status.success() => Some(so_path),
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!(
                "cargo:warning=ea compile failed for {}: {}",
                src.display(),
                stderr.lines().next().unwrap_or("unknown error")
            );
            None
        }
        Err(e) => {
            println!(
                "cargo:warning=ea compile error for {}: {e}",
                src.display()
            );
            None
        }
    }
}

/// Resolve the output stem: strip _arm suffix for ARM builds so both
/// architectures produce the same library name.
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

    let prebuilt_dir = if is_arm {
        kernels_root.join("prebuilt/arm")
    } else {
        kernels_root.join("prebuilt/x86")
    };

    // Rerun triggers
    for dir in &["cougar", "olorin", "eachacha", "eakv", "eastat"] {
        println!(
            "cargo:rerun-if-changed={}",
            kernels_root.join(dir).display()
        );
    }
    println!(
        "cargo:rerun-if-changed={}",
        prebuilt_dir.display()
    );

    let ea_compiler = find_ea_compiler();
    if let Some(ref ea) = ea_compiler {
        println!("cargo:warning=ea compiler found: {}", ea.display());
    } else {
        println!("cargo:warning=ea compiler not found, using prebuilt .so files");
    }

    let sources = discover_kernels(&kernels_root);

    // Filter by architecture
    let filtered: Vec<&KernelSource> = sources
        .iter()
        .filter(|s| {
            if is_arm {
                // On ARM: skip x86-only files if an _arm variant exists
                if !s.arm_only {
                    let arm_variant = format!("{}_arm", s.stem);
                    let has_arm = sources.iter().any(|o| o.stem == arm_variant && o.dir == s.dir);
                    !has_arm // skip x86 file only if ARM variant exists
                } else {
                    true
                }
            } else {
                // On x86: skip all _arm files
                !s.arm_only
            }
        })
        .collect();

    let mut hasher = DefaultHasher::new();
    let mut consts: Vec<(String, String, PathBuf)> = Vec::new(); // (CONST_NAME, filename, abs_path)
    let mut failed: Vec<String> = Vec::new();

    for src in &filtered {
        let out_stem = output_stem(src.stem);
        let so_name = format!("lib{out_stem}.so");
        let ea_file = kernels_root
            .join(src.dir)
            .join(format!("{}.ea", src.stem));

        // Try compile
        let compiled = ea_compiler
            .as_ref()
            .and_then(|ea| compile_kernel(ea, &ea_file, out_stem, &out_dir, is_arm));

        let so_path = if let Some(p) = compiled {
            p
        } else {
            // Fallback to prebuilt
            let prebuilt = prebuilt_dir.join(&so_name);
            if prebuilt.exists() {
                let dest = out_dir.join(&so_name);
                fs::copy(&prebuilt, &dest).unwrap_or_else(|e| {
                    panic!("cannot copy prebuilt {}: {e}", prebuilt.display())
                });
                dest
            } else {
                println!("cargo:warning=no prebuilt fallback for {so_name}, emitting empty");
                failed.push(out_stem.to_string());
                continue;
            }
        };

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

    // Empty slices for kernels that failed
    for stem in &failed {
        let const_name = to_upper_snake(stem);
        if !seen.contains(&const_name) {
            code.push_str(&format!("pub const {const_name}: &[u8] = &[];\n"));
            seen.insert(const_name.clone());
        }
    }

    // Manifest: (filename, bytes) pairs for runtime extraction
    code.push_str("\npub const FILES: &[(&str, &[u8])] = &[\n");
    for (const_name, so_name, _) in &consts {
        code.push_str(&format!("    (\"{so_name}\", {const_name}),\n"));
    }
    code.push_str("];\n");

    let out_path = out_dir.join("embedded_kernels.rs");
    fs::write(&out_path, &code)
        .unwrap_or_else(|e| panic!("cannot write embedded_kernels.rs: {e}"));

    println!(
        "cargo:warning=unified build: {} kernels embedded, {} failed",
        consts.len(),
        failed.len()
    );
}
