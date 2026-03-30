use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().chain(c).collect(),
            }
        })
        .collect()
}

fn to_upper_snake(s: &str) -> String {
    s.to_uppercase()
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_root = Path::new(&manifest_dir).join("kernels");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target = std::env::var("TARGET").unwrap_or_default();
    let is_arm = target.starts_with("aarch64");

    println!("cargo:rerun-if-changed=kernels");

    let ea = find_ea();

    // Discover all .ea files
    let mut entries: Vec<_> = fs::read_dir(&kernels_root)
        .unwrap_or_else(|e| panic!("cannot read kernels/: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "ea").unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    struct KernelSrc {
        stem: String,
        arm_only: bool,
        x86_only: bool,
        path: PathBuf,
    }

    let sources: Vec<KernelSrc> = entries
        .iter()
        .map(|e| {
            let fname = e.file_name().to_string_lossy().to_string();
            let stem = fname.strip_suffix(".ea").unwrap().to_string();
            let arm_only = stem.ends_with("_arm");
            let x86_only = stem.contains("_avx2") || stem.contains("_avx512");
            KernelSrc { stem, arm_only, x86_only, path: e.path() }
        })
        .collect();

    // Filter by architecture
    let filtered: Vec<&KernelSrc> = sources
        .iter()
        .filter(|s| {
            if is_arm {
                if s.x86_only { return false; }
                if !s.arm_only {
                    let arm_variant = format!("{}_arm", s.stem);
                    !sources.iter().any(|o| o.stem == arm_variant)
                } else {
                    true
                }
            } else {
                !s.arm_only
            }
        })
        .collect();

    let mut hasher = DefaultHasher::new();
    let mut compiled: Vec<(String, String, PathBuf)> = Vec::new();

    for src in &filtered {
        let out_stem = src.stem.strip_suffix("_arm").unwrap_or(&src.stem);
        let so_path = out_dir.join(format!("lib{out_stem}.so"));

        let mut cmd = Command::new(&ea);
        cmd.arg(&src.path)
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
            .unwrap_or_else(|e| panic!("failed to run ea on {}: {e}", src.path.display()));
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("ea compile failed for {}:\n{}", src.path.display(), stderr);
        }

        let abs = fs::canonicalize(&so_path)
            .unwrap_or_else(|e| panic!("cannot resolve {}: {e}", so_path.display()));
        let bytes = fs::read(&abs)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", abs.display()));
        bytes.hash(&mut hasher);

        compiled.push((out_stem.to_string(), format!("lib{out_stem}.so"), abs));
    }

    // Deduplicate by output stem
    let mut seen = std::collections::HashSet::new();
    compiled.retain(|(stem, _, _)| seen.insert(stem.clone()));

    let content_hash = format!("{:012x}", hasher.finish());
    let version = std::env::var("CARGO_PKG_VERSION").unwrap();

    let mut code = String::with_capacity(8192);

    // KernelId enum
    code.push_str("#[repr(u8)]\n#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum KernelId {\n");
    for (i, (stem, _, _)) in compiled.iter().enumerate() {
        code.push_str(&format!("    {} = {i},\n", to_pascal_case(stem)));
    }
    code.push_str("}\n\n");
    code.push_str(&format!("pub const KERNEL_COUNT: usize = {};\n\n", compiled.len()));

    // Version
    code.push_str(&format!(
        "pub const VERSION: &str = \"v{version}-{content_hash}\";\n\n"
    ));

    // Byte constants
    for (stem, _, abs_path) in &compiled {
        code.push_str(&format!(
            "pub const {}: &[u8] = include_bytes!(\"{}\");\n",
            to_upper_snake(stem),
            abs_path.display(),
        ));
    }

    // FILES array
    code.push_str("\npub const FILES: &[(KernelId, &str, &[u8])] = &[\n");
    for (stem, so_name, _) in &compiled {
        code.push_str(&format!(
            "    (KernelId::{}, \"{so_name}\", {}),\n",
            to_pascal_case(stem),
            to_upper_snake(stem),
        ));
    }
    code.push_str("];\n");

    let out_path = out_dir.join("embedded_kernels.rs");
    fs::write(&out_path, &code)
        .unwrap_or_else(|e| panic!("cannot write embedded_kernels.rs: {e}"));
}
