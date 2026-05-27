use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_ea() -> PathBuf {
    if let Ok(p) = std::env::var("EA") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return pb;
        }
    }
    let names: &[&str] = if cfg!(windows) { &["ea.exe", "ea"] } else { &["ea"] };
    let sep = if cfg!(windows) { ';' } else { ':' };
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(sep) {
            for name in names {
                let candidate = Path::new(dir).join(name);
                if candidate.is_file() {
                    return candidate;
                }
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
    let is_windows = target.contains("windows");
    let dynlib_filename = |stem: &str| -> String {
        if target.contains("windows") {
            format!("{stem}.dll")
        } else if target.contains("darwin") {
            format!("lib{stem}.dylib")
        } else {
            format!("lib{stem}.so")
        }
    };

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
        i8mm: bool,
        path: PathBuf,
    }

    let sources: Vec<KernelSrc> = entries
        .iter()
        .map(|e| {
            let fname = e.file_name().to_string_lossy().to_string();
            let stem = fname.strip_suffix(".ea").unwrap().to_string();
            let i8mm = stem.ends_with("_arm_i8mm");
            let arm_only = !i8mm && stem.ends_with("_arm");
            let content = fs::read_to_string(e.path()).unwrap_or_default();
            let has_cfg_x86 = content.contains("#[cfg(x86_64)]");
            let has_cfg_arm = content.contains("#[cfg(aarch64)]");
            let x86_only = stem.contains("_avx2") || stem.contains("_avx512")
                || (has_cfg_x86 && !has_cfg_arm);
            KernelSrc { stem, arm_only, x86_only, i8mm, path: e.path() }
        })
        .collect();

    // Filter by architecture
    let filtered: Vec<&KernelSrc> = sources
        .iter()
        .filter(|s| {
            if is_arm {
                if s.x86_only { return false; }
                if s.i8mm || s.arm_only {
                    return true;
                }
                // Skip generic if ARM variant exists
                let arm_variant = format!("{}_arm", s.stem);
                !sources.iter().any(|o| o.stem == arm_variant)
            } else {
                !s.arm_only && !s.i8mm
            }
        })
        .collect();

    let mut hasher = DefaultHasher::new();
    let mut compiled: Vec<(String, String, PathBuf)> = Vec::new();

    for src in &filtered {
        let out_stem = if src.i8mm {
            // foo_arm_i8mm -> foo_i8mm
            let base = src.stem.strip_suffix("_arm_i8mm").unwrap();
            format!("{base}_i8mm")
        } else {
            src.stem.strip_suffix("_arm").unwrap_or(&src.stem).to_string()
        };
        let so_path = out_dir.join(dynlib_filename(&out_stem));

        let mut cmd = Command::new(&ea);
        cmd.arg(&src.path)
            .arg("--lib")
            .arg("--opt-level=3")
            .arg("-o")
            .arg(&so_path);
        if is_arm {
            cmd.arg("--target-triple=aarch64-unknown-linux-gnu");
            cmd.arg("--target=cortex-a76");
            cmd.arg("--dotprod");
            if src.i8mm {
                cmd.arg("--i8mm");
            }
            cmd.env("CC", "aarch64-linux-gnu-gcc");
        } else if is_windows {
            // Tell ea to emit COFF + link via mingw-w64 (when host is Linux).
            // x86-64-v3 = SSSE3+SSE4.2+AVX2+BMI+FMA+F16C — matches what
            // Olorin's _avx2 kernels assume; without an explicit --target,
            // ea's cross-compile path falls back to "generic" (SSE2 only)
            // and SSSE3 intrinsics like pmaddubsw fail to select.
            cmd.arg(format!("--target-triple={target}"));
            cmd.arg("--target=x86-64-v3");
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

        compiled.push((out_stem.to_string(), dynlib_filename(&out_stem), abs));
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

    // ── Rune auto-discovery ───────────────────────────────────────────
    // Scans src/runes/*.rs (excluding mod.rs, common.rs, output.rs). For each file,
    // verify it exports `pub const RUNE` (grep-parse, no syntactic Rust
    // analysis); emit runes_registry.rs into OUT_DIR.
    let runes_dir = Path::new(&manifest_dir).join("src/runes");
    let mut rune_mods: Vec<String> = Vec::new();
    if runes_dir.is_dir() {
        for entry in fs::read_dir(&runes_dir).unwrap() {
            let entry = entry.unwrap();
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".rs") { continue; }
            if name == "mod.rs" || name == "common.rs" || name == "output.rs"
                || name == "eajson_aggregate.rs" || name == "narration.rs" { continue; }
            let stem = name.strip_suffix(".rs").unwrap();
            let contents = fs::read_to_string(entry.path()).unwrap_or_default();
            if !contents.contains("pub const RUNE") {
                panic!("rune file {name} is missing `pub const RUNE: <Type> = ...`");
            }
            rune_mods.push(stem.to_string());
        }
    }
    rune_mods.sort();

    let mut out = String::new();
    for m in &rune_mods {
        // Use #[path] with the manifest-relative source path so that
        // `pub mod` inside the include!()'d registry file resolves correctly
        // (Rust otherwise looks in OUT_DIR, not src/runes/).
        let src_path = Path::new(&manifest_dir)
            .join("src/runes")
            .join(format!("{m}.rs"));
        out.push_str(&format!(
            "#[path = \"{}\"]\npub mod {m};\n",
            src_path.display()
        ));
    }
    out.push_str("pub const RUNES: &[&(dyn crate::runes::Rune + Sync)] = &[\n");
    for m in &rune_mods {
        out.push_str(&format!("    &{m}::RUNE,\n"));
    }
    out.push_str("];\n");

    let out_file = out_dir.join("runes_registry.rs");
    fs::write(&out_file, out)
        .unwrap_or_else(|e| panic!("cannot write runes_registry.rs: {e}"));
    println!("cargo:rerun-if-changed=src/runes");
}
