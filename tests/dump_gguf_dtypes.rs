//! Dump per-tensor dtype info for a GGUF file.
//! Run: cargo test --release --test dump_gguf_dtypes -- --ignored --nocapture

use olorin::inference::gguf::GgufFile;
use std::path::Path;

#[test]
#[ignore]
fn dump_adaptive_vs_original() {
    let home = std::env::var("HOME").unwrap();
    let originals = [
        ("ORIGINAL", Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")),
        ("ADAPTIVE", Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M-adaptive.gguf")),
    ];

    for (label, path) in &originals {
        if !path.exists() {
            eprintln!("SKIP {label}: no file at {}", path.display());
            continue;
        }
        let gguf = GgufFile::open(path).expect("open gguf");
        let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        let mut sample: Vec<(String, u32)> = Vec::new();
        for (name, &idx) in gguf.tensor_map.iter() {
            let tensor = &gguf.tensors[idx];
            *counts.entry(tensor.dtype).or_insert(0) += 1;
            if name.starts_with("blk.0.") || name.starts_with("blk.34.") {
                sample.push((name.clone(), tensor.dtype));
            }
        }
        let dtype_name = |t: u32| -> &'static str {
            match t {
                0 => "F32", 1 => "F16", 30 => "BF16",
                12 => "Q4K", 13 => "Q5K", 14 => "Q6K",
                _ => "OTHER",
            }
        };
        eprintln!("\n=== {label} ({}) ===", path.display());
        eprintln!("Total tensors: {}", gguf.tensors.len());
        for (t, n) in &counts {
            eprintln!("  {:<6} = {:>5}  count: {}", dtype_name(*t), t, n);
        }
        eprintln!("\nSample (L0 + L34):");
        sample.sort();
        for (name, t) in &sample {
            eprintln!("  {:<44} {}", name, dtype_name(*t));
        }
    }
}
