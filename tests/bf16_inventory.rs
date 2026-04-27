//! BF16 tensor inventory for the loaded gguf — what's BF16, where, and how big.
//!
//! Drives the BF16 PLE → INT8 scoping decision: lists every BF16 tensor
//! with its shape and byte size, groups by role (norms / PLE table /
//! PLE projector / other), and reports total BF16 footprint plus the
//! INT8 with-per-channel-scale conversion savings estimate.
//!
//! Run: cargo test --release --test bf16_inventory -- --ignored --nocapture
//!
//! Requires: ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf

use olorin::inference::gguf::GgufFile;
use std::path::Path;

const BF16: u32 = 30;

fn dtype_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 30 => "BF16",
        12 => "Q4K", 13 => "Q5K", 14 => "Q6K", 8 => "Q8K",
        2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1",
        _ => "OTHER",
    }
}

fn elem_bytes_estimate(dtype: u32, n_elem: u64) -> u64 {
    match dtype {
        0 => n_elem * 4,         // F32
        1 | 30 => n_elem * 2,    // F16, BF16
        12 => ((n_elem + 255) / 256) * 144,  // Q4K
        13 => ((n_elem + 255) / 256) * 176,  // Q5K (32+128+16)
        14 => ((n_elem + 255) / 256) * 210,  // Q6K
        _ => n_elem * 2,         // rough fallback
    }
}

#[test]
#[ignore]
fn dump_all_dtypes() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }
    let gguf = GgufFile::open(&path).expect("open gguf");

    let mut by_dtype: std::collections::BTreeMap<u32, (u32, u64)> = Default::default();
    let mut all_total: u64 = 0;
    for &idx in gguf.tensor_map.values() {
        let t = &gguf.tensors[idx];
        let n_elem: u64 = t.dims.iter().product();
        let bytes = elem_bytes_estimate(t.dtype, n_elem);
        let entry = by_dtype.entry(t.dtype).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += bytes;
        all_total += bytes;
    }
    eprintln!("\n=== All dtypes (gemma-4-e2b-it-Q4_K_M.gguf) ===\n");
    eprintln!("{:<8}  {:>5}  {:>12}  {:>10}", "dtype", "count", "MB", "% total");
    eprintln!("{:-<48}", "");
    let mut entries: Vec<_> = by_dtype.iter().collect();
    entries.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (dt, (count, bytes)) in entries {
        let mb = *bytes as f64 / 1_048_576.0;
        let pct = if all_total > 0 { 100.0 * *bytes as f64 / all_total as f64 } else { 0.0 };
        eprintln!("{:<8}  {:>5}  {:>12.2}  {:>9.2}%", dtype_name(*dt), count, mb, pct);
    }
    eprintln!("\nGrand total: {:.2} MB", all_total as f64 / 1_048_576.0);

    // Show top 25 tensors across ALL dtypes by byte size
    eprintln!("\n=== Top 25 tensors across ALL dtypes by size ===\n");
    let mut all_tensors: Vec<(&String, &olorin::inference::gguf::TensorInfo, u64)> = Vec::new();
    for (name, &idx) in gguf.tensor_map.iter() {
        let t = &gguf.tensors[idx];
        let n_elem: u64 = t.dims.iter().product();
        let bytes = elem_bytes_estimate(t.dtype, n_elem);
        all_tensors.push((name, t, bytes));
    }
    all_tensors.sort_by(|a, b| b.2.cmp(&a.2));
    eprintln!("{:<48}  {:<6}  {:>20}  {:>10}  {:>8}", "name", "dtype", "shape", "MB", "% total");
    eprintln!("{:-<102}", "");
    for (name, t, bytes) in all_tensors.iter().take(25) {
        let shape_s = format!("[{}]",
            t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
        let pct = 100.0 * *bytes as f64 / all_total as f64;
        eprintln!("{:<48}  {:<6}  {:>20}  {:>10.2}  {:>7.2}%",
            name, dtype_name(t.dtype), shape_s, *bytes as f64 / 1_048_576.0, pct);
    }

    // Show top 10 individual non-quant tensors (F32 + BF16 + F16) with sizes
    eprintln!("\n=== Top 15 non-quant tensors (F32/F16/BF16) ===\n");
    let mut nonquant: Vec<(&String, &olorin::inference::gguf::TensorInfo, u64)> = Vec::new();
    for (name, &idx) in gguf.tensor_map.iter() {
        let t = &gguf.tensors[idx];
        if matches!(t.dtype, 0 | 1 | 30) {
            let n_elem: u64 = t.dims.iter().product();
            nonquant.push((name, t, n_elem * if t.dtype == 0 { 4 } else { 2 }));
        }
    }
    nonquant.sort_by(|a, b| b.2.cmp(&a.2));
    eprintln!("{:<48}  {:<6}  {:>20}  {:>10}", "name", "dtype", "shape", "MB");
    eprintln!("{:-<92}", "");
    for (name, t, bytes) in nonquant.iter().take(15) {
        let shape_s = format!("[{}]",
            t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(","));
        eprintln!("{:<48}  {:<6}  {:>20}  {:>10.2}",
            name, dtype_name(t.dtype), shape_s, *bytes as f64 / 1_048_576.0);
    }
}

#[test]
#[ignore]
fn dump_bf16_tensors() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }

    let gguf = GgufFile::open(&path).expect("open gguf");

    let mut bf16: Vec<(&String, &olorin::inference::gguf::TensorInfo)> = gguf
        .tensor_map
        .iter()
        .filter_map(|(name, &idx)| {
            let t = &gguf.tensors[idx];
            if t.dtype == BF16 { Some((name, t)) } else { None }
        })
        .collect();
    bf16.sort_by(|a, b| {
        let sa: u64 = a.1.dims.iter().product();
        let sb: u64 = b.1.dims.iter().product();
        sb.cmp(&sa)
    });

    let mut total_bytes: u64 = 0;
    let mut by_role: std::collections::BTreeMap<&'static str, (u32, u64)> = Default::default();

    eprintln!("\n=== BF16 tensors in {} ===\n", path.display());
    eprintln!(
        "{:<48}  {:>20}  {:>10}",
        "name", "shape", "MB"
    );
    eprintln!("{:-<82}", "");

    for (name, t) in &bf16 {
        let n_elem: u64 = t.dims.iter().product();
        let bytes = n_elem * 2;
        total_bytes += bytes;

        let role = classify_role(name);
        let entry = by_role.entry(role).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += bytes;

        let shape_s = format!(
            "[{}]",
            t.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",")
        );
        eprintln!(
            "{:<48}  {:>20}  {:>10.2}",
            name,
            shape_s,
            bytes as f64 / 1_048_576.0
        );
    }

    eprintln!("\n=== Aggregate by role ===\n");
    eprintln!("{:<24}  {:>5}  {:>10}  {:>10}", "role", "count", "MB", "% of BF16");
    eprintln!("{:-<60}", "");
    let mut roles: Vec<_> = by_role.iter().collect();
    roles.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (role, (count, bytes)) in &roles {
        let mb = *bytes as f64 / 1_048_576.0;
        let pct = if total_bytes > 0 { 100.0 * *bytes as f64 / total_bytes as f64 } else { 0.0 };
        eprintln!("{:<24}  {:>5}  {:>10.2}  {:>9.1}%", role, count, mb, pct);
    }

    let total_mb = total_bytes as f64 / 1_048_576.0;
    eprintln!("\n=== Totals ===\n");
    eprintln!("BF16 tensors:           {}", bf16.len());
    eprintln!("Total BF16 footprint:   {:.2} MB", total_mb);

    // INT8 with per-channel f32 scales saving estimate.
    // Per-channel scales: one f32 scale per output column.
    // For a [rows, cols] tensor: bytes_int8 = rows*cols + 4*cols (or 4*rows)
    // BF16 baseline: 2*rows*cols. Savings ≈ 50% minus scale overhead.
    let mut int8_estimate: u64 = 0;
    for (_, t) in &bf16 {
        let n_elem: u64 = t.dims.iter().product();
        let scale_axis_len = *t.dims.first().unwrap_or(&1);
        int8_estimate += n_elem + 4 * scale_axis_len;
    }
    let savings_mb = (total_bytes as i64 - int8_estimate as i64) as f64 / 1_048_576.0;
    let savings_pct = 100.0 * savings_mb / total_mb;
    eprintln!(
        "INT8 + per-channel scales:  {:.2} MB  (saves {:.2} MB, {:.1}%)",
        int8_estimate as f64 / 1_048_576.0,
        savings_mb,
        savings_pct
    );
    eprintln!("Note: scale axis assumed = first dim of each tensor; refine if quantizing along columns.");
}

fn classify_role(name: &str) -> &'static str {
    if name.contains("per_layer_token_embd") {
        "ple_token_embd"
    } else if name.contains("per_layer_model_proj") {
        "ple_model_proj"
    } else if name.contains("per_layer_proj_norm") {
        "ple_proj_norm"
    } else if name.ends_with("_norm.weight") || name.contains("_norm.") {
        "norm"
    } else if name.contains("inp_gate") || name.contains("proj.weight") {
        "ple_per_layer_inp_gate_or_proj"
    } else {
        "other"
    }
}
