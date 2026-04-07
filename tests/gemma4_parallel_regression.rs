//! Bit-exact regression for parallelized inference.
//!
//! On first run with no snapshot, captures the current logits as ground truth
//! and panics, forcing a re-run. Every subsequent run asserts the new logits
//! match the snapshot byte-for-byte. This guards parallelization changes
//! against numerical drift — any tid/slab off-by-one or accidental aliasing
//! will produce a different bit pattern.
//!
//! Requires: ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf

use std::fs;
use std::path::{Path, PathBuf};

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

fn snapshot_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/snapshots/gemma4_logits_bos.bin");
    p
}

#[test]
fn forward_one_bos_logits_bit_exact() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: {} not present", model_path());
        return;
    }

    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let pool = olorin::inference::threadpool::ThreadPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);

    // BOS token id = 2 (matches gemma4_verify::step5_logits)
    let logits = state.forward_one(&model, 2, &pool).to_vec();

    // Serialize as raw little-endian f32 bytes.
    let mut bytes = Vec::with_capacity(logits.len() * 4);
    for v in &logits {
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    let path = snapshot_path();
    if !path.exists() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, &bytes).unwrap();
        panic!(
            "captured baseline snapshot at {} ({} bytes, {} logits) — re-run test to verify",
            path.display(),
            bytes.len(),
            logits.len(),
        );
    }

    let expected = fs::read(&path).unwrap();
    assert_eq!(
        bytes.len(),
        expected.len(),
        "logits length changed: got {} bytes, snapshot {}",
        bytes.len(),
        expected.len(),
    );
    assert!(
        bytes == expected,
        "logits drifted from snapshot — parallelization changed numerics"
    );
}
