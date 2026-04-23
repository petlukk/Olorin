//! Activation tracker end-to-end — runs a short decode with
//! `OLORIN_ACTIVATION_TRACK=1`, flushes the CSV, asserts shape.
//!
//! Run: cargo test --release --test activation_tracker -- --ignored --nocapture
//!
//! Needs ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf.

use olorin::inference::activation_track;
use olorin::inference::generate::{Engine, GenEvent};
use std::path::Path;

#[test]
#[ignore = "loads model + runs decode; use --ignored"]
fn tracker_records_ffn_stats_and_writes_csv() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }

    let pid = std::process::id();
    let out = std::env::temp_dir().join(format!("olorin_track_test_{pid}.csv"));
    std::env::set_var("OLORIN_ACTIVATION_TRACK", "1");
    std::env::set_var("OLORIN_ACTIVATION_DOMAIN", "test");
    std::env::set_var("OLORIN_ACTIVATION_OUT", &out);

    // Keep the output file around if OLORIN_ACTIVATION_KEEP is set (peek mode).
    let keep = std::env::var("OLORIN_ACTIVATION_KEEP").is_ok();
    struct CleanupFile(std::path::PathBuf, bool);
    impl Drop for CleanupFile {
        fn drop(&mut self) {
            if !self.1 { let _ = std::fs::remove_file(&self.0); }
        }
    }
    let _guard = CleanupFile(out.clone(), keep);

    let mut engine = Box::new(Engine::load(&path, 512).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = 8;

    let on_event = |_: GenEvent| {};
    engine.generate("Hello", "", &on_event).expect("generate");

    let written = activation_track::flush_csv().expect("flush");
    assert_eq!(written, out, "flush should write to OLORIN_ACTIVATION_OUT path");

    let contents = std::fs::read_to_string(&out).expect("read csv");
    let lines: Vec<&str> = contents.lines().collect();
    assert!(lines.len() >= 3, "CSV too short: {} lines", lines.len());
    assert!(lines[0].starts_with("# threshold="), "missing header comment: {}", lines[0]);
    assert!(lines[0].contains("domain=test"), "domain not in header: {}", lines[0]);
    assert_eq!(lines[1], "layer,neuron,count,sum_abs,sum_sq,max_abs,samples");

    // Sample first data row, confirm numeric shape.
    let first = lines[2].split(',').collect::<Vec<_>>();
    assert_eq!(first.len(), 7, "unexpected column count: {:?}", first);
    let samples: u32 = first[6].parse().expect("samples is u32");
    assert!(samples >= 8, "expected ≥8 samples (max_tokens), got {}", samples);

    // Decode-path records 35 layers × ffn_dim neurons. Prefill also goes
    // through forward_graph_layer now; don't assert exact line count.
    let data_rows = lines.len() - 2;
    assert!(data_rows >= 35 * 6144,
        "expected ≥{} rows (35 layers × 6144 neurons), got {}",
        35 * 6144, data_rows);
}
