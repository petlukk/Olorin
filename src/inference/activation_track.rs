//! Per-layer FFN activation statistics tracker.
//!
//! Off by default — enabled via `OLORIN_ACTIVATION_TRACK=1`. When enabled,
//! every call to `record(layer, hidden)` accumulates four stats per neuron:
//!   count   = #{|x| > threshold}
//!   sum_abs = Σ|x|
//!   sum_sq  = Σx²
//!   max_abs = max|x|
//!
//! `flush_csv()` writes the per-neuron rows to `OLORIN_ACTIVATION_OUT`
//! (default `/tmp/olorin_activation.csv`). Call once at end of run.
//!
//! Domain routing: set `OLORIN_ACTIVATION_DOMAIN=code` (or prose, dialogue,
//! …) per-run. The label is written to the CSV header so post-processing
//! can diff masks across domains.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

struct LayerStats {
    count: Vec<u32>,
    sum_abs: Vec<f32>,
    sum_sq: Vec<f32>,
    max_abs: Vec<f32>,
    samples: u32,
}

impl LayerStats {
    fn new(n: usize) -> Self {
        Self {
            count: vec![0; n],
            sum_abs: vec![0.0; n],
            sum_sq: vec![0.0; n],
            max_abs: vec![0.0; n],
            samples: 0,
        }
    }

    fn observe(&mut self, hidden: &[f32], threshold: f32) {
        debug_assert_eq!(hidden.len(), self.count.len(),
            "activation_track: per-layer width must be stable across calls");
        for (i, &x) in hidden.iter().enumerate() {
            let a = x.abs();
            if a > threshold {
                self.count[i] += 1;
            }
            self.sum_abs[i] += a;
            self.sum_sq[i] += x * x;
            if a > self.max_abs[i] {
                self.max_abs[i] = a;
            }
        }
        self.samples += 1;
    }
}

struct Tracker {
    layers: Vec<LayerStats>,
    threshold: f32,
    domain: String,
}

static ENABLED: OnceLock<bool> = OnceLock::new();
static TRACKER: OnceLock<Mutex<Tracker>> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("OLORIN_ACTIVATION_TRACK").is_ok())
}

fn tracker() -> &'static Mutex<Tracker> {
    TRACKER.get_or_init(|| {
        let threshold = std::env::var("OLORIN_ACTIVATION_THRESHOLD")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(1e-3);
        let domain = std::env::var("OLORIN_ACTIVATION_DOMAIN")
            .unwrap_or_else(|_| "unknown".to_string());
        Mutex::new(Tracker { layers: Vec::new(), threshold, domain })
    })
}

/// Record one FFN hidden vector for `layer`. No-op when tracker is disabled.
/// Call from the main orchestration thread (ith == 0) after gelu_mul so the
/// full vector is visible.
pub fn record(layer: usize, hidden: &[f32]) {
    if !enabled() { return; }
    let mut g = tracker().lock().unwrap_or_else(|e| e.into_inner());
    let threshold = g.threshold;
    while g.layers.len() <= layer {
        g.layers.push(LayerStats::new(hidden.len()));
    }
    g.layers[layer].observe(hidden, threshold);
}

/// Clear all accumulated stats and set a new domain label. Used when the
/// same process runs multiple experiments (e.g. code then text).
/// Requires the tracker to already be initialized — no-op otherwise.
pub fn reset(new_domain: &str) {
    let Some(lock) = TRACKER.get() else { return; };
    let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
    g.layers.clear();
    g.domain.clear();
    g.domain.push_str(new_domain);
}

/// In-memory snapshot of per-layer mean |x| across observed samples.
/// Returns `Vec<Vec<f32>>` indexed as `[layer][neuron]`. Empty when disabled.
pub fn per_layer_mean_abs() -> Vec<Vec<f32>> {
    let Some(lock) = TRACKER.get() else { return Vec::new(); };
    let g = lock.lock().unwrap_or_else(|e| e.into_inner());
    g.layers.iter().map(|s| {
        let n = s.samples.max(1) as f32;
        s.sum_abs.iter().map(|&v| v / n).collect()
    }).collect()
}

/// Per-layer max |x| observed. Safety-check for pruning: any neuron with
/// high max_abs has fired hard at least once and shouldn't be pruned even
/// if its mean is low.
pub fn per_layer_max_abs() -> Vec<Vec<f32>> {
    let Some(lock) = TRACKER.get() else { return Vec::new(); };
    let g = lock.lock().unwrap_or_else(|e| e.into_inner());
    g.layers.iter().map(|s| s.max_abs.clone()).collect()
}

/// Per-layer sample count — number of tokens observed per layer.
pub fn per_layer_samples() -> Vec<u32> {
    let Some(lock) = TRACKER.get() else { return Vec::new(); };
    let g = lock.lock().unwrap_or_else(|e| e.into_inner());
    g.layers.iter().map(|s| s.samples).collect()
}

/// Flush all accumulated stats to `OLORIN_ACTIVATION_OUT` as CSV. Returns
/// the written path. No-op (returns Ok with default path) when tracker was
/// never initialized.
pub fn flush_csv() -> std::io::Result<PathBuf> {
    use std::io::Write;
    let path = PathBuf::from(
        std::env::var("OLORIN_ACTIVATION_OUT")
            .unwrap_or_else(|_| "/tmp/olorin_activation.csv".to_string()),
    );
    let Some(lock) = TRACKER.get() else {
        return Ok(path);
    };
    let g = lock.lock().unwrap_or_else(|e| e.into_inner());
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path)?);
    writeln!(f, "# threshold={} domain={} layers={}",
        g.threshold, g.domain, g.layers.len())?;
    writeln!(f, "layer,neuron,count,sum_abs,sum_sq,max_abs,samples")?;
    for (li, s) in g.layers.iter().enumerate() {
        for n in 0..s.count.len() {
            writeln!(f, "{},{},{},{},{},{},{}",
                li, n, s.count[n], s.sum_abs[n], s.sum_sq[n], s.max_abs[n], s.samples)?;
        }
    }
    f.flush()?;
    eprintln!("[activation-track] wrote {} ({} layers × {} neurons each)",
        path.display(),
        g.layers.len(),
        g.layers.first().map(|s| s.count.len()).unwrap_or(0));
    Ok(path)
}
