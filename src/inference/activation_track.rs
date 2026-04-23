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

// ── Residual-norm tracking (cheap: 4 bytes/layer/token) ──────────────
static RESIDUAL_ENABLED: OnceLock<bool> = OnceLock::new();
static RESIDUAL_NORMS: OnceLock<Mutex<Vec<Vec<f32>>>> = OnceLock::new();

// ── Residual-snapshot tracking (heavy: hidden_dim × 4 bytes/layer/token) ─
static SNAPSHOT_ENABLED: OnceLock<bool> = OnceLock::new();
#[allow(clippy::type_complexity)]
static RESIDUAL_SNAPSHOTS: OnceLock<Mutex<Vec<Vec<Vec<f32>>>>> = OnceLock::new();
// Indexed as [token_idx][layer][neuron].

// ── Logit-entropy tracking (one f32 per decoded token) ───────────────
static ENTROPY_ENABLED: OnceLock<bool> = OnceLock::new();
static LOGIT_ENTROPIES: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();

fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("OLORIN_ACTIVATION_TRACK").is_ok())
}

fn residual_enabled() -> bool {
    *RESIDUAL_ENABLED.get_or_init(|| std::env::var("OLORIN_RESIDUAL_TRACK").is_ok())
}

fn snapshot_enabled() -> bool {
    *SNAPSHOT_ENABLED.get_or_init(|| std::env::var("OLORIN_RESIDUAL_SNAPSHOT").is_ok())
}

fn entropy_enabled() -> bool {
    *ENTROPY_ENABLED.get_or_init(|| std::env::var("OLORIN_LOGIT_ENTROPY").is_ok())
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

// ─────────────────────────────────────────────────────────────────────
// Residual-norm tracking
// ─────────────────────────────────────────────────────────────────────

/// Record the L2 norm of the residual stream at the end of `layer`.
/// No-op unless OLORIN_RESIDUAL_TRACK=1. Called once per decoded token.
pub fn record_residual_norm(layer: usize, residual: &[f32]) {
    if !residual_enabled() { return; }
    let sq: f32 = residual.iter().map(|&v| v * v).sum();
    let norm = sq.sqrt();
    let lock = RESIDUAL_NORMS.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
    while g.len() <= layer {
        g.push(Vec::new());
    }
    g[layer].push(norm);
}

/// Per-layer sequence of observed residual L2 norms.
pub fn residual_norms() -> Vec<Vec<f32>> {
    let Some(lock) = RESIDUAL_NORMS.get() else { return Vec::new(); };
    lock.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

// ─────────────────────────────────────────────────────────────────────
// Residual snapshots (for offline early-exit reprojection)
// ─────────────────────────────────────────────────────────────────────

/// Record the full residual state at the end of `layer` for the current
/// token. A "new token" is detected by seeing layer == 0 again. Heavy —
/// allocates one `Vec<f32>` (hidden_dim * 4 bytes) per layer per token.
/// No-op unless OLORIN_RESIDUAL_SNAPSHOT=1.
pub fn record_residual_snapshot(layer: usize, residual: &[f32]) {
    if !snapshot_enabled() { return; }
    let lock = RESIDUAL_SNAPSHOTS.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = lock.lock().unwrap_or_else(|e| e.into_inner());
    // Start a new token slot when we see layer 0 again.
    if layer == 0 || g.is_empty() {
        g.push(Vec::new());
    }
    let cur = g.last_mut().unwrap();
    while cur.len() <= layer {
        cur.push(Vec::new());
    }
    cur[layer] = residual.to_vec();
}

/// All snapshots captured since the last `reset_snapshots()`.
/// Indexed as `[token_idx][layer][neuron]`.
pub fn residual_snapshots() -> Vec<Vec<Vec<f32>>> {
    let Some(lock) = RESIDUAL_SNAPSHOTS.get() else { return Vec::new(); };
    lock.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Drop all accumulated snapshots (free memory before a new experiment).
pub fn reset_snapshots() {
    if let Some(lock) = RESIDUAL_SNAPSHOTS.get() {
        lock.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}

// ─────────────────────────────────────────────────────────────────────
// Logit-entropy tracking
// ─────────────────────────────────────────────────────────────────────

/// Record the softmax entropy of a per-token logit vector.
/// Entropy is in nats (natural log). No-op unless OLORIN_LOGIT_ENTROPY=1.
pub fn record_logit_entropy(logits: &[f32]) {
    if !entropy_enabled() { return; }
    if logits.is_empty() { return; }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0.0f32;
    for &l in logits {
        z += (l - max).exp();
    }
    let log_z = z.ln();
    // H = -Σ p log p = log Z - (1/Z) Σ exp(l-max) * (l-max)
    // Equivalently: H = log Z + max - Σ p * l  where p = exp(l-max)/Z
    let mut entropy = 0.0f32;
    for &l in logits {
        let p = (l - max).exp() / z;
        if p > 1e-20 {
            entropy -= p * (p.ln());
        }
    }
    let _ = log_z; // computed for documentation symmetry, not needed for the loop above
    let lock = LOGIT_ENTROPIES.get_or_init(|| Mutex::new(Vec::new()));
    lock.lock().unwrap_or_else(|e| e.into_inner()).push(entropy);
}

/// All recorded per-token logit entropies since start / last reset.
pub fn logit_entropies() -> Vec<f32> {
    let Some(lock) = LOGIT_ENTROPIES.get() else { return Vec::new(); };
    lock.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Drop residual norms + logit entropies (keep snapshots — use reset_snapshots).
pub fn reset_telemetry() {
    if let Some(lock) = RESIDUAL_NORMS.get() {
        lock.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    if let Some(lock) = LOGIT_ENTROPIES.get() {
        lock.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
}
