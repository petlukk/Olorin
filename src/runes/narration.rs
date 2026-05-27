//! Post-generation guard against grid-continuation in rune narration.
//!
//! Fed a long, dense, numeric-leading grid, Gemma 4 (q3kffnimpl on Pi) does not
//! summarize — it emits ANOTHER data row (`24:00  3 files  0.1 MB`),
//! deterministically. The trigger is the row shape + count, not byte length:
//! long prose, json, and ≤22-row grids all narrate fine, so a byte/row cap
//! mis-fires in both directions (it silenced good long prose while a 24-row
//! grid at 722 B still slipped the same boundary). Instead we watch the
//! model's OUTPUT and discard the narration when it contains a line matching
//! the INPUT's dominant row shape — keying on the actual failure signature,
//! not a proxy. Pinned over three Pi runs on 2026-05-27 (0 false positives,
//! 0 false negatives across the eval set); see the harness
//! `tests/narration_length_vs_structure.rs`.

use std::collections::HashMap;

/// Collapse a line to a run-length class pattern: D=digit, A=alpha, S=space,
/// O=other, with adjacent equal classes merged. `08:00  4 files` -> `DODSDSA`.
/// Digit/space run lengths are folded away so rows with different values but
/// the same column layout share a signature.
fn line_shape(line: &str) -> String {
    let mut out = String::new();
    let mut last = '\0';
    for c in line.trim().chars() {
        let cls = if c.is_ascii_digit() { 'D' }
                  else if c.is_alphabetic() { 'A' }
                  else if c == ' ' { 'S' }
                  else { 'O' };
        if cls != last {
            out.push(cls);
            last = cls;
        }
    }
    out
}

/// The line shape occurring most in `text`, requiring ≥3 occurrences so only a
/// genuinely repetitive grid yields one. Prose and one-off lines return `None`,
/// which makes [`is_grid_continuation`] a guaranteed no-op for them.
fn dominant_shape(text: &str) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        *counts.entry(line_shape(l)).or_default() += 1;
    }
    counts
        .into_iter()
        .filter(|(_, c)| *c >= 3)
        .max_by_key(|(_, c)| *c)
        .map(|(s, _)| s)
}

/// True when `narration` contains a non-empty line whose shape matches the
/// dominant row shape of `rune_output` — i.e. the model continued the grid
/// instead of summarizing it. Returns `false` whenever `rune_output` is not a
/// repetitive grid (no dominant shape), so prose/json narration is never
/// suppressed.
pub fn is_grid_continuation(rune_output: &str, narration: &str) -> bool {
    let Some(dom) = dominant_shape(rune_output) else { return false };
    narration
        .lines()
        .any(|l| !l.trim().is_empty() && line_shape(l) == dom)
}
