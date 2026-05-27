//! Regression for the post-generation grid-continuation guard
//! (src/runes/narration.rs). Fed a long numeric grid, the production quant
//! sometimes emits another data row instead of a summary; such narrations are
//! discarded so the kernel output stands alone. Pinned empirically over three
//! Pi runs on 2026-05-27 (see the architecture-narration-600b-cap memory and
//! the tests/narration_length_vs_structure.rs harness). Model-free: it
//! exercises the detector logic directly.

use olorin::runes::narration::is_grid_continuation;

/// A 24-row hour grid in the shape that fails on the production quant. The
/// `Output of` header mirrors what build_narration_prompt prepends.
fn grid() -> String {
    let mut s = String::from("Output of `eatime`:\n\n");
    for h in 0..24 {
        s.push_str(&format!(
            "{h:02}:00  {} files  {:.1} MB  {:.1}%\n",
            h % 7, h as f32 * 0.3, h as f32 * 0.5
        ));
    }
    s
}

#[test]
fn flags_a_single_row_continuation() {
    // The exact failure mode: the model emitted another data row.
    assert!(is_grid_continuation(&grid(), "24:00  3 files  2.4 MB  6.1%"));
}

#[test]
fn flags_a_multiline_continuation() {
    let out = "24:00  3 files  0.1 MB  0.0%\n25:00  0 files  0.0 MB  0.0%";
    assert!(is_grid_continuation(&grid(), out));
}

#[test]
fn passes_a_clean_summary_that_names_the_peak() {
    // A good summary may cite the peak hour ("09:00") — that must NOT be
    // mistaken for a grid row. This is the false-positive the old echoed-value
    // heuristic suffered and the shape detector must avoid.
    let out = "Activity peaks around the 09:00 hour, then tapers off through the evening.";
    assert!(!is_grid_continuation(&grid(), out));
}

#[test]
fn prose_input_can_never_be_flagged() {
    // Prose has no dominant repeated row shape, so the guard is a no-op for it
    // regardless of output — long prose/json narration is never suppressed.
    let prose = "The scan covered 1284 files and found nothing dangerous. \
                 The invoice folder is the only area still growing.";
    assert!(!is_grid_continuation(prose, "24:00 3 files 0.1 MB"));
}
