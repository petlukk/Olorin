//! Golden tests for the pure block-bar renderer. `render(&Bars)` is total
//! scalar string assembly — no kernel, no I/O — so its output is asserted
//! exactly. Color is verified separately so the golden grid stays readable.

use olorin::runes::plot::{render, Bars, XTick};

/// A small hand-checkable chart: 8 columns, a clear peak at column 5, a
/// median line at 20, no color. The exact grid is pinned so any change to
/// glyph mapping, scaling, axis, or median placement trips the test.
#[test]
fn golden_block_bars_no_color() {
    let heights = [10.0f32, 20.0, 30.0, 20.0, 40.0, 100.0, 30.0, 15.0];
    let spike = [false, false, false, false, false, true, false, false];
    let ticks = [
        XTick { col: 0, label: "08h".into() },
        XTick { col: 7, label: "20h".into() },
    ];
    let bars = Bars {
        title: Some("spike"),
        heights: &heights,
        spike: &spike,
        median: Some(20.0),
        x_ticks: &ticks,
        height_rows: 5,
        color: false,
    };
    let got = render(&bars);

    // Exact grid pin (trailing spaces trimmed per line — they're invisible
    // and alignment-irrelevant). Any change to glyph mapping, vertical
    // scaling, the median row, or axis layout trips this.
    //   col 5 (=y_max 100) is the lone full-height bar; the median(20)
    //   dashed line shows through row-1 gaps; 20h is right-anchored.
    let expected = "     spike
 100      █
          █
  60      █
     ──▄─██▄─
  20 ▄██████▆
     ────────
     08h  20h
     median (20)";
    assert_eq!(trim_trailing(&got), expected, "\n--- got ---\n{got}");
}

/// Strip trailing spaces from every line and drop the final newline, so a
/// golden literal can't drift on invisible whitespace.
fn trim_trailing(s: &str) -> String {
    s.lines().map(|l| l.trim_end()).collect::<Vec<_>>().join("\n")
}

#[test]
fn median_line_shows_through_gaps() {
    // A flat low series with the median above all bars: the median row must
    // draw the dashed baseline across every column (no bar occludes it).
    let heights = [1.0f32, 1.0, 1.0, 1.0];
    let spike = [false; 4];
    let bars = Bars {
        title: None,
        heights: &heights,
        spike: &spike,
        median: Some(50.0),
        x_ticks: &[],
        height_rows: 4,
        color: false,
    };
    let got = render(&bars);
    assert!(
        got.contains("────"),
        "median baseline should span the canvas:\n{got}"
    );
}

#[test]
fn color_wraps_spike_columns_in_red() {
    let heights = [10.0f32, 90.0];
    let spike = [false, true];
    let bars = Bars {
        title: None,
        heights: &heights,
        spike: &spike,
        median: None,
        x_ticks: &[],
        height_rows: 3,
        color: true,
    };
    let got = render(&bars);
    // Spike column glyphs are wrapped in the red SGR + reset.
    assert!(got.contains("\x1b[31m"), "expected red for spike column:\n{got:?}");
    assert!(got.contains("\x1b[0m"), "expected reset:\n{got:?}");
}

/// Extract the y-axis gridline label values from a chart (lines of the form
/// `<spaces><digits> <chart cells>`). Skips title, x-ticks (contain ':'),
/// and the median legend.
fn y_labels(chart: &str) -> Vec<i64> {
    chart
        .lines()
        .filter_map(|l| {
            let trimmed = l.trim_start();
            let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                return None;
            }
            let rest = &trimmed[digits.len()..];
            // A y-label is "<num> <cells>"; a time tick is "08:00" (digit then ':').
            if !rest.starts_with(' ') || rest.contains(':') {
                return None;
            }
            digits.parse::<i64>().ok()
        })
        .collect()
}

/// Ratio of the lowest to highest y-axis label — near 0 for a zero-based
/// chart, near 1 when the floor has been lifted close to the ceiling.
fn label_floor_ratio(chart: &str) -> f64 {
    let labels = y_labels(chart);
    match (labels.iter().min(), labels.iter().max()) {
        (Some(&lo), Some(&hi)) if hi > 0 => lo as f64 / hi as f64,
        _ => 0.0,
    }
}

#[test]
fn high_floor_series_auto_zooms_off_zero() {
    // A dense high-floor band (1500..1900): the axis should lift its floor so
    // the variation band uses the canvas instead of a solid block from zero.
    let heights: Vec<f32> = (0..24).map(|i| 1500.0 + (i % 5) as f32 * 100.0).collect();
    let spike = vec![false; heights.len()];
    let bars = Bars {
        title: None,
        heights: &heights,
        spike: &spike,
        median: Some(1700.0),
        x_ticks: &[],
        height_rows: 8,
        color: false,
    };
    let got = render(&bars);
    // Floor sits high relative to the ceiling → the band, not zero, anchors.
    assert!(
        label_floor_ratio(&got) > 0.5,
        "high-floor series should lift its baseline near the data band:\n{got}"
    );
}

#[test]
fn spiky_low_floor_series_keeps_zero_baseline() {
    // A low baseline (~200) with a big spike (5000): keep a zero baseline so
    // the spike's magnitude reads true rather than being zoomed away.
    let mut heights = vec![200.0f32; 24];
    heights[12] = 5000.0;
    let spike: Vec<bool> = heights.iter().map(|&h| h > 1000.0).collect();
    let bars = Bars {
        title: None,
        heights: &heights,
        spike: &spike,
        median: Some(200.0),
        x_ticks: &[],
        height_rows: 8,
        color: false,
    };
    let got = render(&bars);
    // Floor near zero → the lowest gridline is a small fraction of the top.
    assert!(
        label_floor_ratio(&got) < 0.4,
        "spiky low-floor series should keep a near-zero baseline:\n{got}"
    );
}

#[test]
fn empty_series_is_graceful() {
    let bars = Bars {
        title: None,
        heights: &[],
        spike: &[],
        median: None,
        x_ticks: &[],
        height_rows: 5,
        color: false,
    };
    assert_eq!(render(&bars), "(no data to plot)\n");
}
