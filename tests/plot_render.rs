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
