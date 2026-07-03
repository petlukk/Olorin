//! Golden tests for the pure block-bar renderer. `render(&Bars)` is total
//! scalar string assembly — no kernel, no I/O — so its output is asserted
//! exactly. Color is verified separately so the golden grid stays readable.

use olorin::runes::output::{Anomaly, Category, RuneOutput, Source};
use olorin::runes::plot::{render, render_series, Bars, Ink, XTick};

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
        ink: Ink::Plain,
        zero_based: false,
        baseline_label: "median",
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
        ink: Ink::Plain,
        zero_based: false,
        baseline_label: "median",
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
        ink: Ink::Ansi,
        zero_based: false,
        baseline_label: "median",
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
        ink: Ink::Plain,
        zero_based: false,
        baseline_label: "median",
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
        ink: Ink::Plain,
        zero_based: false,
        baseline_label: "median",
    };
    let got = render(&bars);
    // Floor near zero → the lowest gridline is a small fraction of the top.
    assert!(
        label_floor_ratio(&got) < 0.4,
        "spiky low-floor series should keep a near-zero baseline:\n{got}"
    );
}

/// An eanet source fan-out ranking: five same-/near-subnet scanner hosts all
/// contacting ~2000 destinations, with the scan anomaly carrying the fan-out
/// median (5 — what a normal host contacts) as its baseline.
fn fanout_output() -> RuneOutput {
    let mut out = RuneOutput::new("eanet", 1);
    out.source = Some(Source { path: "cap.pcap".into(), bytes: 1000, format: "pcap".into() });
    for (ip, n) in [
        ("192.168.202.73", 2049u64),
        ("192.168.202.102", 2049),
        ("192.168.204.45", 2036),
        ("192.168.202.110", 2030),
        ("192.168.202.108", 1967),
    ] {
        out.categories.push(Category { name: ip.into(), count: n });
    }
    out.anomalies.push(Anomaly {
        bucket: "192.168.202.73".into(),
        count: 2049,
        baseline: 5.0,
        ratio: 409.8,
        score: 409.8,
    });
    out
}

#[test]
fn eanet_fanout_chart_is_readable() {
    // The bug this fixes: a 5px sliver of identical "192.168." bars, a lifted
    // 1600 floor compressing the ~2000 values, and a stray eatime "median (5)".
    let out = fanout_output();
    let chart = render_series(&out, 56, 10, Ink::Plain, None);

    // Labels drop the shared 192.168 prefix — the third octet varies (202 vs
    // 204) so two octets remain. The old truncation collapsed all to "192.168.".
    assert!(chart.contains("202.73"), "octet-stripped label missing:\n{chart}");
    assert!(!chart.contains("192.168."), "shared prefix must be stripped:\n{chart}");

    // The baseline is the fan-out median (5), labeled "typical" not "median".
    assert!(chart.contains("typical (5)"), "typical caption missing:\n{chart}");
    assert!(!chart.contains("median ("), "eanet must not borrow eatime's 'median':\n{chart}");

    // Zero-based ranking: the axis floor sits near zero so the bars tower over
    // the typical(5) reference, instead of the old lifted 1600 floor.
    assert!(
        label_floor_ratio(&chart) < 0.4,
        "fan-out ranking must be zero-based, not floor-lifted:\n{chart}"
    );
    assert!(chart.contains('█'), "scanner bars should tower with solid blocks:\n{chart}");

    // Wide, readable bars — the full canvas, not a 5-column sliver.
    let widest = chart.lines().map(|l| l.chars().count()).max().unwrap();
    assert!(widest >= 40, "fan-out bars should be widened, got {widest}:\n{chart}");
}

#[test]
fn eanet_web_ink_marks_scanner_and_bars() {
    // Web mode brackets bar runs in PUA sentinels the frontend colours: the
    // flagged scanner (spike) in U+E002..E003 (red), other bars in U+E004..E005
    // (accent). Restores the "scanner towers red" the ANSI/terminal + SVG paths
    // have but the monochrome web `<pre>` dropped. No ANSI leaks into the web.
    let out = fanout_output(); // 192.168.202.73 is the flagged scanner
    let chart = render_series(&out, 56, 10, Ink::Web, None);
    assert!(chart.contains('\u{E002}'), "scanner bar must open a spike run:\n{chart:?}");
    assert!(chart.contains('\u{E003}'), "spike run must close:\n{chart:?}");
    assert!(chart.contains('\u{E004}'), "non-scanner bars must open a bar run:\n{chart:?}");
    assert!(chart.contains('\u{E005}'), "bar run must close:\n{chart:?}");
    assert!(!chart.contains('\u{1b}'), "web chart must not contain ANSI escapes:\n{chart:?}");
    // Sentinels are balanced (every open has a matching close).
    assert_eq!(
        chart.matches('\u{E002}').count(),
        chart.matches('\u{E003}').count(),
        "unbalanced spike sentinels:\n{chart:?}"
    );
    assert_eq!(
        chart.matches('\u{E004}').count(),
        chart.matches('\u{E005}').count(),
        "unbalanced bar sentinels:\n{chart:?}"
    );
}

#[test]
fn plain_and_ansi_inks_carry_no_web_sentinels() {
    // The web PUA sentinels must never leak into the terminal (Ansi) or the
    // golden (Plain) surfaces.
    let out = fanout_output();
    for ink in [Ink::Plain, Ink::Ansi] {
        let chart = render_series(&out, 56, 10, ink, None);
        assert!(
            !chart.contains('\u{E002}') && !chart.contains('\u{E004}'),
            "web sentinels leaked into a non-web ink:\n{chart:?}"
        );
    }
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
        ink: Ink::Plain,
        zero_based: false,
        baseline_label: "median",
    };
    assert_eq!(render(&bars), "(no data to plot)\n");
}
