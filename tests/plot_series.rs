//! Rung-4 verify: a chronological series `RuneOutput`, round-tripped
//! through the v1 JSON contract (to_json → from_json — the exact seam the
//! file-drop flow's `chart_for` uses), renders to a mockup-shaped chart.
//! Exercises the `col_reduce` SIMD downsample (120 buckets → 56 columns)
//! plus the renderer end to end.

use olorin::kernels::ffi;
use olorin::runes::output::{Anomaly, Category, RuneOutput, Source, Totals};
use olorin::runes::plot::render_series;

const RUNE_VERSION: i64 = 1;

/// Build a NASA-July-1995-style series: ~120 five-minute buckets at a
/// baseline of 300, with a sharp spike of 592 near the middle.
fn nasa_series() -> RuneOutput {
    let n = 120usize;
    let spike_idx = 60usize;
    let baseline = 300u64;

    let mut categories = Vec::with_capacity(n);
    for i in 0..n {
        // ISO instants on 1995-07-13, 5-minute steps from 08:00.
        let minutes = i * 5;
        let hh = 8 + minutes / 60;
        let mm = minutes % 60;
        let name = format!("1995-07-13T{hh:02}:{mm:02}:00");
        // Mild deterministic ripple around the baseline, big spike at peak.
        let count = if i == spike_idx {
            592
        } else {
            baseline + ((i * 7) % 11) as u64 - 5
        };
        categories.push(Category { name, count });
    }

    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.source = Some(Source {
        path: "/tmp/olorin-tmp-9f3a.log".to_string(),
        bytes: 4_200_000,
        format: "log/plain".to_string(),
    });
    out.totals = Totals { rows: categories.iter().map(|c| c.count).sum(), scan_us: 19_000 };
    out.categories = categories;
    out.anomalies = vec![Anomaly {
        bucket:   "1995-07-13T13:00:00".to_string(),
        count:    592,
        baseline: 300.0,
        ratio:    1.97,
        score:    9.4,
    }];
    out
}

#[test]
fn nasa_spike_renders_mockup_shape() {
    ffi::init().expect("kernel init");

    // Round-trip through the JSON contract — the file-drop flow gets the
    // RuneOutput back from eatime's --json answer the same way.
    let json = nasa_series().to_json();
    let out = RuneOutput::from_json(json.as_bytes()).expect("v1 reader");

    let chart = render_series(&out, 56, 10, false, Some("anomalies.log"));

    // Title is the friendly display name, NOT the temp path in `source`.
    assert!(chart.contains("anomalies.log"), "title override:\n{chart}");
    assert!(
        !chart.contains("/tmp/olorin-tmp"),
        "temp path must not leak into the chart title:\n{chart}"
    );

    // Median baseline legend matches the anomaly baseline (300).
    assert!(chart.contains("median (300)"), "median legend:\n{chart}");

    // The spike towers: the top data row carries a tall block (▇/█ — the
    // spike is 592/600 so it nearly, not fully, fills the top cell), and
    // only a column or two reach it (a lone spike, not the wide baseline
    // plateau that fills the lower rows).
    let lines: Vec<&str> = chart.lines().collect();
    let top = lines[1]; // line 0 is the title
    let tall = top.chars().filter(|&c| c == '█' || c == '▇').count();
    assert!(tall >= 1, "spike should reach the top row:\n{chart}");
    assert!(tall <= 3, "only the spike towers, not a plateau:\n{chart}");

    // x-axis carries shortened HH:MM time ticks.
    assert!(chart.contains("08:00"), "first x-tick:\n{chart}");

    // Downsample actually happened: 120 source buckets, 56-column canvas.
    // Each data row is gutter + 56 chars; assert the canvas width.
    let widest = lines.iter().map(|l| l.chars().count()).max().unwrap();
    assert!(widest >= 56, "canvas should be ~56 cols wide, got {widest}:\n{chart}");
}

#[test]
fn flat_series_has_no_spike_but_still_charts() {
    ffi::init().expect("kernel init");
    // A perfectly flat series (no anomalies) still renders, with the median
    // line drawn from the robust median of the counts.
    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.categories = (0..30)
        .map(|i| Category { name: format!("2024-01-01T{i:02}:00:00"), count: 100 })
        .collect();
    let chart = render_series(&out, 40, 6, false, None);
    assert!(chart.contains("median (100)"), "flat-series median:\n{chart}");
}
