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

    // The spike towers: the top data row carries an upper-half block (the
    // axis auto-zooms to the data band, so the spike nearly fills the top
    // cell — ▅▆▇█), and only a column or two reach it (a lone spike, not the
    // wide baseline plateau, which now collapses to the floor after zoom).
    let lines: Vec<&str> = chart.lines().collect();
    let top = lines[1]; // line 0 is the title
    let tall = top.chars().filter(|&c| "▅▆▇█".contains(c)).count();
    assert!(tall >= 1, "spike should reach the top row:\n{chart}");
    assert!(tall <= 3, "only the spike towers, not a plateau:\n{chart}");
    // Auto-zoom lifted the floor off zero: the lowest y-axis label is the
    // baseline band (~300), not 0 — the bottom of the canvas isn't wasted.
    assert!(
        !chart.contains("\n   0 ") && !chart.contains(" 0 ▁"),
        "high-floor series should auto-zoom off zero:\n{chart}"
    );

    // x-axis carries shortened HH:MM time ticks.
    assert!(chart.contains("08:00"), "first x-tick:\n{chart}");

    // Downsample actually happened: 120 source buckets, 56-column canvas.
    // Each data row is gutter + 56 chars; assert the canvas width.
    let widest = lines.iter().map(|l| l.chars().count()).max().unwrap();
    assert!(widest >= 56, "canvas should be ~56 cols wide, got {widest}:\n{chart}");
}

#[test]
fn multi_day_span_shows_dates_in_ticks() {
    ffi::init().expect("kernel init");
    // A 48-hour span crossing midnight: ticks must carry the date, else
    // HH:MM-only labels read backwards across the day boundary.
    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.categories = (0..48)
        .map(|h| {
            let day = 1 + h / 24;
            let hh = h % 24;
            Category {
                name: format!("1995-07-{day:02}T{hh:02}:00:00"),
                count: 100 + (h % 10) as u64,
            }
        })
        .collect();
    let chart = render_series(&out, 56, 8, false, None);
    assert!(
        chart.contains("07-01") || chart.contains("07-02"),
        "multi-day ticks should carry the MM-DD date:\n{chart}"
    );
}

#[test]
fn single_day_span_keeps_hhmm_ticks() {
    ffi::init().expect("kernel init");
    // All on one day → no date prefix, just HH:MM (no '-' inside a tick).
    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.categories = (0..20)
        .map(|i| Category { name: format!("2024-01-01T{:02}:00:00", 8 + i % 12), count: 50 })
        .collect();
    let chart = render_series(&out, 56, 6, false, None);
    // The x-tick line (the one with ':') must not contain a date dash.
    let tick_line = chart.lines().find(|l| l.contains(':')).unwrap_or("");
    assert!(
        !tick_line.contains('-'),
        "single-day ticks should be HH:MM only:\n{tick_line:?}"
    );
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
