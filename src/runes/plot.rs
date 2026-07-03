//! plot — block-bar canvas renderer. Turns a column-height series into a
//! colorized terminal chart string (the same string is shown in the REPL
//! and dropped into a `<pre>` in the web UI, so REPL/web stay pixel-identical).
//!
//! Two layers:
//!   * `render(&Bars)` — pure scalar string assembly. No kernel, no I/O;
//!     given heights + annotations it emits the grid. Golden-testable.
//!   * `render_series(&RuneOutput)` — the bridge: extracts the category
//!     counts, routes them through the `col_reduce` SIMD kernel to fit the
//!     canvas width (peak-per-column so a 1-bucket spike still towers),
//!     marks anomaly columns, and pulls the median baseline from the
//!     detected spikes.
//!
//! Vertical resolution is 8× the row count: each cell is one of the eight
//! Unicode block eighths ▁▂▃▄▅▆▇█ (U+2581..U+2588), so an N-row canvas
//! resolves 8N sub-levels.

use super::output::RuneOutput;
use crate::kernels::ffi;

/// Block eighths indexed by filled-eighth count 0..=8 (0 = empty cell).
const BLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

const C_RED:   &str = "\x1b[31m";
const C_DIM:   &str = "\x1b[2m";
const C_RESET: &str = "\x1b[0m";

/// One labeled tick on the x-axis: the column it sits under and its text.
pub struct XTick {
    pub col:   usize,
    pub label: String,
}

/// Everything `render` needs: a height per column, a spike flag per column,
/// an optional median baseline, x-axis ticks, the canvas row count, a
/// title, and whether to emit ANSI color.
pub struct Bars<'a> {
    pub title:       Option<&'a str>,
    pub heights:     &'a [f32],
    pub spike:       &'a [bool],
    pub median:      Option<f32>,
    pub x_ticks:     &'a [XTick],
    pub height_rows: usize,
    pub color:       bool,
    /// Force a zero baseline (a magnitude *ranking* — bars that tower from
    /// zero) instead of the auto-lifted variation band. A time-bucketed rate
    /// is a level signal where the variation matters, so it lifts; a fan-out
    /// ranking's whole point is "2000 vs a typical 5", which zero shows.
    pub zero_based:  bool,
    /// Legend word for the baseline line: "median" for a time series's
    /// spike reference, "typical" for a fan-out ranking's normal host.
    pub baseline_label: &'a str,
}

/// Round up to a "nice" axis ceiling (1, 2, or 5 × 10^k) so y-labels read
/// cleanly instead of landing on 5873.
pub(super) fn nice_ceil(x: f32) -> f32 {
    if x <= 0.0 {
        return 1.0;
    }
    let exp = x.log10().floor();
    let base = 10f32.powf(exp);
    let frac = x / base; // in [1, 10)
    // Rich step set so the axis hugs the data (592 → 600, not 1000) and the
    // tallest bar uses most of the canvas height.
    const STEPS: [f32; 10] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    let nice = STEPS.iter().copied().find(|&s| s >= frac - 1e-6).unwrap_or(10.0);
    nice * base
}

/// Largest "nice" number (1/1.5/2/2.5/3/4/5/6/8 ×10^k) at or below x.
fn nice_floor(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let exp = x.log10().floor();
    let base = 10f32.powf(exp);
    let frac = x / base; // in [1, 10)
    const STEPS: [f32; 9] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0];
    let nice = STEPS.iter().rev().copied().find(|&s| s <= frac + 1e-6).unwrap_or(1.0);
    nice * base
}

/// Choose the y-axis floor. Returns 0 for a normal zero-based chart, or a
/// lifted "nice" floor when the bulk of the series sits high above zero so
/// the variation band uses the full canvas height instead of a solid block.
/// Uses a robust 10th-percentile low so one near-empty bucket can't veto the
/// zoom, lifts only when that low clears 30% of the ceiling (spiky low-floor
/// series stay zero-based), and never when the nice floor would collapse the
/// range (flat series).
fn baseline_floor(heights: &[f32], y_max: f32) -> f32 {
    if heights.len() < 4 {
        return 0.0;
    }
    let mut sorted: Vec<f32> = heights.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p10 = sorted[sorted.len() / 10];
    if p10 <= 0.30 * y_max {
        return 0.0;
    }
    let floor = nice_floor(p10);
    if floor < y_max {
        floor
    } else {
        0.0
    }
}

/// Filled eighths for column `c` at row `r` (0 = bottom row), given the
/// total eighths the bar occupies and the row's eighth offset.
fn cell_eighths(total_eighths: i32, r: usize) -> i32 {
    (total_eighths - (r as i32) * 8).clamp(0, 8)
}

/// Render a block-bar chart to a string. Pure: no kernel, no I/O.
pub fn render(b: &Bars) -> String {
    let n = b.heights.len();
    if n == 0 || b.height_rows == 0 {
        return "(no data to plot)\n".to_string();
    }
    let rows = b.height_rows;

    // Vertical scale. The ceiling is a nice number above the tallest bar
    // (or the median, if higher) so nothing clips. The floor auto-lifts off
    // zero when the bulk of the series sits high above it — a time-bucketed
    // rate is a level signal, so the variation band matters more than
    // magnitude-from-zero, and a high baseline would otherwise waste the
    // bottom of the canvas on identical solid fill. Spiky low-floor series
    // (peak >> bulk) keep a zero baseline so a spike still reads true.
    let peak = b
        .heights
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(b.median.unwrap_or(0.0));
    let y_max = nice_ceil(peak);
    let y_min = if b.zero_based { 0.0 } else { baseline_floor(b.heights, y_max) };
    let range = (y_max - y_min).max(f32::EPSILON);

    // Per-column total eighths over [y_min, y_max]. Bars below the floor
    // clamp to empty (cell_eighths floors at 0).
    let total_eighths: Vec<i32> = b
        .heights
        .iter()
        .map(|&h| (((h - y_min) / range) * (rows as f32) * 8.0).round() as i32)
        .collect();

    // Row that the median baseline falls in (its dashed line is drawn
    // through the gaps where no bar reaches).
    let med_row: Option<usize> = b.median.map(|m| {
        ((((m - y_min) / range) * rows as f32).floor().max(0.0) as usize)
            .min(rows.saturating_sub(1))
    });

    // y-axis gutter: width of the widest label (the ceiling value).
    let gutter = format!("{}", y_max.round() as i64).len().max(3) + 1;

    let mut out = String::with_capacity(rows * (gutter + n) + 64);

    if let Some(t) = b.title {
        out.push_str(&" ".repeat(gutter + 1));
        out.push_str(t);
        out.push('\n');
    }

    // Label roughly four evenly-spaced gridlines (including the top).
    let label_step = rows.div_ceil(4).max(1);

    for r in (0..rows).rev() {
        // y-axis label at the boundary at the TOP of this row.
        let show_label = (rows - 1 - r) % label_step == 0;
        if show_label {
            let val = y_min + ((r + 1) as f32 / rows as f32) * range;
            out.push_str(&format!("{:>width$} ", val.round() as i64, width = gutter));
        } else {
            out.push_str(&" ".repeat(gutter + 1));
        }

        let is_med = med_row == Some(r);
        for c in 0..n {
            let e = cell_eighths(total_eighths[c], r);
            if e > 0 {
                let g = BLOCKS[e as usize];
                if b.color && b.spike[c] {
                    out.push_str(C_RED);
                    out.push(g);
                    out.push_str(C_RESET);
                } else {
                    out.push(g);
                }
            } else if is_med {
                // Median baseline shows through where no bar reaches.
                if b.color {
                    out.push_str(C_DIM);
                    out.push('─');
                    out.push_str(C_RESET);
                } else {
                    out.push('─');
                }
            } else {
                out.push(' ');
            }
        }
        out.push('\n');
    }

    // x-axis rule + tick labels.
    out.push_str(&" ".repeat(gutter + 1));
    out.push_str(&"─".repeat(n));
    out.push('\n');
    if !b.x_ticks.is_empty() {
        out.push_str(&render_xticks(b.x_ticks, gutter + 1, n));
    }

    // Baseline legend line, matching the mockup's "median (NNN)".
    if let Some(m) = b.median {
        out.push_str(&format!(
            "{}{} ({})\n",
            " ".repeat(gutter + 1),
            b.baseline_label,
            m.round() as i64
        ));
    }

    out
}

/// Lay tick labels along the x-axis under their columns, left-anchored at
/// each tick's column and skipping any that would overlap the previous one.
fn render_xticks(ticks: &[XTick], lead: usize, width: usize) -> String {
    let mut line = vec![b' '; lead + width];
    // Place ticks in priority order — first, last, then inner — so the span
    // endpoints (which carry the dates on a multi-day axis) survive when
    // wide labels would otherwise collide and drop the right-anchored end.
    let mut order: Vec<usize> = Vec::new();
    if !ticks.is_empty() {
        order.push(0);
    }
    if ticks.len() > 1 {
        order.push(ticks.len() - 1);
    }
    order.extend(1..ticks.len().saturating_sub(1));

    let mut placed: Vec<(usize, usize)> = Vec::new();
    for &idx in &order {
        let label = &ticks[idx].label;
        let mut start = lead + ticks[idx].col.min(width.saturating_sub(1));
        // Right-anchor labels that would spill past the canvas edge.
        if start + label.len() > lead + width {
            start = (lead + width).saturating_sub(label.len());
        }
        let end = start + label.len();
        // Skip if it would touch an already-placed label (need a 1-col gap).
        if placed.iter().any(|&(s, e)| start < e + 1 && end + 1 > s) {
            continue;
        }
        for (i, ch) in label.bytes().enumerate() {
            if start + i < line.len() {
                line[start + i] = ch;
            }
        }
        placed.push((start, end));
    }
    let mut s = String::from_utf8(line).unwrap_or_default();
    s.push('\n');
    s
}

/// Bridge: render an eatime-style series `RuneOutput` (categories +
/// anomalies) into a chart `width` columns by `height` rows. `title`
/// overrides the chart heading (the file-drop flow passes the friendly
/// display name instead of the temp-file path in `source`).
pub fn render_series(
    out: &RuneOutput,
    width: usize,
    height: usize,
    color: bool,
    title: Option<&str>,
) -> String {
    if out.categories.is_empty() {
        return "(no series to plot)\n".to_string();
    }
    // eanet is a short host *ranking*, not a dense time series: wide labeled
    // bars that tower from zero over the `typical` host, not a downsampled
    // variation band. Route it through its own builder.
    if out.rune == "eanet" {
        return super::plot_fanout::render_fanout(out, width, height, color, title);
    }
    let counts: Vec<f32> = out.categories.iter().map(|c| c.count as f32).collect();
    let n_src = counts.len();
    let cols = width.max(1).min(n_src);

    // Per-column bar heights (peak in column) via the SIMD kernel when
    // downsampling; 1:1 passthrough when the series already fits.
    let heights = columnize(&counts, cols);

    // A column is a spike if any source bucket inside its range was flagged.
    let spike_src = spike_flags(out);
    let spike: Vec<bool> = (0..cols)
        .map(|c| {
            let lo = (c * n_src) / cols;
            let hi = ((c + 1) * n_src) / cols;
            spike_src[lo..hi].iter().any(|&f| f)
        })
        .collect();

    // Median baseline: the value the spikes were scored against, or the
    // robust median of the counts when nothing was flagged.
    let median = out
        .anomalies
        .first()
        .map(|a| a.baseline as f32)
        .or_else(|| Some(robust_median(&counts)));

    let ticks = x_ticks(out, cols, n_src);
    let title = title.or_else(|| out.source.as_ref().map(|s| s.path.as_str()));

    let bars = Bars {
        title,
        heights: &heights,
        spike: &spike,
        median,
        x_ticks: &ticks,
        height_rows: height,
        color,
        zero_based: false,
        baseline_label: "median",
    };
    render(&bars)
}

/// True when a rune's `RuneOutput` carries a chartable bar series in
/// `categories[]`. eatime charts a chronological series (ISO-instant labels
/// carry a 'T'); eanet charts its source fan-out ranking (one bar per host).
/// Other runes don't chart. Shared by the REPL/web (`chart_for`) and the
/// report (`svg_chart`) gates so they stay in sync.
pub fn is_chartable(out: &RuneOutput) -> bool {
    if out.categories.len() < 2 {
        return false;
    }
    match out.rune.as_str() {
        "eatime" => out.categories.first().is_some_and(|c| c.name.contains('T')),
        "eanet" => true,
        _ => false,
    }
}

/// Downsample `counts` to `cols` peak-per-column heights. Uses `col_reduce`
/// (returns the max envelope) when there is something to reduce; otherwise
/// the series already fits and is returned as-is.
pub(super) fn columnize(counts: &[f32], cols: usize) -> Vec<f32> {
    if counts.len() <= cols {
        return counts.to_vec();
    }
    let mut mins = vec![0f32; cols];
    let mut maxs = vec![0f32; cols];
    let mut means = vec![0f32; cols];
    unsafe {
        ffi::col_reduce(
            counts.as_ptr(),
            counts.len() as i32,
            cols as i32,
            mins.as_mut_ptr(),
            maxs.as_mut_ptr(),
            means.as_mut_ptr(),
        );
    }
    maxs
}

/// One bool per source category: true if that bucket was flagged as a spike.
pub(super) fn spike_flags(out: &RuneOutput) -> Vec<bool> {
    out.categories
        .iter()
        .map(|c| out.anomalies.iter().any(|a| a.bucket == c.name))
        .collect()
}

pub(super) fn robust_median(counts: &[f32]) -> f32 {
    if counts.is_empty() {
        return 0.0;
    }
    let mut v = counts.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Four evenly-spaced x-axis ticks from the source category names,
/// shortened to keep the axis readable.
pub(super) fn x_ticks(out: &RuneOutput, cols: usize, n_src: usize) -> Vec<XTick> {
    // A span crossing midnight needs the date in labels, else HH:MM-only
    // ticks read backwards (18:30 → 04:30). Detect it from first vs last
    // ISO date prefix.
    let multi_day = match (out.categories.first(), out.categories.last()) {
        (Some(a), Some(b)) => date_prefix(&a.name) != date_prefix(&b.name),
        _ => false,
    };
    // Multi-day labels are wider ("MM-DD HH:MM"), so use fewer of them —
    // otherwise the right-anchored end tick collides and gets dropped,
    // leaving both endpoints showing the same date.
    let n_ticks = (if multi_day { 3 } else { 4 }).min(cols);
    if n_ticks == 0 {
        return Vec::new();
    }
    (0..n_ticks)
        .map(|i| {
            let col = if n_ticks == 1 {
                0
            } else {
                i * (cols - 1) / (n_ticks - 1)
            };
            let src = (col * n_src) / cols;
            let label = short_label(&out.categories[src.min(n_src - 1)].name, multi_day);
            XTick { col, label }
        })
        .collect()
}

/// The `YYYY-MM-DD` date portion of an ISO instant, or the whole name.
fn date_prefix(name: &str) -> &str {
    match name.find('T') {
        Some(t) => &name[..t],
        None => name,
    }
}

/// Trim a category label to something axis-sized. ISO instants
/// (`2024-07-13T13:00:00`) collapse to `HH:MM`, or `MM-DD HH:MM` when the
/// span crosses days; everything else is truncated to 8 chars.
fn short_label(name: &str, multi_day: bool) -> String {
    if let Some(t) = name.find('T') {
        let time = name[t + 1..].get(..5).unwrap_or(&name[t + 1..]);
        if multi_day {
            // MM-DD from the YYYY-MM-DD prefix (chars 5..10).
            let mmdd = name.get(5..10).unwrap_or(&name[..t]);
            return format!("{mmdd} {time}");
        }
        return time.to_string();
    }
    name.chars().take(8).collect()
}
