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
}

/// Round up to a "nice" axis ceiling (1, 2, or 5 × 10^k) so y-labels read
/// cleanly instead of landing on 5873.
fn nice_ceil(x: f32) -> f32 {
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

    // Vertical scale: the tallest bar (or the median line, if higher) sets
    // the ceiling so nothing clips off the top.
    let peak = b
        .heights
        .iter()
        .copied()
        .fold(0.0f32, f32::max)
        .max(b.median.unwrap_or(0.0));
    let y_max = nice_ceil(peak);

    // Pre-compute per-column total eighths once.
    let total_eighths: Vec<i32> = b
        .heights
        .iter()
        .map(|&h| ((h / y_max) * (rows as f32) * 8.0).round() as i32)
        .collect();

    // Row that the median baseline falls in (its dashed line is drawn
    // through the gaps where no bar reaches).
    let med_row: Option<usize> = b.median.map(|m| {
        (((m / y_max) * rows as f32).floor() as usize).min(rows.saturating_sub(1))
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
            let val = ((r + 1) as f32 / rows as f32) * y_max;
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

    // Median legend line, matching the mockup's "median (NNN)".
    if let Some(m) = b.median {
        out.push_str(&format!(
            "{}median ({})\n",
            " ".repeat(gutter + 1),
            m.round() as i64
        ));
    }

    out
}

/// Lay tick labels along the x-axis under their columns, left-anchored at
/// each tick's column and skipping any that would overlap the previous one.
fn render_xticks(ticks: &[XTick], lead: usize, width: usize) -> String {
    let mut line = vec![b' '; lead + width];
    let mut next_free = 0usize;
    for t in ticks {
        let mut start = lead + t.col.min(width.saturating_sub(1));
        // Right-anchor labels that would spill past the canvas edge so the
        // last tick (e.g. under the final column) stays fully on-screen.
        if start + t.label.len() > lead + width {
            start = (lead + width).saturating_sub(t.label.len());
        }
        if start < next_free {
            continue; // would collide with the prior label
        }
        for (i, ch) in t.label.bytes().enumerate() {
            let pos = start + i;
            if pos >= line.len() {
                break;
            }
            line[pos] = ch;
        }
        next_free = start + t.label.len() + 1;
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
    };
    render(&bars)
}

/// Downsample `counts` to `cols` peak-per-column heights. Uses `col_reduce`
/// (returns the max envelope) when there is something to reduce; otherwise
/// the series already fits and is returned as-is.
fn columnize(counts: &[f32], cols: usize) -> Vec<f32> {
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
fn spike_flags(out: &RuneOutput) -> Vec<bool> {
    out.categories
        .iter()
        .map(|c| out.anomalies.iter().any(|a| a.bucket == c.name))
        .collect()
}

fn robust_median(counts: &[f32]) -> f32 {
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
fn x_ticks(out: &RuneOutput, cols: usize, n_src: usize) -> Vec<XTick> {
    let n_ticks = 4.min(cols);
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
            let label = short_label(&out.categories[src.min(n_src - 1)].name);
            XTick { col, label }
        })
        .collect()
}

/// Trim a category label to something axis-sized. ISO instants
/// (`2024-07-13T13:00:00`) collapse to `HH:MM`; everything else is
/// truncated to 8 chars.
fn short_label(name: &str) -> String {
    if let Some(t) = name.find('T') {
        let rest = &name[t + 1..];
        return rest.get(..5).unwrap_or(rest).to_string();
    }
    name.chars().take(8).collect()
}
