//! plot_fanout — the eanet bridge into the block-bar renderer.
//!
//! eanet emits a short host *ranking* (top-N source fan-out), not the dense
//! chronological series the eatime bridge in `plot` handles. A ranking wants a
//! different treatment: wide, per-host-labeled bars towering from a zero
//! baseline over a `typical` reference line — so a scanner that contacted 2000
//! destinations reads against the ~5 a normal host does. Routing it through the
//! dense-series downsample instead produced 1px slivers with every bar
//! collapsed to an identical "192.168." truncation. This module builds the
//! `Bars` the shared renderer draws; `plot::render` still does the drawing.

use super::output::RuneOutput;
use super::plot::{render, spike_flags, Bars, Ink, XTick};

/// Render eanet's source fan-out ranking: one wide, host-labeled bar per top
/// host, towering from a zero baseline over the dashed `typical` line (the
/// fan-out median — a normal host's destination count). Bars are widened with
/// a 1-column gap so per-host octet labels fit and read individually.
pub(super) fn render_fanout<'a>(
    out: &'a RuneOutput,
    width: usize,
    height: usize,
    ink: Ink,
    title: Option<&'a str>,
) -> String {
    let n = out.categories.len();
    let labels = fanout_labels(out);
    let spike_src = spike_flags(out);
    // Bar width so every bar plus a 1-col gap between them fits the canvas.
    let bar_w = (width.saturating_sub(n.saturating_sub(1)) / n.max(1)).clamp(1, 10);

    let mut heights: Vec<f32> = Vec::with_capacity(n * (bar_w + 1));
    let mut spike: Vec<bool> = Vec::with_capacity(n * (bar_w + 1));
    let mut ticks: Vec<XTick> = Vec::with_capacity(n);
    for (i, cat) in out.categories.iter().enumerate() {
        ticks.push(XTick { col: heights.len(), label: labels[i].clone() });
        for _ in 0..bar_w {
            heights.push(cat.count as f32);
            spike.push(spike_src[i]);
        }
        if i + 1 < n {
            heights.push(0.0); // gap column between bars
            spike.push(false);
        }
    }

    let title = title.or_else(|| out.source.as_ref().map(|s| s.path.as_str()));
    let bars = Bars {
        title,
        heights: &heights,
        spike: &spike,
        median: fanout_baseline(out),
        x_ticks: &ticks,
        height_rows: height,
        ink,
        zero_based: true,
        baseline_label: "typical",
    };
    render(&bars)
}

/// The `typical` reference for a fan-out chart: the scan anomaly's baseline
/// (the fan-out median — destinations a normal host contacts). None when no
/// scan was flagged, so quiet captures draw bars without a phantom line.
/// The talker anomaly (bucket `src -> dst`, baseline in *bytes*) is skipped —
/// its units don't belong on a destinations axis.
pub(super) fn fanout_baseline(out: &RuneOutput) -> Option<f32> {
    out.anomalies
        .iter()
        .find(|a| !a.bucket.contains("->"))
        .map(|a| a.baseline as f32)
}

/// Host labels for the fan-out axis: drop the leading octets every shown host
/// shares (they carry no signal), so five same-subnet hosts read as ".73 .102
/// .45 …" instead of five identical "192.168." truncations. Always keeps at
/// least the final octet; non-IPv4 names pass through unchanged.
pub(super) fn fanout_labels(out: &RuneOutput) -> Vec<String> {
    let ips: Vec<Vec<&str>> = out
        .categories
        .iter()
        .map(|c| c.name.split('.').collect())
        .collect();
    let four = ips.len() > 1 && ips.iter().all(|o| o.len() == 4);
    let mut common = 0usize; // shared leading octets; never all four
    if four {
        for i in 0..3 {
            if ips[1..].iter().all(|o| o[i] == ips[0][i]) {
                common = i + 1;
            } else {
                break;
            }
        }
    }
    out.categories
        .iter()
        .enumerate()
        .map(|(k, c)| {
            if common > 0 {
                ips[k][common..].join(".")
            } else {
                c.name.clone()
            }
        })
        .collect()
}
