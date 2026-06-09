//! Rung-6 (REPL surface) verify: `/rune eatime --bucket series <log>` driven
//! through the real non-streaming dispatch path renders a COLOR block-bar
//! chart above the text body — and a `--json` run does not (clean JSONL).
//! Also confirms the web streaming path emits the chart (color off).

use olorin::core::router::{DispatchContext, StreamEvent};
use std::sync::mpsc;

/// A timestamped log with a clear burst, so `eatime --bucket series` finds a
/// chronological series with a spike.
fn write_series_log(name: &str) -> String {
    let mut log = String::new();
    for i in 0..60 {
        log.push_str(&format!(
            "2026-06-01T08:{:02}:00+00:00 INFO svc handled id={}\n",
            i % 60,
            1000 + i
        ));
    }
    // Burst in the 08:10 minute → a spike bucket.
    for j in 0..80 {
        log.push_str(&format!(
            "2026-06-01T08:10:{:02}+00:00 ERROR svc timeout retry={j}\n",
            j % 60
        ));
    }
    let path = format!("/tmp/{name}");
    std::fs::write(&path, &log).unwrap();
    path
}

/// Block-glyph presence proves a chart was rendered into the output.
fn has_chart(s: &str) -> bool {
    s.contains('█') || s.contains('▇') || s.contains('▆') || s.contains('▁')
}

#[test]
fn repl_rune_eatime_series_renders_color_chart() {
    let _ = olorin::kernels::ffi::init();
    let path = write_series_log("plot_repl_series.log");

    let mut ctx = DispatchContext::new_no_engine(None);
    let resp = ctx.dispatch(&format!("/rune eatime --bucket series {path}"));

    assert!(has_chart(&resp.text), "REPL should render a chart:\n{}", resp.text);
    // Color is ON in the REPL: spike columns are wrapped in the red SGR.
    assert!(
        resp.text.contains("\x1b[31m"),
        "REPL chart should be colorized:\n{:?}",
        resp.text
    );
    // The text body still follows the chart (series summary).
    assert!(
        resp.text.contains("buckets:") || resp.text.contains("peak bucket"),
        "text body present below the chart:\n{}",
        resp.text
    );
}

#[test]
fn repl_rune_json_has_no_chart() {
    let _ = olorin::kernels::ffi::init();
    let path = write_series_log("plot_repl_json.log");

    let mut ctx = DispatchContext::new_no_engine(None);
    let resp = ctx.dispatch(&format!("/rune eatime --bucket series --json {path}"));

    // --json must stay clean JSONL: no chart, no ANSI escapes.
    assert!(!has_chart(&resp.text), "json output must not be charted:\n{}", resp.text);
    assert!(!resp.text.contains('\x1b'), "json output must have no color:\n{}", resp.text);
    assert!(resp.text.trim_start().starts_with('{'), "still JSONL:\n{}", resp.text);
}

#[test]
fn web_stream_eatime_series_renders_chart_no_color() {
    let _ = olorin::kernels::ffi::init();
    let path = write_series_log("plot_web_series.log");

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_file_streaming("app.log", &path, &tx);
    drop(tx);

    let mut text = String::new();
    while let Ok(ev) = rx.try_recv() {
        if let StreamEvent::Token(t) = ev {
            text.push_str(&t);
        }
    }

    assert!(has_chart(&text), "web stream should include the chart:\n{text}");
    // Web chat bubble is not ANSI-aware → color must be OFF.
    assert!(!text.contains('\x1b'), "web chart must have no ANSI color:\n{text:?}");
    // Friendly display name used as the title, not the temp path.
    assert!(text.contains("app.log"), "title is the display name:\n{text}");
}
