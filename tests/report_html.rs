//! HTML report renderer: byte-golden over a fixed in-process RuneOutput
//! (fully deterministic — no scrubbing), structural assertions over the
//! real incident fixtures, HTML-escaping of hostile file-derived strings,
//! and a spawn test of the `olorin report` CLI.

use olorin::runes::correlation::Correlation;
use olorin::runes::incident::{Anchor, Incident, Step};
use olorin::runes::output::{Anomaly, Category, RuneOutput, Source, Totals};
use olorin::runes::report::{build_report, render_report, svg_chart, ReportSection};
use std::path::PathBuf;
use std::process::Command;

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

/// Stage an incident fixture into /tmp (rune path allowlist), tagged per
/// test so parallel tests can't race each other's cleanup.
fn stage(tag: &str, name: &str) -> String {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incident")
        .join(name);
    let dst = format!("/tmp/olorin_report_{tag}_{name}");
    std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("stage {name}: {e}"));
    dst
}

/// A fixed two-section report with one correlation — every field pinned,
/// so the rendered bytes are identical on every machine and arch.
fn fixed_sections() -> (Vec<ReportSection>, RuneOutput) {
    let mut log = RuneOutput::new("eatime", 1);
    log.source = Some(Source {
        path: "/tmp/syslog.log".into(), bytes: 11_580, format: "iso8601".into(),
    });
    log.totals = Totals { rows: 226, scan_us: 10 };
    log.categories = (0..6).map(|i| Category {
        name:  format!("2026-06-11T0{}:00:00", 2 + i / 2),
        count: if i == 1 { 20 } else { 5 },
    }).collect();
    log.anomalies = vec![Anomaly {
        bucket: "2026-06-11T02:20:00".into(),
        count: 20, baseline: 5.0, ratio: 4.0, score: 11.2,
    }];

    let mut csv = RuneOutput::new("eacrunch", 1);
    csv.source = Some(Source {
        path: "/tmp/deploys.csv".into(), bytes: 120, format: "csv".into(),
    });
    csv.totals = Totals { rows: 3, scan_us: 864 };

    let mut corr = RuneOutput::new("eacorrelate", 1);
    corr.totals = Totals { rows: 229, scan_us: 38 };
    corr.correlations = vec![Correlation {
        stream_a: "syslog.log (errors)".into(),
        stream_b: "deploys.csv".into(),
        lag_seconds: 240, score: 0.9375,
        peak_bucket: "2026-06-11T02:24:00".into(),
        events_a: 45, events_b: 3, width_seconds: 30,
    }];
    corr.incident = Some(Incident {
        anchor: Anchor {
            kind: "trigger".into(),
            stream: "deploys.csv".into(),
            time: "2026-06-11T02:20:00".into(),
        },
        steps: vec![Step {
            stream: "syslog.log (errors)".into(),
            lag_seconds: 240,
            direction: "increase".into(),
            score: 0.9375,
            kind: "correlated".into(),
        }],
        confidence: 0.9375,
    });

    let sections = vec![
        ReportSection { display: "syslog.log".into(),  rune: "eatime".into(),   output: log },
        ReportSection { display: "deploys.csv".into(), rune: "eacrunch".into(), output: csv },
    ];
    (sections, corr)
}

#[test]
fn fixed_report_matches_golden() {
    olorin::kernels::ffi::init().unwrap(); // svg_chart uses col_reduce
    let (sections, corr) = fixed_sections();
    let html = render_report(&sections, Some(&corr), "vTEST");

    let golden = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runes/golden/report_fixed.html");
    if std::env::var_os("BLESS").is_some_and(|v| !v.is_empty() && v != "0") {
        std::fs::write(&golden, &html).expect("write golden");
        eprintln!("BLESS: wrote {}", golden.display());
        return;
    }
    let expected = std::fs::read_to_string(&golden).unwrap_or_else(|e| {
        panic!("missing golden {}: {e} — BLESS=1 to capture", golden.display())
    });
    assert_eq!(expected, html, "report bytes drifted from golden");
}

#[test]
fn eanet_report_uses_findings_prose() {
    olorin::kernels::ffi::init().unwrap(); // svg_chart uses col_reduce
    let mut net = RuneOutput::new("eanet", 1);
    net.source = Some(Source {
        path: "/tmp/capture.pcap".into(), bytes: 22_084, format: "pcap".into(),
    });
    net.totals = Totals { rows: 80, scan_us: 138 };
    net.categories = vec![
        Category { name: "10.0.0.66".into(), count: 50 },
        Category { name: "192.168.0.2".into(), count: 1 },
    ];
    net.anomalies = vec![
        Anomaly { bucket: "10.0.0.66".into(), count: 50, baseline: 1.0, ratio: 50.0, score: 50.0 },
        Anomaly { bucket: "10.0.0.99 -> 203.0.113.7".into(), count: 15_000, baseline: 54.0, ratio: 277.0, score: 277.0 },
    ];
    let sections = vec![ReportSection {
        display: "capture.pcap".into(), rune: "eanet".into(), output: net,
    }];
    let html = render_report(&sections, None, "vTEST");

    // The report renders eanet's anomalies in the rune's own findings prose
    // (human byte units), identical to the chat — not the generic time-spike
    // phrasing meant for eatime.
    assert!(
        html.contains("contacted 50 distinct destinations — likely a horizontal scan"),
        "scan findings prose missing from report"
    );
    assert!(
        html.contains("moved 14.6 KB to a single destination"),
        "talker findings prose with human bytes missing from report"
    );
    assert!(!html.contains("spike at"), "report must not use the generic spike phrasing for eanet");
}

#[test]
fn renders_deterministically() {
    olorin::kernels::ffi::init().unwrap();
    let (sections, corr) = fixed_sections();
    let a = render_report(&sections, Some(&corr), "vTEST");
    let b = render_report(&sections, Some(&corr), "vTEST");
    assert_eq!(a, b);
}

#[test]
fn incident_report_structure() {
    olorin::kernels::ffi::init().unwrap();
    let files: Vec<(String, String)> = ["syslog.log", "deploys.csv", "access.log"]
        .iter().map(|n| (n.to_string(), stage("structure", n))).collect();
    let (html, summary) = build_report(&files, "vTEST").expect("build_report");
    for (_, p) in &files {
        let _ = std::fs::remove_file(p);
    }

    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("Cross-file correlations"), "findings banner missing");
    assert!(
        html.contains("follows <strong>deploys.csv</strong> by +240"),
        "planted lag not in banner"
    );
    assert_eq!(html.matches("<svg").count(), 2, "two series charts expected");
    assert_eq!(html.matches("<section>").count(), 3, "three file sections");
    assert!(html.contains("via eatime") && html.contains("via eacrunch"));
    assert!(!html.contains("<script"), "report must carry no JS");
    assert!(html.contains("vTEST"));
    assert!(summary.contains("eacorrelate"));
}

#[test]
fn hostile_names_are_escaped() {
    olorin::kernels::ffi::init().unwrap();
    let mut out = RuneOutput::new("ealog", 1);
    out.totals = Totals { rows: 1, scan_us: 1 };
    out.categories = vec![Category {
        name:  "<img src=x onerror=alert(1)>".into(),
        count: 1,
    }];
    let sections = vec![ReportSection {
        display: "evil<script>.log\" onload=\"x".into(),
        rune:    "ealog".into(),
        output:  out,
    }];
    let html = render_report(&sections, None, "vTEST");
    assert!(!html.contains("<img"), "category name not escaped");
    assert!(!html.contains("evil<script>"), "display name not escaped");
    assert!(html.contains("&lt;img"), "escaped form expected");
}

#[test]
fn non_series_output_gets_no_chart() {
    olorin::kernels::ffi::init().unwrap();
    let mut out = RuneOutput::new("ealog", 1);
    out.categories = vec![
        Category { name: "ERROR".into(), count: 7 },
        Category { name: "INFO".into(),  count: 99 },
    ];
    assert!(svg_chart(&out).is_none(), "severity ladder must not chart");
}

#[test]
fn cli_writes_report_file() {
    let out_path = "/tmp/olorin_report_cli_test.html";
    let _ = std::fs::remove_file(out_path);
    let log = stage("cli", "syslog.log");
    let csv = stage("cli", "deploys.csv");
    let status = Command::new(OLORIN)
        .args(["report", &log, &csv, "-o", out_path])
        .status()
        .expect("spawn olorin report");
    assert!(status.success(), "CLI exit code");
    let html = std::fs::read_to_string(out_path).expect("report file written");
    assert!(html.contains("Cross-file correlations"));
    let _ = std::fs::remove_file(out_path);
    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn cli_usage_error_without_files() {
    let status = Command::new(OLORIN)
        .args(["report"])
        .status()
        .expect("spawn olorin report");
    assert_eq!(status.code(), Some(2), "usage error exit code");
}
