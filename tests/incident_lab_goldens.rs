//! incident-lab golden gate: the committed fixtures under
//! `benchmarks/incident-lab/goldens/` must drive `eacorrelate` to the right
//! conclusion — and, just as importantly, to *no* conclusion on the controls.
//!
//! The deterministic simulator's byte-for-byte reproducibility is checked by
//! `benchmarks/incident-lab/verify_goldens.py`; this test pins the part that
//! matters in CI without a Python dependency: the real rune, run in-process on
//! the frozen logs, finds the bad-deploy cascade and stays silent on the
//! healthy baseline and the clean deploy.

use olorin::runes::run_rune;
use std::path::PathBuf;

const INCIDENT_STREAMS: [&str; 3] = ["deploy.log", "db.log", "access.log"];

/// Copy a scenario's incident streams into /tmp (the rune path guard allows only
/// ~ or /tmp) and return the space-joined arg string for `eacorrelate`.
fn stage(scenario: &str) -> String {
    let goldens = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/incident-lab/goldens")
        .join(scenario);
    let mut paths = Vec::new();
    for stream in INCIDENT_STREAMS {
        let src = goldens.join(stream);
        let dst = std::env::temp_dir().join(format!(
            "olorin_lab_{scenario}_{}_{stream}", std::process::id()
        ));
        std::fs::copy(&src, &dst)
            .unwrap_or_else(|e| panic!("copy {src:?} -> {dst:?}: {e}"));
        paths.push(dst.to_string_lossy().into_owned());
    }
    paths.join(" ")
}

fn correlate(scenario: &str) -> String {
    olorin::kernels::ffi::init().expect("kernel init");
    let r = run_rune("eacorrelate", &stage(scenario))
        .expect("eacorrelate is registered");
    assert!(r.success, "eacorrelate failed on {scenario}: {}", r.answer);
    r.answer
}

#[test]
fn bad_deploy_yields_the_leading_indicator_timeline() {
    let out = correlate("bad-deploy");
    assert!(out.contains("incident timeline"),
        "bad-deploy must produce an incident timeline:\n{out}");
    // The cascade's signature: db-pool errors lead, access-log 5xx follow by a
    // positive lag. The timeline spans two lines, so assert on the whole block.
    let timeline = &out[out.find("incident timeline").unwrap()..];
    assert!(timeline.contains("db.log") && timeline.contains("access.log")
        && timeline.contains("later"),
        "timeline lacks the db->access leading-indicator lag:\n{out}");
}

#[test]
fn quiet_baseline_raises_no_false_incident() {
    let out = correlate("quiet");
    assert!(!out.contains("incident timeline"),
        "FALSE POSITIVE: quiet baseline raised an incident:\n{out}");
}

#[test]
fn clean_deploy_raises_no_false_incident() {
    let out = correlate("good-deploy");
    assert!(!out.contains("incident timeline"),
        "FALSE POSITIVE: a healthy deploy raised an incident:\n{out}");
}
