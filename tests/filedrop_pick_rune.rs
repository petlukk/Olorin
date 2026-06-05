//! Step 1 of the file-drop analyst: deterministic rune selection.
//!
//! `pick_rune_name` is pure (filename + content sniff → rune name); these
//! tests pin the extension routing and the timestamp-sniff split between
//! `eatime` and `ealog`. `pick_rune` additionally resolves the name against
//! the live RUNES registry.

use olorin::runes::select::{pick_rune, pick_rune_name};

const ISO_LOG: &[u8] =
    b"2026-06-01T08:00:00+00:00 INFO svcA request handled id=1000\n\
      2026-06-01T08:01:00+00:00 INFO svcA request handled id=1001\n";

const CLF_LOG: &[u8] =
    b"127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] \"GET /a HTTP/1.0\" 200 2326\n";

const SEVERITY_ONLY_LOG: &[u8] =
    b"INFO starting up\nWARN cache miss\nERROR upstream timeout\nDEBUG retry\n";

#[test]
fn extension_routing() {
    assert_eq!(pick_rune_name("sales.csv", b""), Some("eacrunch"));
    assert_eq!(pick_rune_name("events.jsonl", b""), Some("eajson"));
    assert_eq!(pick_rune_name("events.ndjson", b""), Some("eajson"));
    assert_eq!(pick_rune_name("blob.json", b""), Some("eajson"));
    assert_eq!(pick_rune_name("data.parquet", b""), Some("eaparquet"));
}

#[test]
fn extension_is_case_insensitive_and_path_stripped() {
    assert_eq!(pick_rune_name("REPORT.CSV", b""), Some("eacrunch"));
    assert_eq!(pick_rune_name("/tmp/drop/Data.Parquet", b""), Some("eaparquet"));
}

#[test]
fn log_with_timestamps_picks_eatime() {
    assert_eq!(pick_rune_name("app.log", ISO_LOG), Some("eatime"));
    assert_eq!(pick_rune_name("access.log", CLF_LOG), Some("eatime"));
    // No extension but timestamped content still routes to eatime.
    assert_eq!(pick_rune_name("syslog", ISO_LOG), Some("eatime"));
}

#[test]
fn log_without_timestamps_picks_ealog() {
    assert_eq!(pick_rune_name("app.log", SEVERITY_ONLY_LOG), Some("ealog"));
    assert_eq!(pick_rune_name("notes.txt", b"just some prose, no dates"), Some("ealog"));
}

#[test]
fn unknown_extension_returns_none() {
    assert_eq!(pick_rune_name("image.png", b"\x89PNG"), None);
    assert_eq!(pick_rune_name("archive.tar.gz", b""), None);
}

#[test]
fn pick_rune_resolves_against_registry() {
    // The chosen name must actually exist in the live RUNES registry.
    let r = pick_rune("sales.csv", b"a,b\n1,2\n").expect("csv -> a real rune");
    assert_eq!(r.name(), "eacrunch");

    let r = pick_rune("app.log", ISO_LOG).expect("timestamped log -> a real rune");
    assert_eq!(r.name(), "eatime");

    let r = pick_rune("app.log", SEVERITY_ONLY_LOG).expect("severity log -> a real rune");
    assert_eq!(r.name(), "ealog");

    assert!(pick_rune("photo.png", b"").is_none());
}
