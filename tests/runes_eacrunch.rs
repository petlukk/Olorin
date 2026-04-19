use olorin::runes::{RuneResult, OutputSafety};

#[test]
fn rune_result_defaults() {
    let r = RuneResult {
        answer: "hello".into(),
        details: None,
        success: true,
        timing_us: 42,
    };
    assert!(r.success);
    assert_eq!(r.answer, "hello");
    assert_eq!(r.timing_us, 42);
}

#[test]
fn output_safety_variants() {
    let t = OutputSafety::Trusted;
    let u = OutputSafety::UntrustedQuoted;
    assert!(matches!(t, OutputSafety::Trusted));
    assert!(matches!(u, OutputSafety::UntrustedQuoted));
}

use olorin::runes::common::{resolve_path, PathError};
use std::path::PathBuf;

#[test]
fn resolve_path_accepts_home_relative() {
    let home = std::env::var("HOME").unwrap();
    let p = resolve_path("~/Downloads/foo.csv", &PathBuf::from(&home)).unwrap();
    assert!(p.starts_with(&home));
    assert!(p.ends_with("Downloads/foo.csv"));
}

#[test]
fn resolve_path_rejects_traversal() {
    let home = std::env::var("HOME").unwrap();
    let err = resolve_path("~/../../etc/passwd", &PathBuf::from(&home)).unwrap_err();
    assert!(matches!(err, PathError::OutsideAllowlist));
}

#[test]
fn resolve_path_accepts_tmp() {
    let p = resolve_path("/tmp/test.csv", &PathBuf::from("/home/nobody")).unwrap();
    assert_eq!(p, PathBuf::from("/tmp/test.csv"));
}

#[test]
fn resolve_path_rejects_absolute_outside_allowlist() {
    let home = std::env::var("HOME").unwrap();
    let err = olorin::runes::common::resolve_path(
        "/var/data/foo.csv", &PathBuf::from(&home)
    ).unwrap_err();
    assert!(matches!(err, olorin::runes::common::PathError::OutsideAllowlist));
}

#[test]
fn truncate_answer_handles_utf8_boundary() {
    use olorin::runes::common::{truncate_answer, MAX_ANSWER_BYTES};
    // Build a string just under the limit, then append multi-byte chars that
    // span the cut point. "Å" is 2 bytes in UTF-8.
    let mut s = String::with_capacity(MAX_ANSWER_BYTES + 50);
    // Fill up to MAX_ANSWER_BYTES - 33 so the char-boundary walk has to
    // step back past a multi-byte char.
    while s.len() < MAX_ANSWER_BYTES - 33 { s.push('a'); }
    while s.len() < MAX_ANSWER_BYTES + 20  { s.push('Å'); }
    let out = truncate_answer(&s);              // must not panic
    assert!(out.contains("[...truncated"));
    assert!(
        out.len() <= MAX_ANSWER_BYTES,
        "out.len()={} exceeds MAX_ANSWER_BYTES={}",
        out.len(), MAX_ANSWER_BYTES
    );
}

#[test]
fn truncate_answer_passthrough_under_limit() {
    use olorin::runes::common::truncate_answer;
    let s = "hello world";
    assert_eq!(truncate_answer(s), s);
}

#[test]
fn open_capped_rejects_symlink_outside_allowlist() {
    use std::os::unix::fs::symlink;
    // Create a symlink in /tmp pointing outside the allowlist.
    let link = std::env::temp_dir().join("olorin_runes_symlink_test.link");
    let _ = std::fs::remove_file(&link);
    symlink("/etc/hostname", &link).unwrap();
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let err = olorin::runes::common::open_capped(&link, &home).unwrap_err();
    assert!(
        matches!(err, olorin::runes::common::PathError::OutsideAllowlist),
        "expected OutsideAllowlist, got {err:?}"
    );
    let _ = std::fs::remove_file(&link);
}

#[test]
fn slash_rune_dispatches() {
    olorin::kernels::ffi::init().unwrap();
    let (cmd, arg) = olorin::core::dispatch::match_command(b"/rune eacrunch /tmp/x.csv");
    assert_eq!(cmd, olorin::core::dispatch::CMD_RUNE);
    assert_eq!(std::str::from_utf8(arg).unwrap(), "eacrunch /tmp/x.csv");
}

#[test]
fn slash_rune_unknown_returns_message() {
    olorin::kernels::ffi::init().unwrap();
    // Simulate dispatching to an unknown rune — should give a friendly message.
    let result = olorin::runes::run_rune("nonexistent_rune_xyz", "");
    assert!(result.is_none());
}
