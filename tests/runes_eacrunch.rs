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
