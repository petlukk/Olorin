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

#[test]
fn csv_scan_finds_delimiters() {
    olorin::kernels::ffi::init().unwrap();
    let input = b"a,b,c\nd,e,f\n";
    let mut commas  = vec![0i32; input.len()];
    let mut nlines  = vec![0i32; input.len()];
    let mut n_comma = 0i32;
    let mut n_nline = 0i32;
    unsafe {
        olorin::kernels::ffi::csv_scan(
            input.as_ptr(), input.len() as i32,
            commas.as_mut_ptr(), nlines.as_mut_ptr(),
            &mut n_comma, &mut n_nline,
        );
    }
    assert_eq!(n_comma, 4);
    assert_eq!(n_nline, 2);
    assert_eq!(&commas[..4], &[1, 3, 7, 9]);
    assert_eq!(&nlines[..2], &[5, 11]);
}

#[test]
fn csv_scan_empty_input() {
    olorin::kernels::ffi::init().unwrap();
    let input: &[u8] = b"";
    let mut commas  = vec![0i32; 1];
    let mut nlines  = vec![0i32; 1];
    let mut n_comma = 0i32;
    let mut n_nline = 0i32;
    unsafe {
        olorin::kernels::ffi::csv_scan(
            input.as_ptr(), 0,
            commas.as_mut_ptr(), nlines.as_mut_ptr(),
            &mut n_comma, &mut n_nline,
        );
    }
    assert_eq!(n_comma, 0);
    assert_eq!(n_nline, 0);
}

#[test]
fn csv_scan_no_final_newline() {
    olorin::kernels::ffi::init().unwrap();
    let input = b"a,b,c";
    let mut commas  = vec![0i32; input.len()];
    let mut nlines  = vec![0i32; input.len()];
    let mut n_comma = 0i32;
    let mut n_nline = 0i32;
    unsafe {
        olorin::kernels::ffi::csv_scan(
            input.as_ptr(), input.len() as i32,
            commas.as_mut_ptr(), nlines.as_mut_ptr(),
            &mut n_comma, &mut n_nline,
        );
    }
    assert_eq!(n_comma, 2);
    assert_eq!(n_nline, 0);
    assert_eq!(&commas[..2], &[1, 3]);
}

#[test]
fn f32_stats_basic() {
    olorin::kernels::ffi::init().unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut count = 0i32;
    let mut sum   = 0.0f32;
    let mut min_v = 0.0f32;
    let mut max_v = 0.0f32;
    unsafe {
        olorin::kernels::ffi::f32_stats(
            data.as_ptr(), data.len() as i32,
            &mut count, &mut sum, &mut min_v, &mut max_v,
        );
    }
    assert_eq!(count, 5);
    assert!((sum - 15.0).abs() < 1e-5);
    assert!((min_v - 1.0).abs() < 1e-5);
    assert!((max_v - 5.0).abs() < 1e-5);
}

#[test]
fn f32_stats_empty_safe() {
    olorin::kernels::ffi::init().unwrap();
    let data: Vec<f32> = vec![];
    let mut count = 0i32;
    let mut sum   = 0.0f32;
    let mut min_v = 0.0f32;
    let mut max_v = 0.0f32;
    unsafe {
        olorin::kernels::ffi::f32_stats(
            data.as_ptr(), 0,
            &mut count, &mut sum, &mut min_v, &mut max_v,
        );
    }
    assert_eq!(count, 0);
    assert_eq!(sum, 0.0);
    assert_eq!(min_v, 0.0);
    assert_eq!(max_v, 0.0);
}

#[test]
fn f32_stats_single_element() {
    olorin::kernels::ffi::init().unwrap();
    let data: Vec<f32> = vec![42.5];
    let mut count = 0i32;
    let mut sum   = 0.0f32;
    let mut min_v = 0.0f32;
    let mut max_v = 0.0f32;
    unsafe {
        olorin::kernels::ffi::f32_stats(
            data.as_ptr(), 1,
            &mut count, &mut sum, &mut min_v, &mut max_v,
        );
    }
    assert_eq!(count, 1);
    assert!((sum - 42.5).abs() < 1e-5);
    assert!((min_v - 42.5).abs() < 1e-5);
    assert!((max_v - 42.5).abs() < 1e-5);
}

#[test]
fn f32_stats_negatives() {
    // Real bank data has refunds/credits — negatives must be handled.
    olorin::kernels::ffi::init().unwrap();
    let data: Vec<f32> = vec![-10.0, 5.0, -3.0, 2.0];
    let mut count = 0i32;
    let mut sum   = 0.0f32;
    let mut min_v = 0.0f32;
    let mut max_v = 0.0f32;
    unsafe {
        olorin::kernels::ffi::f32_stats(
            data.as_ptr(), data.len() as i32,
            &mut count, &mut sum, &mut min_v, &mut max_v,
        );
    }
    assert_eq!(count, 4);
    assert!((sum - (-6.0)).abs() < 1e-5);
    assert!((min_v - (-10.0)).abs() < 1e-5);
    assert!((max_v - 5.0).abs() < 1e-5);
}

#[test]
fn eacrunch_runs_on_tiny_fixture() {
    olorin::kernels::ffi::init().unwrap();
    let path = std::env::current_dir().unwrap()
        .join("tests/fixtures/runes/tiny.csv");
    let args = format!("{}", path.display());
    let result = olorin::runes::run_rune("eacrunch", &args)
        .expect("eacrunch should exist");
    assert!(result.success, "rune failed: {}", result.answer);
    assert!(result.timing_us > 0);
    assert!(
        result.answer.contains("10") || result.answer.contains("rows"),
        "expected row count in answer: {}", result.answer,
    );
    assert!(result.answer.contains("amount"),
        "expected amount column mentioned: {}", result.answer);
}
