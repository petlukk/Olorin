//! Adversarial-input contract fuzz: every rune's `--json` mode must
//! either succeed with a parseable RuneOutput, or fail cleanly with
//! `success:false` + a non-empty `error`. No panics. No empty stdout.
//! No crashes.
//!
//! The v1 contract pins this — the rune's own error path emits JSON so
//! downstream chained runes (eadiff, future ones) read structured
//! failure instead of choking on free-form text. This test exercises
//! that contract against pathological inputs each rune's parser /
//! kernel could plausibly trip on.
//!
//! Per-rune four-case matrix:
//!   - empty file
//!   - structural malformation (rune-specific)
//!   - embedded NULs / control bytes
//!   - format-specific edge shape (huge single line, magic-only, etc.)
//!
//! Path-allowlist + symlink defenses are NOT re-tested here — see
//! `runes_eatime.rs::eatime_rejects_outside_allowlist` and friends.

use olorin::runes::output::RuneOutput;
use std::io::Write;
use std::process::{Command, Stdio};

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

struct Case {
    /// Used in panic messages so failures point at the exact case.
    name:  &'static str,
    bytes: Vec<u8>,
    /// What must the rune do? Either it accepts the input (success=true)
    /// or rejects it (success=false, error non-empty). Setting this to
    /// `None` means "either is fine, we just don't want a panic."
    expect_success: Option<bool>,
}

fn write_tmp(name: &str, bytes: &[u8]) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(bytes).expect("tmp write");
    path
}

fn run_olorin_strict(script: &str) -> (bool, String, String) {
    let mut child = Command::new(OLORIN)
        .arg("--strict")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn olorin");
    child.stdin
        .as_mut()
        .expect("stdin")
        .write_all(script.as_bytes())
        .expect("write stdin");
    let out = child.wait_with_output().expect("wait olorin");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn extract_rune_json(stdout: &str) -> Option<&str> {
    let start = stdout.find("{\"schema_version\":")?;
    let rest = &stdout[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Run one adversarial case end-to-end. Asserts every invariant of the
/// v1 contract. Panic message identifies the offending case AND the
/// concrete failure so triage doesn't require rerunning under a debugger.
fn run_case(rune: &str, args_template: &str, case: &Case) {
    // Stable per-case path so simultaneous test runs don't clobber.
    let path = write_tmp(&format!("olorin_adv_{rune}_{}.bin", case.name), &case.bytes);
    let invocation = args_template.replace("{path}", &path);
    let script = format!("/rune {rune} --json {invocation}\n/quit\n");

    let (exit_ok, stdout, stderr) = run_olorin_strict(&script);
    let _ = std::fs::remove_file(&path);

    assert!(
        exit_ok,
        "[{rune}/{}] olorin crashed or exited non-zero\nstderr: {stderr}\nstdout: {stdout}",
        case.name
    );

    let json = extract_rune_json(&stdout).unwrap_or_else(|| {
        panic!(
            "[{rune}/{}] no RuneOutput JSON in stdout\nstderr: {stderr}\nstdout: {stdout}",
            case.name
        )
    });

    let out = RuneOutput::from_json(json.as_bytes()).unwrap_or_else(|e| {
        panic!(
            "[{rune}/{}] RuneOutput not parseable: {e}\njson: {json}",
            case.name
        )
    });

    assert_eq!(
        out.rune, rune,
        "[{rune}/{}] rune-name field mismatch: got {}",
        case.name, out.rune
    );

    if !out.success {
        assert!(
            out.error.as_ref().is_some_and(|e| !e.is_empty()),
            "[{rune}/{}] success=false but error is missing or empty: {json}",
            case.name
        );
    }

    if let Some(want) = case.expect_success {
        assert_eq!(
            out.success, want,
            "[{rune}/{}] expected success={want}, got success={}, error={:?}",
            case.name, out.success, out.error
        );
    }
}

// ─── eacrunch (CSV) ────────────────────────────────────────────────────────

#[test]
fn adversarial_eacrunch() {
    let huge_line = {
        let mut v = b"a,b,c\n".to_vec();
        v.extend(std::iter::repeat(b'x').take(1024 * 1024));
        v
    };
    let cases = vec![
        Case {
            name: "empty",
            bytes: Vec::new(),
            expect_success: None,
        },
        Case {
            name: "header_only_no_data",
            bytes: b"col_a,col_b,col_c\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "nul_bytes_midfile",
            bytes: b"a,b,c\n1,2,3\n\x00\x00\x00\n4,5,6\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "huge_single_line",
            bytes: huge_line,
            expect_success: None,
        },
    ];
    for c in &cases { run_case("eacrunch", "{path}", c); }
}

// ─── eajson (JSONL) ────────────────────────────────────────────────────────

#[test]
fn adversarial_eajson() {
    let cases = vec![
        Case {
            name: "empty",
            bytes: Vec::new(),
            expect_success: None,
        },
        Case {
            name: "truncated_no_close_brace",
            bytes: b"{\"a\":1,\"b\":2".to_vec(),
            expect_success: None,
        },
        Case {
            name: "mixed_valid_invalid_lines",
            bytes: b"{\"x\":1}\nnot a json line\n{\"x\":2}\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "nul_bytes_between_lines",
            bytes: b"{\"a\":1}\n\x00\x00\x00\n{\"b\":2}\n".to_vec(),
            expect_success: None,
        },
    ];
    for c in &cases { run_case("eajson", "{path}", c); }
}

// ─── eaparquet ─────────────────────────────────────────────────────────────

#[test]
fn adversarial_eaparquet() {
    let cases = vec![
        Case {
            name: "empty",
            bytes: Vec::new(),
            expect_success: Some(false),
        },
        Case {
            name: "wrong_magic",
            bytes: b"NOTPARQ1and some bytes after".to_vec(),
            expect_success: Some(false),
        },
        Case {
            name: "magic_only_no_footer",
            bytes: b"PAR1".to_vec(),
            expect_success: Some(false),
        },
        Case {
            name: "all_0xff_garbage",
            bytes: vec![0xff; 8192],
            expect_success: Some(false),
        },
    ];
    for c in &cases { run_case("eaparquet", "{path}", c); }
}

// ─── ealog ─────────────────────────────────────────────────────────────────

#[test]
fn adversarial_ealog() {
    let mut huge_line = b"INFO ".to_vec();
    huge_line.extend(std::iter::repeat(b'A').take(1024 * 1024));
    let cases = vec![
        Case {
            name: "empty",
            bytes: Vec::new(),
            expect_success: None,
        },
        Case {
            name: "no_severity_keywords",
            bytes: b"hello\nworld\nthe quick brown fox\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "nul_bytes_interspersed",
            bytes: b"INFO ok\n\x00\x00\nERROR bad\nFATAL worst\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "single_huge_line_with_severity",
            bytes: huge_line,
            expect_success: None,
        },
    ];
    for c in &cases { run_case("ealog", "{path}", c); }
}

// ─── eatime ────────────────────────────────────────────────────────────────

#[test]
fn adversarial_eatime() {
    let cases = vec![
        Case {
            name: "empty",
            bytes: Vec::new(),
            expect_success: None,
        },
        Case {
            name: "no_timestamps",
            bytes: b"hello\nworld\nthe quick brown fox\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "out_of_range_hour_minute",
            bytes: b"2026-13-99T99:99:99 bad\n2026-05-11T25:00:00 also bad\n".to_vec(),
            expect_success: None,
        },
        Case {
            name: "nul_bytes_between_timestamps",
            bytes: b"2026-05-11T06:00:00 a\n\x00\x00\x00\n2026-05-11T07:00:00 b\n".to_vec(),
            expect_success: None,
        },
    ];
    for c in &cases { run_case("eatime", "{path}", c); }
}

// ─── eadiff (two-input) ────────────────────────────────────────────────────

/// eadiff has two file args, both checked. Build a known-valid
/// RuneOutput JSON as the "good" side, pair it with various bad sides.
fn canonical_runeoutput_bytes() -> Vec<u8> {
    use olorin::runes::output::{Category, RuneOutput, Source, Totals};
    let mut o = RuneOutput::new("eatime", 1);
    o.source = Some(Source {
        path:   "canon".into(),
        bytes:  100,
        format: "plaintext".into(),
    });
    o.totals = Totals { rows: 1, scan_us: 0 };
    o.categories = vec![Category { name: "06:00".into(), count: 1 }];
    o.to_json().into_bytes()
}

#[test]
fn adversarial_eadiff() {
    let good_path = write_tmp("olorin_adv_eadiff_good.json", &canonical_runeoutput_bytes());

    let wrong_schema = br#"{"schema_version":99,"rune":"eatime","rune_version":1,"success":true,"totals":{"rows":0,"scan_us":0},"fields":[],"categories":[],"samples":[]}"#.to_vec();

    let cases = vec![
        Case {
            name: "a_is_not_json",
            bytes: b"this is not json at all\n".to_vec(),
            expect_success: Some(false),
        },
        Case {
            name: "a_wrong_schema_version",
            bytes: wrong_schema,
            expect_success: Some(false),
        },
        Case {
            name: "a_is_truncated_json",
            bytes: br#"{"schema_version":1,"rune":"eatime","#.to_vec(),
            expect_success: Some(false),
        },
        Case {
            name: "a_is_empty",
            bytes: Vec::new(),
            expect_success: Some(false),
        },
    ];

    for c in &cases {
        let path_a = write_tmp(&format!("olorin_adv_eadiff_{}.json", c.name), &c.bytes);
        let script = format!(
            "/rune eadiff --json {path_a} {good_path}\n/quit\n"
        );
        let (exit_ok, stdout, stderr) = run_olorin_strict(&script);
        let _ = std::fs::remove_file(&path_a);

        assert!(exit_ok, "[eadiff/{}] olorin crashed\nstderr: {stderr}", c.name);
        let json = extract_rune_json(&stdout).unwrap_or_else(|| {
            panic!("[eadiff/{}] no JSON in stdout\nstdout: {stdout}", c.name)
        });
        let out = RuneOutput::from_json(json.as_bytes()).unwrap_or_else(|e| {
            panic!("[eadiff/{}] JSON not parseable: {e}\njson: {json}", c.name)
        });
        assert_eq!(out.rune, "eadiff", "[eadiff/{}] wrong rune name", c.name);
        assert_eq!(
            out.success, false,
            "[eadiff/{}] expected failure, got success=true; out={json}", c.name
        );
        assert!(
            out.error.as_ref().is_some_and(|e| !e.is_empty()),
            "[eadiff/{}] error field missing or empty: {json}", c.name
        );
    }

    let _ = std::fs::remove_file(&good_path);
}
