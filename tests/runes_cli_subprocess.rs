//! End-to-end CLI tests: spawn the actual `olorin` binary as a
//! subprocess, pipe commands via stdin, and verify the v0.9.4
//! `--json` contract holds end-to-end.
//!
//! The rest of the rune integration tests call `run_rune()`
//! programmatically — that bypasses the REPL chrome, the
//! `wrap_rune_result` path, the `[timing: …]` footer, and the
//! narration-prompt builder. Only a real subprocess test proves the
//! CLI behaves as documented.
//!
//! Uses cargo's `CARGO_BIN_EXE_olorin` env var so the test always
//! uses the same-build binary regardless of debug/release.

use std::io::Write;
use std::process::{Command, Stdio};

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

fn write_tmp(name: &str, bytes: &[u8]) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(bytes).expect("tmp write");
    path
}

/// Spawn `olorin --strict`, pipe `script` to stdin, return combined stdout.
fn run_olorin_strict(script: &str) -> String {
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
    assert!(out.status.success(),
        "olorin exited non-zero: {:?}\nstderr: {}",
        out.status, String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("utf8 stdout")
}

/// Locate the first compact JSON object in `stdout` and return it.
/// The REPL prepends `olorin> ` to the same line as the JSON answer,
/// so the JSON is wherever `{"schema_version":` appears.
fn extract_rune_json(stdout: &str) -> &str {
    let start = stdout.find("{\"schema_version\":")
        .unwrap_or_else(|| panic!("no RuneOutput JSON in stdout:\n{stdout}"));
    let rest = &stdout[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    &rest[..end]
}

#[test]
fn cli_eatime_json_mode_emits_parseable_jsonl() {
    // Three timestamps, two at 06:xx and one at 07:00. The kernel +
    // bucketer + JSON serializer all run inside the spawned binary.
    let log = b"\
2026-05-11T06:00:00 a
2026-05-11T06:30:00 b
2026-05-11T07:00:00 c
";
    let path = write_tmp("olorin_cli_eatime_json.log", log);
    let script = format!("/rune eatime --json {path}\n/quit\n");
    let stdout = run_olorin_strict(&script);

    let json = extract_rune_json(&stdout);
    let out = olorin::runes::output::RuneOutput::from_json(json.as_bytes())
        .unwrap_or_else(|e| panic!("not parseable: {e}\njson={json}"));

    assert_eq!(out.rune, "eatime");
    assert_eq!(out.totals.rows, 3);
    let by_hour: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_hour.get("06:00"), Some(&2));
    assert_eq!(by_hour.get("07:00"), Some(&1));

    // v0.9.4 contract: structured output is NOT wrapped, NOT footered.
    assert!(!stdout.contains("<rune_output"),
        "structured output must not be wrapped in REPL: {stdout}");
    assert!(!stdout.contains("[timing:"),
        "structured output must not carry a timing footer: {stdout}");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn cli_eatime_text_mode_still_has_timing_footer() {
    // Negative test for the v0.9.4 fix: without --json, the legacy
    // text format keeps the [timing: …µs] footer for human users.
    let log = b"2026-05-11T08:00:00 only\n";
    let path = write_tmp("olorin_cli_eatime_text.log", log);
    let script = format!("/rune eatime {path}\n/quit\n");
    let stdout = run_olorin_strict(&script);

    assert!(stdout.contains("hour-of-day:"),
        "text mode should include the hour-of-day label: {stdout}");
    assert!(stdout.contains("[timing:"),
        "text mode keeps the timing footer for humans: {stdout}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cli_eadiff_chain_through_two_subprocess_eatime_runs() {
    // The killer demo, end-to-end via real binary subprocess:
    // 1. Run eatime --json on yesterday.log via subprocess.
    // 2. Capture the JSON to a file.
    // 3. Same for today.log.
    // 4. Run eadiff via subprocess on the two captured files.
    // 5. Verify the +06:00 / -07:00 delta is present.
    let yesterday = b"\
2026-05-10T06:00:00 a
2026-05-10T06:30:00 b
2026-05-10T07:00:00 c
2026-05-10T15:00:00 d
";
    let today = b"\
2026-05-11T06:00:00 a
2026-05-11T06:10:00 b
2026-05-11T06:20:00 c
2026-05-11T06:30:00 d
2026-05-11T15:00:00 e
";
    let y_log = write_tmp("olorin_cli_chain_yest.log", yesterday);
    let t_log = write_tmp("olorin_cli_chain_today.log", today);

    let y_stdout = run_olorin_strict(&format!("/rune eatime --json {y_log}\n/quit\n"));
    let t_stdout = run_olorin_strict(&format!("/rune eatime --json {t_log}\n/quit\n"));
    let y_json = write_tmp("olorin_cli_chain_yest.json",
        extract_rune_json(&y_stdout).as_bytes());
    let t_json = write_tmp("olorin_cli_chain_today.json",
        extract_rune_json(&t_stdout).as_bytes());

    let diff_stdout = run_olorin_strict(&format!(
        "/rune eadiff --json {y_json} {t_json}\n/quit\n"
    ));
    let diff_json = extract_rune_json(&diff_stdout);
    let diff = olorin::runes::output::RuneOutput::from_json(diff_json.as_bytes())
        .expect("eadiff JSON parses");

    let by_name: std::collections::HashMap<&str, u64> =
        diff.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    // yesterday: 06:00=2, 07:00=1, 15:00=1. today: 06:00=4, 07:00=0, 15:00=1.
    // delta: +06:00=2, -07:00=1. 15:00 unchanged → omitted.
    assert_eq!(by_name.get("+06:00"), Some(&2),
        "06:00 should have grown by 2: {:?}", diff.categories);
    assert_eq!(by_name.get("-07:00"), Some(&1),
        "07:00 should have shrunk by 1");

    let _ = std::fs::remove_file(&y_log);
    let _ = std::fs::remove_file(&t_log);
    let _ = std::fs::remove_file(&y_json);
    let _ = std::fs::remove_file(&t_json);
}
