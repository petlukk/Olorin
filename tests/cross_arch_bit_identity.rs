//! Cross-arch bit-identity sweep — every rune's `--json` output must be
//! byte-identical on x86 SSE2 and ARM NEON for the same fixture.
//!
//! Why: the v0.9.x bug class (HashMap-seed non-determinism, debug-format
//! `NotFound`, REPL wrapping `--json`) all showed up as observable-output
//! drift. The 2026-05-12 `sat_sub` codegen miss showed that an Ea kernel
//! can silently produce different numbers across archs. A frozen v1 wire
//! contract needs a permanent guard that captures the bytes themselves,
//! not just structure.
//!
//! Mechanism: one (rune, fixture) case per rune, each spawning the real
//! `olorin` binary, capturing the `--json` line, scrubbing the two
//! machine-variable fields (`totals.scan_us` and `source.path`), then
//! byte-comparing to a checked-in golden file. Any other drift — counts,
//! mins, sample byte offsets, top-N ordering — surfaces as a diff.
//!
//! `BLESS=1 cargo test --test cross_arch_bit_identity` re-captures the
//! goldens. Run on x86 to set the contract; run on Pi to verify it.

use olorin::runes::output::{
    Category, NumericStats, RuneOutput, Source, TextEntry, TextStats, Totals,
};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/runes")
}

fn goldens_dir() -> PathBuf {
    fixtures_dir().join("golden")
}

/// Copy a fixture into /tmp under a stable name so the binary's
/// path-allowlist accepts it AND the resulting `source.path` is the
/// same string on every machine before scrubbing.
fn stage_fixture(name: &str) -> String {
    let src = fixtures_dir().join(name);
    let dst = format!("/tmp/olorin_parity_{name}");
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", src.display(), dst));
    dst
}

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

fn extract_rune_json(stdout: &str) -> &str {
    let start = stdout.find("{\"schema_version\":")
        .unwrap_or_else(|| panic!("no RuneOutput JSON in stdout:\n{stdout}"));
    let rest = &stdout[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    &rest[..end]
}

/// Replace every `"<key>":<value>` segment with `"<key>":<replacement>`,
/// for a numeric or quoted-string value. Scans the raw byte stream so
/// goldens stay byte-comparable for everything else.
fn replace_field(input: &str, key: &str, replacement: &str) -> String {
    let pat = format!("\"{key}\":");
    let mut out = String::with_capacity(input.len());
    let mut cursor = 0;
    while let Some(rel) = input[cursor..].find(&pat) {
        let start = cursor + rel;
        out.push_str(&input[cursor..start]);
        out.push_str(&pat);
        let val_start = start + pat.len();
        let rest = &input[val_start..];
        let first = rest.chars().next()
            .unwrap_or_else(|| panic!("truncated value after {key}: {input}"));
        let val_end = if first == '"' {
            // quoted string: find the next unescaped `"`
            let mut idx = 1;
            let bytes = rest.as_bytes();
            while idx < bytes.len() {
                if bytes[idx] == b'\\' { idx += 2; continue; }
                if bytes[idx] == b'"'  { idx += 1; break; }
                idx += 1;
            }
            val_start + idx
        } else {
            // bare number / true / false / null: end at `,` or `}`
            let n = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
            val_start + n
        };
        out.push_str(replacement);
        cursor = val_end;
    }
    out.push_str(&input[cursor..]);
    out
}

fn normalize(json: &str) -> String {
    let scrubbed = replace_field(json, "scan_us", "0");
    replace_field(&scrubbed, "path", "\"<fixture>\"")
}

fn golden_path(name: &str) -> PathBuf {
    goldens_dir().join(format!("{name}.json"))
}

fn bless() -> bool {
    std::env::var_os("BLESS").is_some_and(|v| !v.is_empty() && v != "0")
}

/// Compare `actual` (already normalized) to the golden file. When
/// `BLESS=1`, write the golden instead and pass. Failure prints the diff
/// at a granular enough level to spot the drifting field.
fn assert_matches_golden(name: &str, actual: &str) {
    let path = golden_path(name);
    if bless() {
        std::fs::create_dir_all(goldens_dir()).expect("create goldens dir");
        std::fs::write(&path, actual).expect("write golden");
        eprintln!("BLESS: wrote {}", path.display());
        return;
    }
    let expected = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden at {}: {e}\nrun `BLESS=1 cargo test --test cross_arch_bit_identity` to capture",
            path.display()
        )
    });
    if expected.as_bytes() != actual.as_bytes() {
        panic!(
            "golden mismatch for {name}\n  golden:  {}\n  actual:  {}\n  diff hint: scan field-by-field for the first divergence; if a kernel produces different SIMD output on this arch, it appears here",
            expected, actual,
        );
    }
}

/// One end-to-end case: pipe `/rune <invocation> /quit`, scrub
/// machine-variable fields, compare to golden.
fn run_case(case_name: &str, invocation: &str) {
    let script = format!("/rune {invocation}\n/quit\n");
    let stdout = run_olorin_strict(&script);
    let raw = extract_rune_json(&stdout);
    let normalized = normalize(raw);
    assert_matches_golden(case_name, &normalized);
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[test]
fn parity_eacrunch_tiny_csv() {
    let path = stage_fixture("tiny.csv");
    run_case("eacrunch_tiny", &format!("eacrunch --json {path}"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn parity_eajson_tiny_jsonl() {
    let path = stage_fixture("tiny.jsonl");
    run_case("eajson_tiny", &format!("eajson --json {path}"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn parity_eaparquet_tiny_parquet() {
    let path = stage_fixture("tiny.parquet");
    run_case("eaparquet_tiny", &format!("eaparquet --json {path}"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn parity_ealog_severity_ladder() {
    let path = stage_fixture("parity_log.log");
    run_case("ealog_parity", &format!("ealog --json {path}"));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn parity_eatime_multi_hour() {
    let path = stage_fixture("parity_times.log");
    run_case("eatime_parity", &format!("eatime --json {path}"));
    let _ = std::fs::remove_file(&path);
}

/// Space-separated ISO (`YYYY-MM-DD HH:MM:SS`, with fractional seconds) —
/// exercises the timestamp_scan kernel's `T`-or-space separator branch on
/// both arches. The `.|`-combined lane mask must lower identically on NEON.
#[test]
fn parity_eatime_space_iso() {
    let path = stage_fixture("parity_times_space.log");
    run_case("eatime_space_parity", &format!("eatime --json {path}"));
    let _ = std::fs::remove_file(&path);
}

/// eacorrelate is multi-input: a 2h ISO log with planted ERROR bursts
/// 120s after each deploy in the CSV. Exercises the corr_sweep kernel,
/// the shared-grid bucketing, AND the additive `correlations[]` block —
/// scores ride the wire 4dp-rounded, so any cross-arch f32 drift larger
/// than rounding absorbs surfaces here.
#[test]
fn parity_eacorrelate_planted_lag() {
    let log = stage_fixture("parity_corr.log");
    let csv = stage_fixture("parity_corr_deploys.csv");
    run_case("eacorrelate_parity", &format!("eacorrelate --json {log} {csv}"));
    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
}

/// eadiff has no file source — it consumes two RuneOutput JSONs. Build
/// the inputs in-process via the public schema so the bytes are
/// deterministic across archs (proves `to_json()` itself is stable
/// alongside the kernels).
#[test]
fn parity_eadiff_synthetic_inputs() {
    let (a_path, b_path) = stage_eadiff_inputs();
    run_case("eadiff_parity", &format!("eadiff --json {a_path} {b_path}"));
    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);
}

fn stage_eadiff_inputs() -> (String, String) {
    // Mimic two eatime-style runs: yesterday vs today, hour buckets,
    // mixed with a numeric field so the diff exercises both axes.
    let mut a = RuneOutput::new("eatime", 1);
    a.source = Some(Source {
        path:   "yesterday".into(),
        bytes:  4096,
        format: "plaintext".into(),
    });
    a.totals = Totals { rows: 6, scan_us: 0 };
    a.categories = vec![
        Category { name: "06:00".into(), count: 2 },
        Category { name: "07:00".into(), count: 1 },
        Category { name: "15:00".into(), count: 1 },
    ];
    a.fields = vec![field_number("latency_ms", 6, 1.0, 10.0, 5.0, 30.0)];

    let mut b = RuneOutput::new("eatime", 1);
    b.source = Some(Source {
        path:   "today".into(),
        bytes:  4096,
        format: "plaintext".into(),
    });
    b.totals = Totals { rows: 8, scan_us: 0 };
    b.categories = vec![
        Category { name: "06:00".into(), count: 4 },
        Category { name: "07:00".into(), count: 0 },
        Category { name: "15:00".into(), count: 1 },
        Category { name: "16:00".into(), count: 1 },
    ];
    b.fields = vec![
        field_number("latency_ms", 8, 2.0, 12.0, 6.0, 48.0),
        field_text("level", &[("info", 4), ("warn", 2)]),
    ];

    let a_path = "/tmp/olorin_parity_eadiff_a.json".to_string();
    let b_path = "/tmp/olorin_parity_eadiff_b.json".to_string();
    write(&a_path, &a.to_json());
    write(&b_path, &b.to_json());
    (a_path, b_path)
}

fn field_number(
    name: &str, count: u64, min: f64, max: f64, mean: f64, sum: f64,
) -> olorin::runes::output::FieldStats {
    olorin::runes::output::FieldStats {
        name:       name.into(),
        kind:       olorin::runes::output::FieldKind::Number,
        count,
        null_count: None,
        numeric:    Some(NumericStats { min, max, mean, sum }),
        text:       None,
        bool:       None,
        timestamp:  None,
    }
}

fn field_text(
    name: &str, top: &[(&str, u64)],
) -> olorin::runes::output::FieldStats {
    olorin::runes::output::FieldStats {
        name:       name.into(),
        kind:       olorin::runes::output::FieldKind::Text,
        count:      top.iter().map(|(_, c)| c).sum(),
        null_count: None,
        numeric:    None,
        text:       Some(TextStats {
            unique: top.len() as u64,
            top: top.iter().map(|(v, c)| TextEntry {
                value: (*v).into(), count: *c,
            }).collect(),
        }),
        bool:      None,
        timestamp: None,
    }
}

fn write(path: &str, contents: &str) {
    let mut f = std::fs::File::create(Path::new(path)).expect("create input");
    f.write_all(contents.as_bytes()).expect("write input");
}

// ────────────────────────────────────────────────────────────────────────────
// AEAD cross-arch bit-identity.  ChaCha20-Poly1305 with fixed inputs must
// produce the same ct+tag bytes on x86 (SSE2/AVX2) and ARM NEON.  Captured
// as a 1040-byte golden (1024 ct + 16 tag) frozen on x86; Pi 5 verification
// re-runs the test and compares to the same fixture.
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn aead_seal_bit_identical() {
    olorin::kernels::ffi::init().expect("kernel init");

    let key = [0xAAu8; 32];
    let nonce = [0xBBu8; 12];
    let aad = b"olorin-vault-v2";
    let pt: Vec<u8> = (0..1024u16).map(|i| i as u8).collect();

    let mut buf = pt.clone();
    let mut tag = [0u8; 16];
    olorin::storage::aead::seal(&key, &nonce, aad, &mut buf, &mut tag);

    let mut blob = Vec::with_capacity(buf.len() + 16);
    blob.extend_from_slice(&buf);
    blob.extend_from_slice(&tag);

    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/aead_golden.bin");

    if std::env::var("BLESS").is_ok() {
        if let Some(parent) = golden_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&golden_path, &blob).expect("write golden");
        return;
    }

    let expected = std::fs::read(&golden_path).unwrap_or_else(|_| {
        panic!(
            "missing AEAD golden at {} — run with BLESS=1 to create it",
            golden_path.display()
        )
    });
    assert_eq!(
        blob, expected,
        "AEAD ct+tag diverges from cross-arch golden (1040 bytes, byte 0..1023 = ct, 1024..1039 = tag)"
    );
}
