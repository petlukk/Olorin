//! Experiment (run 3): PIN the narration grid-continuation failure.
//!
//! Runs 1-2 established: the production `NARRATION_MAX_ANSWER_BYTES = 600`
//! byte cap is a mis-specified proxy. Long prose (866 B), json (687 B), and
//! 8-20 row grids all narrate cleanly on q3kffnimpl/Pi; only a 24-row clock
//! grid fails — and it fails DETERMINISTICALLY (6/6 across greedy + production
//! sampling). A sharp cliff sits between 20 rows (clean) and 24 rows (fail).
//! See the architecture-narration-600b-cap memory.
//!
//! This run isolates the MECHANISM of that cliff with a factorial, each cell
//! changing exactly one thing vs the known-failing 24-row ordered clock grid:
//!
//!   ctrl-20r        20 ordered clock rows      known clean (drift control)
//!   ctrl-24r        24 ordered clock rows      known fail  (drift control)
//!   m-22r           22 ordered clock rows      row-count: is the cliff ~22?
//!   m-24r-shuffled  24 clock rows, shuffled    breaks next-hour priming,
//!                                              keeps rows + values
//!   m-24r-noclock   24 rows, label col 1       breaks continuable sequence,
//!                                              keeps density + rows
//!   json-big        ~1800 B key dump           reproduce original eajson fail
//!
//! Interpretation: if shuffled AND noclock both go clean while ctrl-24r fails,
//! the trigger is a CONTINUABLE ORDERED leading column, not "24 dense rows" —
//! the gate should look for ordered/sequential structure, not mere repetition.
//! If m-22r fails, the cliff is a row count near 21-22 instead.
//!
//! Measurement: `echoed_numbers` (run 1's metric) over-fired on good summaries
//! that cite a value, so this run uses a LINE-SHAPE SIGNATURE detector as the
//! authoritative failure signal — and that detector IS the proposed fix:
//! collapse each line to a digit/alpha/space/other class pattern, find the
//! input's dominant repeated shape, flag any OUTPUT line matching it (= the
//! model emitted a data row instead of prose). It unit-tests offline
//! (`continuation_detector_discriminates`) and is validated on real output here.
//!
//! Run (on Pi): OLORIN_THREADS=3 OLORIN_NARR_SAMPLES=5 \
//!   cargo test --release --test narration_length_vs_structure -- --nocapture --ignored

use olorin::inference::generate::{Engine, GenEvent};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

const DECODE_TOKEN_CAP: usize = 768; // NARRATION_DECODE_TOKEN_CAP

/// Verbatim copy of NARRATION_SYSTEM_PROMPT (pub(crate), can't import).
const SYSTEM: &str =
    "You are a helpful data analyst. Read the user's tool output and reply with \
     1-2 plain-English sentences naming the single most important finding. Be \
     concrete: give the actual date/time, category, or value and the magnitude \
     of any peak, spike, or anomaly — e.g. 'X peaked on <date> at about N× the \
     baseline' — not vague phrasing like 'a significant peak around a certain \
     time'. State the headline finding; do not reproduce the table or list \
     every row.";

fn sample_count() -> usize {
    std::env::var("OLORIN_NARR_SAMPLES").ok().and_then(|s| s.parse().ok()).unwrap_or(5)
}

#[test]
#[ignore = "interactive eval on real hardware, run explicitly with --ignored"]
fn narration_length_vs_structure() {
    let Some(model) = resolve_model() else {
        eprintln!("SKIP: no model. Set OLORIN_NARRATION_MODEL or place \
                   gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf in ~/.olorin/models");
        return;
    };
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || drive(&model))
        .unwrap();
    handle.join().unwrap();
}

fn resolve_model() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("OLORIN_NARRATION_MODEL") {
        let pb = PathBuf::from(p);
        return pb.exists().then_some(pb);
    }
    let models = Path::new(&std::env::var("HOME").ok()?).join(".olorin/models");
    let prod = models.join("gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf");
    if prod.exists() {
        return Some(prod);
    }
    models.join("gemma-4-e2b-it-Q4_K_M.gguf").canonicalize().ok().filter(|p| p.exists())
}

fn drive(model: &Path) {
    let name = model.file_name().unwrap().to_string_lossy();
    let n = sample_count();
    println!("============================================================");
    println!(" MODEL: {name}");
    if name.contains("q3kffnimpl") {
        println!(" -> production quant: results valid for the cap.");
    } else {
        println!(" -> NOT q3kffnimpl. Indicative only; do not change the cap.");
    }
    println!(" greedy 1 @ temp0 + {n} stochastic @ prod defaults (1.0/64/0.95/0.05)");
    println!(" FAIL = >=1 output line matches the input's dominant row shape");
    println!("============================================================");

    let mut engine = Box::new(Engine::load(model, 2048).expect("load"));
    engine.max_tokens = DECODE_TOKEN_CAP;

    let mut rows: Vec<Row> = Vec::new();
    for case in build_cases() {
        let user = format!("Output of `{}`:\n\n{}", case.rune, case.answer);
        let dom = dominant_shape(&case.answer);
        println!("\n\n------------------------------------------------------------");
        println!("CELL: {:<14} in={} B  dom_shape={:?}", case.cell, case.answer.len(), dom.as_deref());
        println!("------------------------------------------------------------");

        engine.temperature = 0.0;
        let g = run_one(&mut engine, &user);
        let gc = continuation_lines(&case.answer, &g);
        println!("[greedy ] {} B  cont={} echo={}{}  | {}",
                 g.len(), gc, echoed_numbers(&case.answer, &g), tag(gc), preview(&g));

        engine.temperature = 1.0;
        let mut fails = 0;
        for i in 0..n {
            let o = run_one(&mut engine, &user);
            let c = continuation_lines(&case.answer, &o);
            if c >= 1 {
                fails += 1;
            }
            println!("[stoch {i}] {} B  cont={} echo={}{}  | {}",
                     o.len(), c, echoed_numbers(&case.answer, &o), tag(c), preview(&o));
        }
        rows.push(Row { cell: case.cell, in_bytes: case.answer.len(), greedy_fail: gc >= 1, fails, n });
    }
    print_summary(&rows);
}

fn run_one(engine: &mut Engine, user: &str) -> String {
    let got = RefCell::new(String::new());
    let on_event = |ev: GenEvent| if let GenEvent::Token(t) = ev { got.borrow_mut().push_str(t); };
    engine.generate(user, SYSTEM, &on_event).expect("generate");
    got.into_inner().trim().to_string()
}

// ---- line-shape signature detector (this is the proposed fix) ----------

/// Collapse a line to a run-length class pattern: D=digit, A=alpha, S=space,
/// O=other. "08:00  4 files" -> "DODSDSA".
fn line_shape(line: &str) -> String {
    let mut out = String::new();
    let mut last = '\0';
    for c in line.trim().chars() {
        let cls = if c.is_ascii_digit() { 'D' }
                  else if c.is_alphabetic() { 'A' }
                  else if c == ' ' { 'S' }
                  else { 'O' };
        if cls != last {
            out.push(cls);
            last = cls;
        }
    }
    out
}

/// The line shape that repeats most in the input, requiring >=3 occurrences so
/// only genuinely repetitive grids have one (prose/mixed return None).
fn dominant_shape(text: &str) -> Option<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        *counts.entry(line_shape(l)).or_default() += 1;
    }
    counts.into_iter().filter(|(_, c)| *c >= 3).max_by_key(|(_, c)| *c).map(|(s, _)| s)
}

/// Output lines that match the input's dominant row shape = the model emitted
/// a data row instead of summarizing (grid continuation).
fn continuation_lines(input: &str, output: &str) -> usize {
    let Some(dom) = dominant_shape(input) else { return 0 };
    output.lines().filter(|l| !l.trim().is_empty() && line_shape(l) == dom).count()
}

/// Distinct multi-digit input values reappearing in the output — kept as a
/// secondary cross-check; known to over-fire on good value-citing summaries.
fn echoed_numbers(input: &str, output: &str) -> usize {
    let nums: HashSet<&str> = input
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| s.len() >= 2)
        .collect();
    nums.iter().filter(|num| output.contains(**num)).count()
}

fn tag(cont: usize) -> &'static str {
    if cont >= 1 { "  <-- FAIL" } else { "" }
}

fn preview(s: &str) -> String {
    let one = s.replace('\n', " / ");
    if one.len() > 110 { format!("{}…", &one[..110]) } else { one }
}

fn print_summary(rows: &[Row]) {
    println!("\n\n===================== SUMMARY =====================");
    println!("{:<14} {:>6} {:>8} {:>11}", "cell", "in_B", "greedy", "stoch_fail");
    println!("{}", "-".repeat(44));
    for r in rows {
        println!("{:<14} {:>6} {:>8} {:>8}/{}",
                 r.cell, r.in_bytes,
                 if r.greedy_fail { "FAIL" } else { "ok" }, r.fails, r.n);
    }
    println!("\nReading it:");
    println!("  - shuffled & noclock CLEAN, ctrl-24r FAIL => trigger is the");
    println!("    ordered/continuable sequence, not 24 dense rows.");
    println!("  - m-22r FAIL => cliff is a row count near 21-22 instead.");
    println!("  - json-big FAIL => the original eajson failure reproduced;");
    println!("    note whether the shape detector caught it (cont>=1).");
}

struct Row { cell: &'static str, in_bytes: usize, greedy_fail: bool, fails: usize, n: usize }
struct Case { cell: &'static str, rune: &'static str, answer: String }

/// 24 hours of (files, MB, %) — slicing the first N gives the row sweep.
const HOURS: [(u32, f32, f32); 24] = [
    (2, 0.1, 0.3), (0, 0.0, 0.0), (1, 0.1, 0.1), (0, 0.0, 0.0),
    (0, 0.0, 0.0), (3, 0.2, 0.5), (9, 1.1, 2.8), (17, 2.4, 6.1),
    (24, 3.8, 9.7), (31, 5.2, 13.2), (28, 4.7, 11.9), (22, 3.1, 7.9),
    (19, 2.6, 6.6), (21, 2.9, 7.4), (26, 4.0, 10.2), (18, 2.2, 5.6),
    (14, 1.7, 4.3), (11, 1.3, 3.3), (8, 0.9, 2.3), (6, 0.5, 1.3),
    (4, 0.3, 0.8), (3, 0.2, 0.5), (2, 0.1, 0.4), (5, 0.4, 1.1),
];

fn row(label: &str, f: u32, mb: f32, pct: f32) -> String {
    format!("{label:<5} {f:2} files  {mb:.1} MB  {pct:.1}%")
}

fn hour_grid(rows: usize) -> String {
    HOURS.iter().take(rows).enumerate()
        .map(|(h, &(f, mb, pct))| row(&format!("{h:02}:00"), f, mb, pct))
        .collect::<Vec<_>>().join("\n")
}

/// 24 clock rows in a fixed non-monotonic order — breaks "next hour" priming
/// while keeping clock-shaped values and 24 rows.
fn shuffled_clock_grid() -> String {
    const ORDER: [usize; 24] = [7, 15, 2, 21, 9, 0, 18, 5, 23, 11, 3, 16,
                                8, 20, 1, 13, 6, 22, 10, 4, 19, 12, 17, 14];
    ORDER.iter().map(|&h| { let (f, mb, pct) = HOURS[h]; row(&format!("{h:02}:00"), f, mb, pct) })
        .collect::<Vec<_>>().join("\n")
}

/// 24 rows, non-sequential text label as column 1 — same density and row count,
/// no continuable leading sequence.
fn labeled_grid() -> String {
    const LABELS: [&str; 24] = ["png", "jpg", "pdf", "txt", "csv", "log", "zip", "mp4",
        "docx", "xlsx", "key", "mov", "gif", "svg", "json", "yaml", "toml", "rs",
        "py", "md", "wav", "bin", "tar", "iso"];
    LABELS.iter().zip(HOURS.iter())
        .map(|(lab, &(f, mb, pct))| row(lab, f, mb, pct))
        .collect::<Vec<_>>().join("\n")
}

const FIELDS: [(&str, &str); 34] = [
    ("user.id", "88421"), ("user.name", "\"alice\""),
    ("user.email", "\"alice@example.com\""), ("user.age", "34"),
    ("user.verified", "true"), ("user.created", "\"2021-03-14\""),
    ("addr.street", "\"12 Elm Road\""), ("addr.city", "\"Uppsala\""),
    ("addr.zip", "\"75236\""), ("addr.country", "\"SE\""),
    ("plan.tier", "\"pro\""), ("plan.seats", "5"),
    ("plan.renews", "\"2026-09-01\""), ("plan.price", "149"),
    ("usage.calls", "284113"), ("usage.bytes", "9928374"),
    ("usage.errors", "37"), ("usage.p95_ms", "212"),
    ("flags.beta", "false"), ("flags.sso", "true"),
    ("flags.audit", "true"), ("billing.method", "\"card\""),
    ("billing.last4", "4242"), ("billing.exp", "\"06/27\""),
    ("session.count", "1192"), ("session.last", "\"2026-05-26\""),
    ("session.device", "\"macos\""), ("session.ip", "\"10.0.0.4\""),
    ("session.region", "\"eu-north\""), ("quota.daily", "10000"),
    ("quota.used", "8421"), ("notify.email", "true"),
    ("notify.sms", "false"), ("locale.lang", "\"sv\""),
];

/// ~1800 B key dump (reproduces the original eajson failure size). Three
/// account prefixes avoid a single numeric index sequence.
fn json_big() -> String {
    let mut lines = Vec::new();
    'outer: for prefix in ["acct_a", "acct_b", "acct_c"] {
        for (k, v) in FIELDS.iter() {
            lines.push(format!("{prefix}.{k}: {v}"));
            if lines.iter().map(|l| l.len() + 1).sum::<usize>() >= 1750 {
                break 'outer;
            }
        }
    }
    lines.join("\n")
}

fn build_cases() -> Vec<Case> {
    vec![
        Case { cell: "ctrl-20r",       rune: "eatime", answer: hour_grid(20) },
        Case { cell: "ctrl-24r",       rune: "eatime", answer: hour_grid(24) },
        Case { cell: "m-22r",          rune: "eatime", answer: hour_grid(22) },
        Case { cell: "m-24r-shuffled", rune: "eatime", answer: shuffled_clock_grid() },
        Case { cell: "m-24r-noclock",  rune: "eatime", answer: labeled_grid() },
        Case { cell: "json-big",       rune: "eajson", answer: json_big() },
    ]
}

/// Model-free validation of the proposed fix's detector.
#[test]
fn continuation_detector_discriminates() {
    let grid = hour_grid(24);
    assert!(dominant_shape(&grid).is_some(), "a 24-row grid must have a dominant shape");
    // A prose summary is not a data row.
    assert_eq!(continuation_lines(&grid, "Activity peaks around the nine o'clock hour."), 0);
    // An emitted grid row is caught.
    assert!(continuation_lines(&grid, "24:00   3 files  0.1 MB  0.0%") >= 1);
    // Prose input has no dominant row shape, so nothing is ever flagged.
    assert_eq!(dominant_shape("A normal sentence. Another one here."), None);
    // json-big is ~1800 B as intended.
    assert!(json_big().len() >= 1500, "json-big should be ~1800 B, is {}", json_big().len());
}
