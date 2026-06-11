//! Pinned-stack canary for `sql_scan` (the easql kernel) — same guard as the
//! ealog canary (tests/ealog_large_log_canary.rs). Runs easql on an 8 MiB
//! dump inside a thread with a 2 MiB stack: a fixed-frame kernel fits any
//! input; a regression that grows stack per SIMD iteration overflows it
//! deterministically on any runner and aborts the process (red CI).

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;

const PINNED_STACK: usize = 2 * 1024 * 1024; // 2 MiB
const DUMP_BYTES: usize = 8 * 1024 * 1024; // 8 MiB

#[test]
fn easql_large_dump_does_not_grow_stack() {
    olorin::kernels::ffi::init().expect("kernel init");

    // A pg_dump-shaped dump: one big COPY block of repeated rows.
    let header = b"CREATE TABLE big (id integer, payload text);\nCOPY big (id, payload) FROM stdin;\n";
    let row = b"1\tsome payload text here for the row\n";
    let mut data = Vec::with_capacity(DUMP_BYTES + 256);
    data.extend_from_slice(header);
    while data.len() < DUMP_BYTES { data.extend_from_slice(row); }
    data.extend_from_slice(b"\\.\n");
    let path = std::env::temp_dir().join(format!("olorin_easql_canary_{}.sql", std::process::id()));
    std::fs::write(&path, &data).unwrap();
    let path_str = path.to_string_lossy().into_owned();

    let handle = std::thread::Builder::new()
        .stack_size(PINNED_STACK)
        .spawn(move || {
            let res = run_rune("easql", &format!("--json {path_str}")).expect("easql runs");
            RuneOutput::from_json(res.answer.as_bytes()).expect("parse RuneOutput")
        })
        .expect("spawn pinned-stack thread");

    let out = handle
        .join()
        .expect("easql overflowed a 2 MiB stack — sql_scan is growing stack per iteration");
    let _ = std::fs::remove_file(&path);

    assert!(out.success, "easql should summarize a large dump: {:?}", out.error);
    // One table, a large positive row count (every data line is one row).
    assert_eq!(out.samples.len(), 1);
    assert!(out.totals.rows > 100_000, "expected ~200k rows, got {}", out.totals.rows);
}
