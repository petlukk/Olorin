//! easql — SQL-dump summarizer via the `sql_scan` SIMD kernel.
//!
//! Sweeps a `.sql` dump (pg_dump / mysqldump) for the structural keywords
//! CREATE / INSERT / COPY in one SIMD pass, then does the cheap per-marker
//! scalar work: extract the table name, count columns from a `CREATE TABLE`
//! block, and count rows per statement (newlines in a `COPY … FROM stdin`
//! block for Postgres; `),(` tuples in an `INSERT … VALUES` for MySQL).
//!
//! KISS: this is a *summarizer*, not a SQL parser. The kernel sweeps; the rune
//! nibbles a bounded region per recorded marker. Output maps tables onto the
//! v1 `categories` contract (name = table, count = rows) so the block-bar
//! chart, `--json` pipe, and `eadiff` all work for free.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::output::{Category, RuneOutput, Sample, Source, Totals};
use crate::kernels::ffi;
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;
/// Per-window cap on recorded keyword positions. The sweep is chunked (see
/// `build_output`), so this bounds memory per `sql_scan` call — not the total
/// statement count. A window denser than this resumes past its last marker, so
/// per-table attribution stays exact even for million-statement `--inserts`
/// dumps (one `INSERT` per row).
const MAX_MARKERS: usize = 65536;
/// SIMD sweep window. Snapped back to a newline per iteration so no keyword
/// straddles a cut. 2 MiB holds ~50K single-row `INSERT` markers — under
/// `MAX_MARKERS`, so the common dense case needs no in-window resume.
const CHUNK: usize = 2 * 1024 * 1024;
/// Most-rows tables to surface as chart categories.
const TOP_TABLES: usize = 40;

pub struct Easql;
pub const RUNE: Easql = Easql;

impl Rune for Easql {
    fn name(&self) -> &'static str { "easql" }
    fn description(&self) -> &'static str {
        "Summarize a SQL dump (pg_dump / mysqldump) via SIMD: table count, \
         per-table row counts and column counts, and the dump dialect. Sweeps \
         CREATE/INSERT/COPY in one pass; does not execute SQL. Args: \
         [--json] <path.sql>."
    }
    fn usage(&self) -> &'static str { "easql [--json] <path.sql>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let (path, json_mode) = parse_args(args);
        let output = execute(&path);
        let answer = if json_mode {
            output.to_json()
        } else if let Some(err) = &output.error {
            err.clone()
        } else {
            format_text(&output)
        };
        RuneResult {
            answer:     truncate_answer(&answer),
            details:    None,
            success:    output.success,
            timing_us:  t0.elapsed().as_micros() as u64,
            structured: json_mode,
        }
    }
}

fn parse_args(args: &str) -> (String, bool) {
    let trimmed = args.trim();
    if let Some(rest) = trimmed.strip_prefix("--json ") {
        (rest.trim().to_string(), true)
    } else if let Some(rest) = trimmed.strip_suffix(" --json") {
        (rest.trim().to_string(), true)
    } else if trimmed == "--json" {
        (String::new(), true)
    } else {
        (trimmed.to_string(), false)
    }
}

fn execute(path: &str) -> RuneOutput {
    if path.is_empty() {
        return error_output("usage: easql [--json] <path.sql>");
    }
    let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let resolved = match resolve_path(path, &home) {
        Ok(p) => p,
        Err(PathError::OutsideAllowlist) =>
            return error_output("path rejected: outside allowlist (~ or /tmp only)"),
        Err(PathError::NotFound) => return error_output("file not found"),
        Err(PathError::TooLarge(n)) =>
            return error_output(&format!("file too large: {n} bytes")),
        Err(PathError::Io(e)) => return error_output(&format!("io error: {e}")),
    };
    let bytes = match open_capped(&resolved, &home) {
        Ok(b) => b,
        Err(PathError::NotFound) => return error_output("file not found"),
        Err(PathError::TooLarge(n)) =>
            return error_output(&format!("file too large: {n} bytes")),
        Err(PathError::OutsideAllowlist) =>
            return error_output("path rejected: outside allowlist (~ or /tmp only)"),
        Err(PathError::Io(e)) => return error_output(&format!("io error: {e}")),
    };
    build_output(&bytes, resolved.to_string_lossy().into_owned())
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("easql", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Kw { Create, Insert, Copy, Other }

fn build_output(bytes: &[u8], path: String) -> RuneOutput {
    if bytes.is_empty() {
        return error_output("empty file");
    }
    if bytes.len() > i32::MAX as usize {
        return error_output(&format!(
            "file too large for sql_scan: {} bytes (2 GB limit)", bytes.len()
        ));
    }

    let t_scan = Instant::now();
    let mut counts = [0i32; 4];
    let mut positions = vec![0i32; MAX_MARKERS];
    let mut n_pos = 0i32;
    let mut scratch = [0u8; 16];

    // Ordered per-table accumulation: name -> (rows, cols). Insertion order
    // preserved via a parallel Vec so the schema reads in declaration order.
    let mut order: Vec<String> = Vec::new();
    let mut rows: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut cols: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    let mut saw_copy = false;

    // Chunked SIMD sweep: scan the dump in newline-aligned windows so per-table
    // attribution stays exact even when there are more keyword markers than one
    // position buffer holds (`pg_dump --inserts` / `mysqldump --skip-extended-
    // insert` → one statement per row). The window end snaps to a newline so no
    // keyword straddles the cut; the nibble functions read the full buffer, so a
    // statement extending past the window tail is still counted whole. A window
    // denser than MAX_MARKERS resumes just past its last marker (the backstop).
    let len = bytes.len();
    let mut start = 0usize;
    while start < len {
        let mut end = (start + CHUNK).min(len);
        if end < len {
            if let Some(nl) = bytes[start..end].iter().rposition(|&b| b == b'\n') {
                end = start + nl + 1;
            }
            // No newline in the window → mid-data of a giant single-line
            // statement; safe to cut on CHUNK (its keyword was found already).
        }
        let sub = &bytes[start..end];
        unsafe {
            ffi::sql_scan(
                sub.as_ptr(), sub.len() as i32,
                counts.as_mut_ptr(),
                positions.as_mut_ptr(), MAX_MARKERS as i32, &mut n_pos,
                scratch.as_mut_ptr(),
            );
        }
        let n = n_pos as usize;
        for &off in &positions[..n] {
            let p = start + off as usize;
            match keyword_at(bytes, p) {
                Kw::Create => {
                    // CREATE <ws> TABLE <ws> <name> ( … )
                    let after = skip_kw(bytes, p, b"create");
                    if !word_is(bytes, after, b"table") { continue; }
                    let after = skip_kw(bytes, after, b"table");
                    let after = skip_optional(bytes, after, b"if not exists");
                    if let Some((name, cend)) = read_ident(bytes, after) {
                        register(&mut order, &mut rows, &mut cols, &name);
                        cols.insert(name, count_columns(bytes, cend));
                    }
                }
                Kw::Insert => {
                    // INSERT <ws> INTO <ws> <name> … VALUES (…),(…);
                    let after = skip_kw(bytes, p, b"insert");
                    if !word_is(bytes, after, b"into") { continue; }
                    let after = skip_kw(bytes, after, b"into");
                    if let Some((name, iend)) = read_ident(bytes, after) {
                        register(&mut order, &mut rows, &mut cols, &name);
                        *rows.entry(name).or_insert(0) += count_insert_rows(bytes, iend);
                    }
                }
                Kw::Copy => {
                    // COPY <ws> <name> … FROM stdin; <rows…> \.
                    saw_copy = true;
                    let after = skip_kw(bytes, p, b"copy");
                    if let Some((name, cend)) = read_ident(bytes, after) {
                        register(&mut order, &mut rows, &mut cols, &name);
                        *rows.entry(name).or_insert(0) += count_copy_rows(bytes, cend);
                    }
                }
                Kw::Other => {}
            }
        }
        if n == MAX_MARKERS {
            start += positions[n - 1] as usize + 1; // denser than the buffer: resume past last marker
        } else {
            start = end;
        }
    }
    let _ = counts; // per-window counts unused; dialect uses saw_copy + content sniff
    let dialect = detect_dialect(bytes, saw_copy);

    let scan_us = t_scan.elapsed().as_micros() as u64;
    let total_rows: u64 = rows.values().sum();

    // Categories: tables by row count, biggest first (chart-friendly), capped.
    let mut tbl: Vec<(String, u64)> = order.iter()
        .map(|n| (n.clone(), *rows.get(n).unwrap_or(&0)))
        .collect();
    tbl.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let categories: Vec<Category> = tbl.iter().take(TOP_TABLES)
        .map(|(n, c)| Category { name: n.clone(), count: *c })
        .collect();

    // Samples: per-table column counts in declaration order (schema view).
    let samples: Vec<Sample> = order.iter()
        .filter_map(|n| cols.get(n).map(|c| Sample {
            byte_offset: None, line: None, timestamp: None,
            text: format!("{n}: {c} cols"),
        }))
        .collect();

    let mut out = RuneOutput::new("easql", RUNE_VERSION);
    out.source = Some(Source { path, bytes: bytes.len() as u64, format: dialect.to_string() });
    out.totals = Totals { rows: total_rows, scan_us };
    out.categories = categories;
    out.samples = samples;
    out
}

fn register(
    order: &mut Vec<String>,
    rows: &mut std::collections::HashMap<String, u64>,
    cols: &mut std::collections::HashMap<String, u64>,
    name: &str,
) {
    if !rows.contains_key(name) && !cols.contains_key(name) {
        order.push(name.to_string());
        rows.insert(name.to_string(), 0);
    }
}

/// Classify the dump dialect, most-definitive signal first:
/// 1. `COPY … FROM stdin` → postgres (PostgreSQL-only bulk-load syntax).
/// 2. A backtick anywhere → mysql (MySQL's identifier quote; pg never emits one).
/// 3. A PostgreSQL fingerprint that survives `pg_dump --inserts` (where the
///    COPY blocks are replaced by INSERTs) → postgres.
/// 4. Otherwise genuinely ambiguous (INSERT-only, no quoting/fingerprint) → sql.
/// Scans a bounded prefix — the tool preamble and first `CREATE TABLE` (where
/// MySQL's backticks first appear) are always near the top of a dump.
fn detect_dialect(bytes: &[u8], saw_copy: bool) -> &'static str {
    if saw_copy { return "postgres"; }
    let head = &bytes[..bytes.len().min(256 * 1024)];
    if head.contains(&b'`') { return "mysql"; }
    const PG: [&[u8]; 5] = [
        b"pg_catalog", b"standard_conforming", b"-- PostgreSQL", b"\\connect", b"\\c ",
    ];
    if PG.iter().any(|p| head.windows(p.len()).any(|w| w == *p)) { return "postgres"; }
    "sql"
}

/// Classify the keyword the kernel recorded at byte `p` (case-insensitive).
fn keyword_at(bytes: &[u8], p: usize) -> Kw {
    if word_is(bytes, p, b"create") { Kw::Create }
    else if word_is(bytes, p, b"insert") { Kw::Insert }
    else if word_is(bytes, p, b"copy") { Kw::Copy }
    else { Kw::Other }
}

/// True if `bytes[p..]` is `kw` as a whole word (ASCII case-insensitive): the
/// keyword bytes match *and* the next byte is a word boundary, so `table` does
/// not match inside `tables` (a `-- Create Tables` comment must not register a
/// phantom `s` table).
fn word_is(bytes: &[u8], p: usize, kw: &[u8]) -> bool {
    if p + kw.len() > bytes.len() { return false; }
    if !bytes[p..p + kw.len()].iter().zip(kw).all(|(b, k)| b.to_ascii_lowercase() == *k) {
        return false;
    }
    match bytes.get(p + kw.len()) {
        Some(c) => !(c.is_ascii_alphanumeric() || *c == b'_'),
        None => true,
    }
}

/// Skip `kw` (assumed present at `p`) then any following ASCII whitespace.
fn skip_kw(bytes: &[u8], p: usize, kw: &[u8]) -> usize {
    skip_ws(bytes, p + kw.len())
}

fn skip_ws(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() && bytes[p].is_ascii_whitespace() { p += 1; }
    p
}

/// Skip an optional space-separated phrase (e.g. `if not exists`) if present.
fn skip_optional(bytes: &[u8], p: usize, phrase: &[u8]) -> usize {
    let mut q = p;
    for word in phrase.split(|&b| b == b' ') {
        let s = skip_ws(bytes, q);
        if word_is(bytes, s, word) { q = s + word.len(); } else { return p; }
    }
    skip_ws(bytes, q)
}

/// Read a table identifier at `p`: bare `name`, `"quoted"`, `` `backtick` ``,
/// or schema-qualified `schema.table` (last segment kept). Returns the
/// unquoted leaf name and the byte index just past the identifier.
fn read_ident(bytes: &[u8], p: usize) -> Option<(String, usize)> {
    let p = skip_ws(bytes, p);
    if p >= bytes.len() { return None; }
    let mut i = p;
    let mut leaf_start = p;
    while i < bytes.len() {
        match bytes[i] {
            b'"' | b'`' => {
                let q = bytes[i];
                i += 1; leaf_start = i;
                while i < bytes.len() && bytes[i] != q { i += 1; }
                let name = String::from_utf8_lossy(&bytes[leaf_start..i]).into_owned();
                i += 1; // closing quote
                // schema-qualified: a '.' continues to the real table.
                if i < bytes.len() && bytes[i] == b'.' { i += 1; leaf_start = i; continue; }
                return Some((name, i));
            }
            b'.' => { i += 1; leaf_start = i; }
            c if c.is_ascii_whitespace() || c == b'(' || c == b';' => break,
            _ => { i += 1; }
        }
    }
    if leaf_start >= i { return None; }
    Some((String::from_utf8_lossy(&bytes[leaf_start..i]).into_owned(), i))
}

/// Count top-level columns in a `CREATE TABLE … ( … )` block starting at/after
/// `p`: commas at paren-depth 1, ignoring quoted strings. Returns 0 if no `(`.
fn count_columns(bytes: &[u8], p: usize) -> u64 {
    let mut i = skip_ws(bytes, p);
    if i >= bytes.len() || bytes[i] != b'(' { return 0; }
    i += 1;
    let mut depth = 1i32;
    let mut commas = 0u64;
    let mut in_str = false;
    while i < bytes.len() && depth > 0 {
        let c = bytes[i];
        if in_str {
            if c == b'\'' { in_str = false; }
        } else {
            match c {
                b'\'' => in_str = true,
                b'(' => depth += 1,
                b')' => depth -= 1,
                b',' if depth == 1 => commas += 1,
                _ => {}
            }
        }
        i += 1;
    }
    // N top-level commas → N+1 column definitions (CONSTRAINT lines inflate
    // this slightly; acceptable for a summary).
    commas + 1
}

/// Count rows in a single `INSERT … VALUES (…),(…);` statement at `p`:
/// top-level value tuples, single-quote-aware. Scans to the statement-
/// terminating `;` at paren-depth 0.
fn count_insert_rows(bytes: &[u8], p: usize) -> u64 {
    // Skip an optional column list — `INSERT INTO t (c1, c2, …) VALUES …`.
    // It is the only parenthesised group between the table name and VALUES;
    // counting it as a value tuple would over-count every statement by one
    // (real mysqldump/pg_dump always emit it). Consume it single-quote-aware.
    let mut i = skip_ws(bytes, p);
    if i < bytes.len() && bytes[i] == b'(' {
        let mut depth = 0i32;
        let mut in_str = false;
        while i < bytes.len() {
            let c = bytes[i];
            if in_str {
                if c == b'\'' {
                    if i + 1 < bytes.len() && bytes[i + 1] == b'\'' { i += 2; continue; }
                    in_str = false;
                }
            } else {
                match c {
                    b'\'' => in_str = true,
                    b'(' => depth += 1,
                    b')' => { depth -= 1; if depth == 0 { i += 1; break; } }
                    _ => {}
                }
            }
            i += 1;
        }
    }
    // Now count value tuples: top-level '(' groups up to the ';' at depth 0.
    let mut depth = 0i32;
    let mut in_str = false;
    let mut tuples = 0u64;
    let mut seen_open = false;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if c == b'\'' {
                // '' is an escaped quote inside the string.
                if i + 1 < bytes.len() && bytes[i + 1] == b'\'' { i += 2; continue; }
                in_str = false;
            }
        } else {
            match c {
                b'\'' => in_str = true,
                b'(' => { depth += 1; if depth == 1 { tuples += 1; seen_open = true; } }
                b')' => { if depth > 0 { depth -= 1; } }
                b';' if depth == 0 => break,
                _ => {}
            }
        }
        i += 1;
    }
    if seen_open { tuples } else { 0 }
}

/// Count rows in a Postgres `COPY t … FROM stdin;` block: newlines between the
/// statement-terminating `;` and the `\.` end marker on its own line.
fn count_copy_rows(bytes: &[u8], p: usize) -> u64 {
    // Advance to the ';' that ends the COPY header.
    let mut i = p;
    while i < bytes.len() && bytes[i] != b';' && bytes[i] != b'\n' { i += 1; }
    while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
    if i >= bytes.len() { return 0; }
    i += 1; // first data byte
    let mut n = 0u64;
    while i < bytes.len() {
        // End marker: a line that is exactly "\.".
        if bytes[i] == b'\\' && i + 1 < bytes.len() && bytes[i + 1] == b'.' {
            let at_line_start = i == 0 || bytes[i - 1] == b'\n';
            if at_line_start { break; }
        }
        if bytes[i] == b'\n' { n += 1; }
        i += 1;
    }
    n
}

fn format_text(out: &RuneOutput) -> String {
    let src = out.source.as_ref().expect("build_output populates source on success");
    let mut buf = String::with_capacity(512);
    buf.push_str(&format!("dialect: {}\n", src.format));
    buf.push_str(&format!("tables:  {}\n", out.samples.len()));
    buf.push_str(&format!("rows:    {}\n", out.totals.rows));
    buf.push_str(&format!("scan:    {}\n", super::common::format_scan_time(out.totals.scan_us)));
    if !out.categories.is_empty() {
        buf.push('\n');
        buf.push_str("rows by table:\n");
        for c in &out.categories {
            buf.push_str(&format!("  {:<28} {}\n", c.name, c.count));
        }
    }
    buf
}
