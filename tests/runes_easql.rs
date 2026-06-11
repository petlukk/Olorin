//! easql — SQL-dump summarizer. Correctness on pg_dump (COPY) + mysqldump
//! (INSERT VALUES) shapes, plus the v1 contract on edge inputs.

use olorin::runes::output::RuneOutput;
use olorin::runes::{run_rune, RUNES, OutputSafety};

fn ensure_kernels() { olorin::kernels::ffi::init().expect("kernel init"); }

fn run(stem: &str, sql: &str) -> RuneOutput {
    ensure_kernels();
    let path = std::env::temp_dir().join(format!("olorin_easql_{stem}_{}.sql", std::process::id()));
    std::fs::write(&path, sql).unwrap();
    let res = run_rune("easql", &format!("--json {}", path.display())).expect("easql runs");
    let _ = std::fs::remove_file(&path);
    RuneOutput::from_json(res.answer.as_bytes()).expect("parse RuneOutput")
}

fn rows(out: &RuneOutput, table: &str) -> u64 {
    out.categories.iter().find(|c| c.name == table).map(|c| c.count).unwrap_or(u64::MAX)
}

#[test]
fn easql_registered_and_untrusted() {
    assert!(RUNES.iter().any(|r| r.name() == "easql"), "easql missing from registry");
    let r = RUNES.iter().find(|r| r.name() == "easql").unwrap();
    assert_eq!(r.output_safety(), OutputSafety::UntrustedQuoted,
        "dump contains file-derived identifiers; must be wrapped");
}

const PG: &str = "\
CREATE TABLE users (id integer, name text, email text);
COPY users (id, name, email) FROM stdin;
1\talice\ta@x
2\tbob\tb@x
3\tcarol\tc@x
\\.
CREATE TABLE orders (id integer, user_id integer);
COPY orders (id, user_id) FROM stdin;
1\t1
2\t2
\\.
";

#[test]
fn easql_postgres_copy_dump() {
    let out = run("pg", PG);
    assert!(out.success, "{:?}", out.error);
    assert_eq!(out.source.as_ref().unwrap().format, "postgres");
    assert_eq!(out.samples.len(), 2, "two tables");
    assert_eq!(rows(&out, "users"), 3);
    assert_eq!(rows(&out, "orders"), 2);
    assert_eq!(out.totals.rows, 5);
    assert!(out.samples.iter().any(|s| s.text == "users: 3 cols"));
}

const MY: &str = "\
CREATE TABLE `users` (`id` int, `name` varchar(50));
INSERT INTO `users` VALUES (1,'alice'),(2,'bob'),(3,'carol');
CREATE TABLE `orders` (`id` int, `user_id` int);
INSERT INTO `orders` VALUES (1,1),(2,2);
";

#[test]
fn easql_mysql_insert_dump() {
    let out = run("my", MY);
    assert!(out.success, "{:?}", out.error);
    assert_eq!(out.source.as_ref().unwrap().format, "mysql");
    assert_eq!(rows(&out, "users"), 3, "three INSERT tuples");
    assert_eq!(rows(&out, "orders"), 2);
    assert_eq!(out.totals.rows, 5);
}

// Real mysqldump/pg_dump emit an explicit column list before VALUES, often
// with each tuple on its own line. The column list is a top-level paren group
// and must NOT be counted as a value tuple (regression: was +1 per statement).
const MY_COLLIST: &str = "\
INSERT INTO `Genre` (`GenreId`, `Name`) VALUES
(1,'Rock'),
(2,'Jazz'),
(3,'Metal');
INSERT INTO invoice_line (id, qty) VALUES (1,2),(3,4);
";

#[test]
fn easql_insert_column_list_not_counted_as_row() {
    let out = run("collist", MY_COLLIST);
    assert!(out.success, "{:?}", out.error);
    assert_eq!(rows(&out, "Genre"), 3, "3 tuples, not 4 — column list excluded");
    assert_eq!(rows(&out, "invoice_line"), 2, "single-line column-list INSERT");
}

#[test]
fn easql_quoted_value_with_paren_not_overcounted() {
    // A string value containing "),(" must not inflate the tuple count.
    let out = run("paren", "INSERT INTO t VALUES ('a),(b',1),('c',2);\n");
    assert!(out.success);
    assert_eq!(rows(&out, "t"), 2, "two real tuples despite '),(' inside a string");
}

// pg_dump --inserts emits no COPY blocks, but its preamble is unmistakable.
// Such a dump must be labeled postgres, not mysql (the INSERT-only fallback).
const PG_INSERTS: &str = "\
-- PostgreSQL database dump
SET standard_conforming_strings = on;
CREATE TABLE public.users (id integer, name text);
INSERT INTO public.users (id, name) VALUES (1,'alice'),(2,'bob');
";

#[test]
fn easql_pg_dump_inserts_labeled_postgres() {
    let out = run("pgins", PG_INSERTS);
    assert!(out.success, "{:?}", out.error);
    assert_eq!(out.source.as_ref().unwrap().format, "postgres",
        "pg_dump --inserts has no COPY but is unmistakably postgres");
    assert_eq!(rows(&out, "users"), 2);
}

#[test]
fn easql_ambiguous_insert_only_is_sql() {
    // No COPY, no backticks, no PG fingerprint → honestly ambiguous, not mysql.
    let out = run("amb", "CREATE TABLE t (id int);\nINSERT INTO t (id) VALUES (1),(2);\n");
    assert!(out.success);
    assert_eq!(out.source.as_ref().unwrap().format, "sql");
    assert_eq!(rows(&out, "t"), 2);
}

#[test]
fn easql_empty_fails_cleanly() {
    let out = run("empty", "");
    assert!(!out.success);
    assert!(out.error.as_ref().is_some_and(|e| !e.is_empty()));
}

#[test]
fn easql_non_sql_degrades_to_zero_tables() {
    let out = run("notsql", "just some text\nnothing structured here\n");
    assert!(out.success, "non-sql should not error, just find nothing");
    assert_eq!(out.samples.len(), 0);
    assert_eq!(out.totals.rows, 0);
}

#[test]
fn easql_create_in_comment_does_not_phantom_table() {
    // Real mysqldump has `/* Create Tables */`-style comments. `table` must not
    // match the prefix of `Tables` and register a phantom `s` table.
    let out = run("comment",
        "/* Create Tables */\nCREATE TABLE `Album` (`id` int);\nINSERT INTO `Album` VALUES (1);\n");
    assert_eq!(out.samples.len(), 1, "only Album — no phantom from the comment");
    assert_eq!(rows(&out, "Album"), 1);
    assert!(rows(&out, "s") == u64::MAX, "no phantom 's' table");
}

#[test]
fn easql_create_table_disambiguated_from_index() {
    // CREATE INDEX / VIEW must not be counted as tables.
    let out = run("idx",
        "CREATE TABLE t (id int);\nCREATE INDEX t_idx ON t (id);\nCREATE VIEW v AS SELECT 1;\n");
    assert_eq!(out.samples.len(), 1, "only the TABLE counts, not INDEX/VIEW");
}
