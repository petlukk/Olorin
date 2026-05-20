//! Tests for the env-file parser used to load `~/.olorin/env` at startup.

use olorin::config::parse_line;

#[test]
fn parses_plain_key_value() {
    assert_eq!(parse_line("FOO=bar"), Some(("FOO", "bar")));
}

#[test]
fn tolerates_export_prefix() {
    assert_eq!(parse_line("export FOO=bar"), Some(("FOO", "bar")));
}

#[test]
fn strips_double_quotes() {
    assert_eq!(parse_line("FOO=\"bar baz\""), Some(("FOO", "bar baz")));
}

#[test]
fn strips_single_quotes() {
    assert_eq!(parse_line("FOO='bar baz'"), Some(("FOO", "bar baz")));
}

#[test]
fn skips_comments() {
    assert_eq!(parse_line("# FOO=bar"), None);
}

#[test]
fn skips_blank_lines() {
    assert_eq!(parse_line(""), None);
    assert_eq!(parse_line("   "), None);
}

#[test]
fn rejects_invalid_key() {
    assert_eq!(parse_line("FOO BAR=baz"), None);
    assert_eq!(parse_line("=value"), None);
}

#[test]
fn allows_underscore_and_digits_in_key() {
    assert_eq!(
        parse_line("ANTHROPIC_API_KEY=sk-abc"),
        Some(("ANTHROPIC_API_KEY", "sk-abc"))
    );
    assert_eq!(parse_line("OLORIN_THREADS=4"), Some(("OLORIN_THREADS", "4")));
}

#[test]
fn value_with_equals_kept() {
    assert_eq!(parse_line("URL=https://a=b"), Some(("URL", "https://a=b")));
}

#[test]
fn trailing_whitespace_trimmed() {
    assert_eq!(parse_line("FOO=bar   "), Some(("FOO", "bar")));
    assert_eq!(parse_line("  FOO=bar"), Some(("FOO", "bar")));
}

#[test]
fn unmatched_quotes_not_stripped() {
    // Only matched, paired quotes are stripped; lone quotes are kept.
    assert_eq!(parse_line("FOO=\"bar"), Some(("FOO", "\"bar")));
    assert_eq!(parse_line("FOO=bar'"), Some(("FOO", "bar'")));
}
