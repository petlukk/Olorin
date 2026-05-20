//! Native env-file reader for `~/.olorin/env`.
//!
//! Format: `KEY=VALUE` per line. `#` starts a comment. A leading `export `
//! is tolerated so files copied from shell rc snippets work. Values may be
//! wrapped in single or double quotes; the quotes are stripped, no escape
//! processing. Lines that already exist in the process environment are
//! left untouched — explicit env beats file env.
//!
//! The file is optional. A missing file is not an error.

use std::path::PathBuf;

/// Load `~/.olorin/env` into the process environment if it exists.
/// Variables already set in the environment are not overwritten.
pub fn load_env_file() {
    let Some(home) = crate::home_dir() else { return };
    load_from_path(home.join(".olorin").join("env"));
}

fn load_from_path(path: PathBuf) {
    let Ok(contents) = std::fs::read_to_string(&path) else { return };
    for line in contents.lines() {
        if let Some((key, value)) = parse_line(line) {
            if std::env::var_os(key).is_none() {
                // Safety: single-threaded at startup, no other threads
                // are reading env vars yet (kernel init hasn't run).
                unsafe { std::env::set_var(key, value); }
            }
        }
    }
}

/// Parse a single env-file line. Returns `None` for blanks, comments,
/// or invalid keys. Exposed for testing.
pub fn parse_line(raw: &str) -> Option<(&str, &str)> {
    let line = raw.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
    let eq = line.find('=')?;
    let key = line[..eq].trim();
    if key.is_empty() || !key.chars().all(|c| c == '_' || c.is_ascii_alphanumeric()) {
        return None;
    }
    let value = line[eq + 1..].trim();
    let value = strip_matching_quotes(value);
    Some((key, value))
}

fn strip_matching_quotes(value: &str) -> &str {
    let bytes = value.as_bytes();
    if bytes.len() >= 2 {
        let first = bytes[0];
        let last = bytes[bytes.len() - 1];
        if first == last && (first == b'"' || first == b'\'') {
            return &value[1..value.len() - 1];
        }
    }
    value
}
