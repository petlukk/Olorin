//! Audit log — append-only JSON Lines record of every dispatch step.
//!
//! Enabled by `--audit <path>` on the command line. Each event is one
//! JSON object on its own line: `{"ts_ms":...,"turn":...,"phase":"...",...}`.
//! Events are written in order with a `Mutex<File>` serializing across
//! threads so the JSONL stream stays valid even under concurrent
//! server-mode dispatches.
//!
//! Goal: a privacy-conscious user can read this file and see exactly
//! what Olorin did with their data — which kernels ran, which paths
//! matched, where (and whether) the LLM was invoked. The file does NOT
//! capture user text or rune output content, only metadata: lengths,
//! match names, timing, blocked/not-blocked. That keeps the audit log
//! itself from leaking the data it's meant to protect.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Field value in an audit event. Kept narrow on purpose — anything
/// richer (objects, arrays) would push us toward needing a real JSON
/// builder. The dispatch-step events we emit are flat key-value.
#[derive(Debug, Clone, Copy)]
pub enum AuditValue<'a> {
    Str(&'a str),
    I64(i64),
    Bool(bool),
}

/// Append-only audit log. Construct with [`AuditLog::open`]; pass into
/// `DispatchContext` via the constructor. Cheap to clone-of-Option since
/// the actual handle is wrapped in `Arc` at the consumer.
pub struct AuditLog {
    inner: Mutex<File>,
    turn: AtomicI32,
}

impl AuditLog {
    /// Open or create an audit log at `path` in append mode. The file is
    /// kept open for the life of the process; events are flushed per
    /// `emit` call to survive a kill (no buffered loss on SIGTERM).
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true).append(true).open(path)?;
        Ok(Self {
            inner: Mutex::new(file),
            turn: AtomicI32::new(0),
        })
    }

    /// Allocate a new turn number. Each dispatch call should call this
    /// once at entry and pass the result through to all emits for that
    /// turn so events can be correlated.
    pub fn next_turn(&self) -> i32 {
        self.turn.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Emit one event to the log. Best-effort: write failures are
    /// surfaced to stderr but never bubbled up, since killing dispatch
    /// over an audit-log failure would be worse than the missing line.
    pub fn emit(&self, turn: i32, phase: &str, fields: &[(&str, AuditValue<'_>)]) {
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let mut line = String::with_capacity(128 + fields.len() * 32);
        line.push('{');
        write_json_kv_int(&mut line, "ts_ms", ts_ms);
        line.push(',');
        write_json_kv_int(&mut line, "turn", turn as i64);
        line.push(',');
        write_json_kv_str(&mut line, "phase", phase);
        for (k, v) in fields {
            line.push(',');
            match v {
                AuditValue::Str(s) => write_json_kv_str(&mut line, k, s),
                AuditValue::I64(n) => write_json_kv_int(&mut line, k, *n),
                AuditValue::Bool(b) => write_json_kv_bool(&mut line, k, *b),
            }
        }
        line.push_str("}\n");
        if let Ok(mut f) = self.inner.lock() {
            if let Err(e) = f.write_all(line.as_bytes()) {
                eprintln!("[audit] write failed: {e}");
            }
        }
    }
}

// ── JSON helpers (flat-only, no nested objects/arrays) ────────────────────────

fn write_json_kv_str(out: &mut String, key: &str, value: &str) {
    out.push('"');
    write_json_string_escaped(out, key);
    out.push_str("\":\"");
    write_json_string_escaped(out, value);
    out.push('"');
}

fn write_json_kv_int(out: &mut String, key: &str, value: i64) {
    out.push('"');
    write_json_string_escaped(out, key);
    out.push_str("\":");
    out.push_str(&value.to_string());
}

fn write_json_kv_bool(out: &mut String, key: &str, value: bool) {
    out.push('"');
    write_json_string_escaped(out, key);
    out.push_str("\":");
    out.push_str(if value { "true" } else { "false" });
}

/// Escape a string for JSON output. Handles `"`, `\`, and control
/// characters (< 0x20). Non-ASCII bytes pass through assuming UTF-8 —
/// this is fine for the small set of strings we emit (phase names,
/// command names, kernel names) which are all ASCII.
fn write_json_string_escaped(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}
