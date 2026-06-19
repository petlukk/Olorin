//! Web-UI palantír start/stop: the path guard and request validation that gate
//! `POST /api/palantir/watch`. The spawn + socket glue is proven on-target (Pi);
//! here we pin the security-critical decisions, which are pure.
//!
//! Two properties matter most:
//!   1. The watch guard permits read-only `/var/log` (ops logs) on top of
//!      `$HOME`/`/tmp`, but still refuses `..`, sensitive subtrees, and paths
//!      outside every allowed root.
//!   2. Widening the *watch* guard must NOT widen the *LLM-tool* guard — a
//!      prompt-injected `read_file /var/log/...` must still be refused.

#![cfg(unix)]

use olorin::core::path_guard::{resolve_safe_path_checked, resolve_watch_path, AccessMode, PathError};
use olorin::interface::server_palantir::plan_watch;

// ── watch path guard ────────────────────────────────────────────────────────

#[test]
fn watch_guard_allows_var_log() {
    // /var/log exists on Linux; the leaf need not exist (parent canonicalizes).
    let p = resolve_watch_path("/var/log/olorin-webui-test.log")
        .expect("/var/log must be allowed for watch targets");
    assert!(p.starts_with("/var/log"), "resolved under /var/log: {p:?}");
}

#[test]
fn watch_guard_allows_tmp() {
    assert!(resolve_watch_path("/tmp/olorin-webui-test.log").is_ok());
}

#[test]
fn watch_guard_refuses_outside_roots() {
    assert_eq!(resolve_watch_path("/etc/passwd"), Err(PathError::OutsideAllowlist));
}

#[test]
fn watch_guard_refuses_sensitive_and_traversal() {
    assert!(matches!(resolve_watch_path("~/.ssh/id_rsa"), Err(PathError::Sensitive(_))));
    assert_eq!(resolve_watch_path("/var/log/../../etc/passwd"), Err(PathError::ParentTraversal));
}

#[test]
fn llm_tool_guard_was_not_widened_to_var_log() {
    // The decisive regression: the *tool* guard must still reject /var/log, so
    // adding it to the watch guard didn't hand read_file/grep/ls a new root.
    assert_eq!(
        resolve_safe_path_checked("/var/log/syslog", AccessMode::Read),
        Err(PathError::OutsideAllowlist),
    );
}

// ── request validation (plan_watch) ─────────────────────────────────────────

#[test]
fn rejects_alert_sinks_from_web() {
    // exec: would be RCE, webhook: SSRF — refused before anything else.
    assert!(plan_watch(br#"{"path":"/var/log/a.log","notify":"exec:rm -rf /"}"#)
        .unwrap_err().contains("sinks"));
    assert!(plan_watch(br#"{"path":"/var/log/a.log","sink":"webhook:http://x"}"#)
        .unwrap_err().contains("sinks"));
}

#[test]
fn rejects_missing_path_and_bad_sensitivity() {
    assert!(plan_watch(br#"{}"#).unwrap_err().contains("path"));
    assert!(plan_watch(br#"{"path":"   "}"#).unwrap_err().contains("path"));
    assert!(plan_watch(br#"{"path":"/var/log/a.log","sensitivity":"loud"}"#)
        .unwrap_err().contains("low|med|high"));
}

#[test]
fn rejects_path_outside_allowlist() {
    assert!(plan_watch(br#"{"path":"/etc/shadow"}"#).is_err());
}

#[test]
fn accepts_valid_request_and_derives_name() {
    let plan = plan_watch(br#"{"path":"/var/log/myservice.log"}"#)
        .expect("a /var/log path is a valid watch target");
    assert!(plan.path.starts_with("/var/log"));
    assert!(!plan.name.is_empty(), "a watcher name is derived from the path");
    assert!(plan.sensitivity.is_none(), "no sensitivity → daemon default");

    let named = plan_watch(br#"{"path":"/var/log/x.log","name":"frontend","sensitivity":"high"}"#)
        .expect("explicit name + sensitivity accepted");
    assert!(named.name.contains("frontend"));
    assert_eq!(named.sensitivity.as_deref(), Some("high"));
}
