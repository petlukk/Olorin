//! Palantír daemon — name derivation, pidfile, snapshot, liveness, status/stop.
//!
//! The fork/detach itself is verified on-target (it can't be unit-tested in a
//! shared test process); these cover everything around it. State lives under
//! `$HOME/.olorin/palantir`, so each test isolates `HOME` to a tmpdir.

use olorin::palantir::daemon::{
    is_alive, read_pid, remove_pid, state_dir, status, status_json, stop, watcher_name, write_pid,
    write_snapshot,
};
use olorin::palantir::watch::Alert;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> &'static Mutex<()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
}

struct IsoHome {
    _guard: std::sync::MutexGuard<'static, ()>,
    old: Option<String>,
    dir: PathBuf,
}
impl IsoHome {
    fn new(tag: &str) -> Self {
        let guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let old = std::env::var("HOME").ok();
        let dir = std::env::temp_dir().join(format!("olorin-pal-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("HOME", &dir);
        Self { _guard: guard, old, dir }
    }
}
impl Drop for IsoHome {
    fn drop(&mut self) {
        match &self.old {
            Some(h) => std::env::set_var("HOME", h),
            None => std::env::remove_var("HOME"),
        }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn watcher_name_derives_and_sanitizes() {
    assert_eq!(watcher_name("/var/log/app.log", None), "app.log");
    assert_eq!(watcher_name("/var/log/app.log", Some("prod")), "prod");
    assert_eq!(watcher_name("/x/y", Some("My Watch!")), "My_Watch_");
    assert_eq!(watcher_name("", None), "watch"); // empty → safe default
}

#[test]
fn is_alive_distinguishes_real_from_bogus() {
    assert!(is_alive(std::process::id() as i32), "this test process is alive");
    assert!(!is_alive(0x7FFF_FFFF), "an absurd pid is not alive");
}

#[test]
fn pidfile_roundtrips() {
    let _h = IsoHome::new("pid");
    assert_eq!(read_pid("w"), None, "no pidfile yet");
    write_pid("w").unwrap();
    assert_eq!(read_pid("w"), Some(std::process::id() as i32));
    remove_pid("w");
    assert_eq!(read_pid("w"), None);
}

#[test]
fn snapshot_is_written_atomically_with_the_last_alert() {
    let _h = IsoHome::new("snap");
    let alert = Alert::Anomaly { at: 5, rate: 9, baseline: 1.0 };
    write_snapshot("w", "/p/system.log", "iso8601", Some(10), Some(&alert), 1234);
    let body = std::fs::read_to_string(state_dir().join("w.json")).expect("snapshot written");
    assert!(body.contains("\"name\":\"w\""), "{body}");
    assert!(body.contains("\"path\":\"/p/system.log\""));
    assert!(body.contains("\"format\":\"iso8601\""));
    assert!(body.contains("\"lag\":10"));
    assert!(body.contains("\"status\":\"alerting\""), "an anomaly means alerting: {body}");
    assert!(body.contains("\"kind\":\"anomaly\""));
    assert!(body.contains("\"at_unix\":5"), "alert time is the alert's own, not the write time: {body}");
    assert!(body.contains("\"updated_at_unix\":1234"));
    // No leftover temp file.
    assert!(!state_dir().join("w.json.tmp").exists(), "temp file should be renamed away");
}

#[test]
fn snapshot_status_is_watching_without_an_alert() {
    let _h = IsoHome::new("snap2");
    write_snapshot("w", "/p/log", "pending", None, None, 1);
    let body = std::fs::read_to_string(state_dir().join("w.json")).unwrap();
    assert!(body.contains("\"status\":\"watching\""), "{body}");
    assert!(body.contains("\"lag\":null"));
    assert!(body.contains("\"last_alert\":null"));
}

#[test]
fn status_json_reflects_live_watchers_only() {
    let _h = IsoHome::new("statusjson");
    let now = 1_000_000i64;

    // live (this process) + a recent alert → "alerting" with the message
    write_pid("a").unwrap();
    write_snapshot("a", "/p/a.log", "iso8601", Some(10),
        Some(&Alert::Confirmed { trigger_at: now - 10, at: now - 5, errors: 114 }), now);
    // live + no alert → "watching"
    write_pid("b").unwrap();
    write_snapshot("b", "/p/b.log", "iso8601", None, None, now);
    // a stale pidfile (bogus pid) → omitted from the badge
    std::fs::write(state_dir().join("dead.pid"), "2147483647").unwrap();
    write_snapshot("dead", "/p/dead.log", "iso8601", None,
        Some(&Alert::Anomaly { at: now, rate: 9, baseline: 0.0 }), now);

    let j = status_json(now);
    assert!(j.starts_with('[') && j.ends_with(']'), "valid array: {j}");
    assert!(j.contains("\"name\":\"a\"") && j.contains("\"status\":\"alerting\""), "{j}");
    assert!(j.contains("CASCADE CONFIRMED"), "alert message surfaced: {j}");
    assert!(j.contains("\"name\":\"b\""), "watching watcher listed: {j}");
    assert!(!j.contains("\"name\":\"dead\""), "stale daemon excluded: {j}");
}

#[test]
fn cleared_stand_down_returns_the_badge_to_green() {
    // Regression: a predicted cascade that resolves to a `clear` stand-down is a
    // recent alert but *good news*. The badge must read "watching" (green), not
    // stay red for the rest of the freshness window. A live `predicted` incident
    // alongside it still reads "alerting" (red), so the badge can change colour.
    let _h = IsoHome::new("clear");
    let now = 1_000_000i64;

    write_pid("c").unwrap();
    write_snapshot("c", "/p/c.log", "iso8601", Some(10),
        Some(&Alert::Clear { trigger_at: now - 45, window: 45 }), now);
    write_pid("p").unwrap();
    write_snapshot("p", "/p/p.log", "iso8601", Some(10),
        Some(&Alert::Predicted { at: now - 3, eta: Some(now + 7), window: 45 }), now);

    let j = status_json(now);
    assert!(j.contains("\"name\":\"c\""), "cleared watcher still listed: {j}");
    assert!(!j.contains("window clear"), "a clear must not surface as an alert: {j}");
    assert!(j.contains("\"name\":\"p\"") && j.contains("trigger detected"),
        "the live predicted incident still alerts: {j}");
    assert_eq!(j.matches("\"status\":\"alerting\"").count(), 1,
        "only the predicted incident is alerting, not the clear: {j}");
}

#[test]
fn status_and_stop_handle_a_recorded_then_missing_watcher() {
    let _h = IsoHome::new("life");
    write_pid("w").unwrap(); // this process is the "daemon" for the test
    write_snapshot("w", "/p/log", "iso8601", Some(3), None, 9);
    assert_eq!(status(Some("w")), 0, "status of a recorded watcher succeeds");
    // Stopping a ghost (no pidfile) is an error and is idempotent-safe.
    assert_eq!(stop(Some("ghost")), 1);
}
