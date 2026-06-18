//! Palantír daemon — name derivation, pidfile, snapshot, liveness, status/stop.
//!
//! The fork/detach itself is verified on-target (it can't be unit-tested in a
//! shared test process); these cover everything around it. State lives under
//! `$HOME/.olorin/palantir`, so each test isolates `HOME` to a tmpdir.

use olorin::palantir::daemon::{
    is_alive, read_pid, remove_pid, state_dir, status, stop, watcher_name, write_pid,
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
fn status_and_stop_handle_a_recorded_then_missing_watcher() {
    let _h = IsoHome::new("life");
    write_pid("w").unwrap(); // this process is the "daemon" for the test
    write_snapshot("w", "/p/log", "iso8601", Some(3), None, 9);
    assert_eq!(status(Some("w")), 0, "status of a recorded watcher succeeds");
    // Stopping a ghost (no pidfile) is an error and is idempotent-safe.
    assert_eq!(stop(Some("ghost")), 1);
}
