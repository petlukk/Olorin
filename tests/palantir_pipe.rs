//! Palantír → chat Pipe injection. Every *fresh* daemon snapshot is surfaced to
//! the chat model as a `<recent_observations>` line: a live incident, a just-fired
//! stand-down, or an affirmative all-clear when the watcher is healthy and quiet.
//! A *stale* daemon contributes nothing — its state is unknown.

use olorin::core::router::DispatchContext;
use olorin::palantir::daemon::{recent_observations, write_snapshot};
use olorin::palantir::watch::Alert;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

fn env_lock() -> &'static Mutex<()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
}
struct IsoHome { _g: std::sync::MutexGuard<'static, ()>, old: Option<String>, dir: PathBuf }
impl IsoHome {
    fn new(tag: &str) -> Self {
        let g = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        olorin::kernels::ffi::init().ok(); // ctx construction touches kernels
        let old = std::env::var("HOME").ok();
        let dir = std::env::temp_dir().join(format!("olorin-pipe-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("HOME", &dir);
        Self { _g: g, old, dir }
    }
}
impl Drop for IsoHome {
    fn drop(&mut self) {
        match &self.old { Some(h) => std::env::set_var("HOME", h), None => std::env::remove_var("HOME") }
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Find the one line for a named watcher in the observation set.
fn line_for<'a>(obs: &'a [String], name: &str) -> Option<&'a str> {
    obs.iter().map(String::as_str).find(|l| l.starts_with(&format!("[palantir:{name}]")))
}

// ── recent_observations: incident vs all-clear vs stale gating ──────────────

#[test]
fn fresh_watchers_report_incident_or_all_clear_stale_ones_stay_silent() {
    let _h = IsoHome::new("obs");
    let now = 1_000_000i64;

    // fresh daemon, live incident → the incident line
    write_snapshot("live", "/p/system.log", "iso8601", Some(10),
        Some(&Alert::Confirmed { trigger_at: now - 10, at: now - 30, errors: 114 }), now);
    // fresh daemon, never alerted → affirmative all-clear
    write_snapshot("quiet", "/p/quiet.log", "iso8601", None, None, now);
    // fresh daemon, incident long resolved (>10m) → all-clear, not the old alert
    write_snapshot("recovered", "/p/old.log", "iso8601", None,
        Some(&Alert::Confirmed { trigger_at: 0, at: now - 5000, errors: 9 }), now);
    // stale daemon (snapshot not updated recently) → omitted entirely, even with
    // an alert: we don't know its state, so we must not claim all-clear for it.
    write_snapshot("dead", "/p/dead.log", "iso8601", None,
        Some(&Alert::Anomaly { at: now - 4000, rate: 20, baseline: 0.0 }), now - 4000);

    let obs = recent_observations(now);
    assert_eq!(obs.len(), 3, "three fresh watchers, the dead one silent: {obs:?}");
    assert!(line_for(&obs, "dead").is_none(), "a stale daemon must not surface: {obs:?}");

    let live = line_for(&obs, "live").expect("live incident present");
    assert!(live.contains("CASCADE CONFIRMED") && live.contains("ago)"), "{live}");

    for q in ["quiet", "recovered"] {
        let l = line_for(&obs, q).unwrap_or_else(|| panic!("{q} should report all-clear: {obs:?}"));
        assert!(l.contains("all clear") && l.contains("no active incident"), "{l}");
        assert!(!l.contains("CASCADE"), "a resolved incident must not read as live: {l}");
    }
}

#[test]
fn a_recent_stand_down_surfaces_as_an_all_clear() {
    // A just-fired `clear` is good news (the predicted cascade never came). It
    // surfaces with its own "no cascade" wording — which already reads as an
    // all-clear, never as a live problem.
    let _h = IsoHome::new("clear");
    let now = 3_000_000i64;
    write_snapshot("c", "/p/c.log", "iso8601", Some(10),
        Some(&Alert::Clear { trigger_at: now - 45, window: 45 }), now);
    let obs = recent_observations(now);
    assert_eq!(obs.len(), 1, "the stand-down watcher surfaces: {obs:?}");
    assert!(obs[0].contains("no cascade"),
        "a fresh stand-down keeps its 'no cascade' wording: {:?}", obs[0]);
}

#[test]
fn an_aged_out_stand_down_becomes_a_plain_all_clear() {
    // The stand-down's own message is transient: once it ages past the
    // clear-freshness window the watcher falls back to the plain all-clear (still
    // a daemon we can see and that is healthy), rather than repeating the stale
    // "no cascade" line or going silent. Snapshot stays fresh (updated at
    // `later`) so this isolates the clear-age gate, not the stale-daemon gate.
    let _h = IsoHome::new("clear_aged");
    let now = 3_000_000i64;
    let later = now + 200; // > CLEAR_FRESH_SECS past the stand-down
    write_snapshot("c", "/p/c.log", "iso8601", Some(10),
        Some(&Alert::Clear { trigger_at: now - 45, window: 45 }), later);
    let obs = recent_observations(later);
    assert_eq!(obs.len(), 1, "a fresh daemon still reports: {obs:?}");
    assert!(obs[0].contains("no active incident"), "{:?}", obs[0]);
    assert!(!obs[0].contains("no cascade"), "the stale stand-down message must drop: {:?}", obs[0]);
}

#[test]
fn a_fresh_quiet_watcher_reports_all_clear() {
    let _h = IsoHome::new("quiet");
    let now = 2_000_000i64;
    write_snapshot("a", "/p/a.log", "iso8601", None, None, now);
    let obs = recent_observations(now);
    assert_eq!(obs.len(), 1);
    assert!(obs[0].contains("all clear"), "{:?}", obs[0]);
}

#[test]
fn a_stale_daemon_is_silent() {
    // No fresh daemon at all → nothing, so the chat doesn't claim an all-clear
    // for a watcher that may have crashed.
    let _h = IsoHome::new("stale");
    let now = 2_000_000i64;
    write_snapshot("a", "/p/a.log", "iso8601", None, None, now - 4000);
    assert!(recent_observations(now).is_empty(), "a dead daemon must not report all-clear");
}

// ── router wiring: the block appears for any fresh watcher ───────────────────

#[test]
fn system_prompt_gains_observations_block_for_a_fresh_watcher() {
    let _h = IsoHome::new("wire");
    let ctx = DispatchContext::new(None, None);
    let base = ctx.system_prompt_for_test().to_string();

    // No watcher at all → prompt is exactly the base (no block, no prefill cost).
    assert_eq!(ctx.system_prompt_for_turn(), base, "no watcher must not change the prompt");

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;

    // A fresh quiet watcher → the block appears with an affirmative all-clear, so
    // the chat can say "everything's ok" rather than going generic.
    write_snapshot("system", "/var/log/system.log", "iso8601", Some(10), None, now);
    let quiet = ctx.system_prompt_for_turn();
    assert!(quiet.starts_with(&base), "base prompt preserved");
    assert!(quiet.contains("<recent_observations>") && quiet.contains("</recent_observations>"));
    assert!(quiet.contains("[palantir:system]") && quiet.contains("all clear"), "{quiet}");

    // A live incident on the same watcher → the block now carries the incident.
    write_snapshot("system", "/var/log/system.log", "iso8601", Some(10),
        Some(&Alert::Confirmed { trigger_at: now - 10, at: now - 5, errors: 114 }), now);
    let incident = ctx.system_prompt_for_turn();
    assert!(incident.contains("[palantir:system]") && incident.contains("CASCADE CONFIRMED"), "{incident}");
}
