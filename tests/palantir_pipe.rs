//! Palantír → chat Pipe injection. A live incident in a daemon snapshot is
//! surfaced to the chat model as a `<recent_observations>` block; quiet and
//! stale watchers contribute nothing.

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

// ── recent_observations: the freshness/quiet gating ─────────────────────────

#[test]
fn surfaces_only_fresh_watchers_with_a_recent_alert() {
    let _h = IsoHome::new("obs");
    let now = 1_000_000i64;

    // fresh daemon, recent alert → surfaced
    write_snapshot("live", "/p/system.log", "iso8601", Some(10),
        Some(&Alert::Confirmed { trigger_at: now - 10, at: now - 30, errors: 114 }), now);
    // fresh daemon, NO alert → omitted
    write_snapshot("quiet", "/p/quiet.log", "iso8601", None, None, now);
    // alert too old (>10m) → omitted
    write_snapshot("old", "/p/old.log", "iso8601", None,
        Some(&Alert::Confirmed { trigger_at: 0, at: now - 5000, errors: 9 }), now);
    // stale daemon (snapshot not updated recently) → omitted even with an alert
    write_snapshot("dead", "/p/dead.log", "iso8601", None,
        Some(&Alert::Anomaly { at: now - 4000, rate: 20, baseline: 0.0 }), now - 4000);

    let obs = recent_observations(now);
    assert_eq!(obs.len(), 1, "only the live incident should surface: {obs:?}");
    assert!(obs[0].starts_with("[palantir:live]"), "{:?}", obs[0]);
    assert!(obs[0].contains("CASCADE CONFIRMED"));
    assert!(obs[0].contains("ago)"));
}

#[test]
fn a_cleared_stand_down_is_not_surfaced_as_a_live_incident() {
    // A recent `clear` is good news (the predicted cascade never came), not an
    // incident — the chat must not be told there's a live problem.
    let _h = IsoHome::new("clear");
    let now = 3_000_000i64;
    write_snapshot("c", "/p/c.log", "iso8601", Some(10),
        Some(&Alert::Clear { trigger_at: now - 45, window: 45 }), now);
    assert!(recent_observations(now).is_empty(),
        "a cleared cascade must not surface as a live incident");
}

#[test]
fn nothing_to_surface_when_all_quiet() {
    let _h = IsoHome::new("quiet");
    let now = 2_000_000i64;
    write_snapshot("a", "/p/a.log", "iso8601", None, None, now);
    assert!(recent_observations(now).is_empty());
}

// ── router wiring: the block is appended only during an incident ─────────────

#[test]
fn system_prompt_gains_observations_block_during_an_incident() {
    let _h = IsoHome::new("wire");
    let ctx = DispatchContext::new(None, None);
    let base = ctx.system_prompt_for_test().to_string();

    // No watcher → prompt is exactly the base (no block, no Pi prefill cost).
    assert_eq!(ctx.system_prompt_for_turn(), base, "quiet must not change the prompt");

    // A live incident → the block is appended.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    write_snapshot("system", "/var/log/system.log", "iso8601", Some(10),
        Some(&Alert::Confirmed { trigger_at: now - 10, at: now - 5, errors: 114 }), now);

    let turn = ctx.system_prompt_for_turn();
    assert!(turn.starts_with(&base), "base prompt preserved");
    assert!(turn.contains("<recent_observations>"), "block opened: {turn}");
    assert!(turn.contains("[palantir:system]"));
    assert!(turn.contains("</recent_observations>"), "block closed");
}
