//! Daemon mode + lifecycle for the logwatch palantír.
//!
//! `--daemon` detaches the watcher from its controlling terminal (double-fork +
//! setsid), so it survives the shell that launched it — including the web-UI
//! terminal, which SIGTERMs its bash session on disconnect. State lives under
//! `~/.olorin/palantir/<name>.{pid,json,log}`:
//!   - `.pid`  — the running daemon's pid (lifecycle).
//!   - `.json` — an atomically-written snapshot (status + last alert; the
//!               contract a future web-UI / Pipe reads).
//!   - `.log`  — daemon stdout/stderr (the stdout sink lands here when detached).
//!
//! `--status` and `--stop` read those. Unix only — palantír is a Linux/Pi ops
//! tool; `--daemon` errors elsewhere.

use crate::palantir::sink::json_escape;
use crate::palantir::watch::Alert;
use std::io;
use std::path::{Path, PathBuf};

/// `~/.olorin/palantir`, created if absent.
pub fn state_dir() -> PathBuf {
    let dir = crate::home_dir().unwrap_or_default().join(".olorin/palantir");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Watcher name: `--name` if given, else the alert file's basename, sanitized to
/// a safe filename stem.
pub fn watcher_name(path: &str, name: Option<&str>) -> String {
    let raw = name.unwrap_or_else(|| path.rsplit(['/', '\\']).next().unwrap_or(path));
    let s: String = raw
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' })
        .collect();
    let s = s.trim_matches('.').to_string();
    if s.is_empty() { "watch".to_string() } else { s }
}

fn pid_path(name: &str) -> PathBuf { state_dir().join(format!("{name}.pid")) }
fn snapshot_path(name: &str) -> PathBuf { state_dir().join(format!("{name}.json")) }
pub fn log_path(name: &str) -> PathBuf { state_dir().join(format!("{name}.log")) }

// ── pidfile ─────────────────────────────────────────────────────────────────

pub fn write_pid(name: &str) -> io::Result<()> {
    std::fs::write(pid_path(name), std::process::id().to_string())
}

pub fn read_pid(name: &str) -> Option<i32> {
    std::fs::read_to_string(pid_path(name)).ok()?.trim().parse().ok()
}

pub fn remove_pid(name: &str) {
    let _ = std::fs::remove_file(pid_path(name));
}

/// Is a process alive? `kill(pid, 0)` probes without signalling.
pub fn is_alive(pid: i32) -> bool {
    #[cfg(unix)]
    unsafe { libc::kill(pid, 0) == 0 }
    #[cfg(not(unix))]
    { let _ = pid; false }
}

/// Refuse to start a second watcher under the same name.
pub fn already_running(name: &str) -> Option<i32> {
    read_pid(name).filter(|&p| is_alive(p))
}

// ── snapshot ────────────────────────────────────────────────────────────────

/// Write the watcher's current state atomically (`.tmp` + rename) for `--status`
/// and any future reader (web-UI / Pipe).
#[allow(clippy::too_many_arguments)]
pub fn write_snapshot(
    name: &str,
    path: &str,
    format: &str,
    lag: Option<i64>,
    last: Option<&Alert>,
    now: i64,
) {
    let lag_json = lag.map(|l| l.to_string()).unwrap_or_else(|| "null".to_string());
    let last_json = match last {
        Some(a) => format!(
            "{{\"kind\":\"{}\",\"severity\":\"{}\",\"message\":\"{}\",\"at_unix\":{}}}",
            a.kind(), a.severity(), json_escape(&a.render()), a.at()
        ),
        None => "null".to_string(),
    };
    let status = if matches!(last, Some(Alert::Confirmed { .. }) | Some(Alert::Anomaly { .. })) {
        "alerting"
    } else {
        "watching"
    };
    let body = format!(
        "{{\"name\":\"{}\",\"path\":\"{}\",\"pid\":{},\"status\":\"{status}\",\
         \"format\":\"{}\",\"lag\":{lag_json},\"updated_at_unix\":{now},\"last_alert\":{last_json}}}\n",
        json_escape(name), json_escape(path), std::process::id(), json_escape(format),
    );
    let final_path = snapshot_path(name);
    let tmp = final_path.with_extension("json.tmp");
    if std::fs::write(&tmp, body.as_bytes()).is_ok() {
        let _ = std::fs::rename(&tmp, &final_path);
    }
}

pub fn remove_snapshot(name: &str) {
    let _ = std::fs::remove_file(snapshot_path(name));
}

// ── daemonize (unix) ────────────────────────────────────────────────────────

/// Detach into the background: double-fork + setsid so the daemon has no
/// controlling terminal and is reparented to init, then redirect stdio to the
/// log file. Only the final grandchild returns; the intermediate parents
/// `exit(0)`, so the launching shell returns immediately.
#[cfg(unix)]
pub fn daemonize(log: &Path) -> io::Result<()> {
    use std::os::unix::io::AsRawFd;
    unsafe {
        // First fork: the launching shell's child. Parent exits → shell returns.
        match libc::fork() {
            n if n < 0 => return Err(io::Error::last_os_error()),
            0 => {}
            _ => std::process::exit(0),
        }
        // New session, no controlling terminal.
        if libc::setsid() < 0 {
            return Err(io::Error::last_os_error());
        }
        // Second fork: the daemon is not a session leader, so it can never
        // re-acquire a controlling tty.
        match libc::fork() {
            n if n < 0 => return Err(io::Error::last_os_error()),
            0 => {}
            _ => std::process::exit(0),
        }
        // Redirect stdio: stdin from /dev/null, stdout+stderr to the log. dup2
        // duplicates the fds, so dropping these handles afterwards is safe.
        let logf = std::fs::OpenOptions::new().create(true).append(true).open(log)?;
        let devnull = std::fs::OpenOptions::new().read(true).open("/dev/null")?;
        libc::dup2(devnull.as_raw_fd(), 0);
        libc::dup2(logf.as_raw_fd(), 1);
        libc::dup2(logf.as_raw_fd(), 2);
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn daemonize(_log: &Path) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "--daemon is only supported on Unix"))
}

// ── status / stop ───────────────────────────────────────────────────────────

/// Print the state of one watcher (by name) or all watchers found in the state
/// dir. Returns the process exit code.
pub fn status(name: Option<&str>) -> i32 {
    let names = match name {
        Some(n) => vec![n.to_string()],
        None => discover_names(),
    };
    if names.is_empty() {
        println!("no palantír watchers (state dir empty)");
        return 0;
    }
    for n in names {
        println!("{}", describe(&n));
    }
    0
}

/// Stop one watcher (by name) or all. SIGTERMs the daemon and removes its
/// pidfile. Returns the exit code.
pub fn stop(name: Option<&str>) -> i32 {
    let names = match name {
        Some(n) => vec![n.to_string()],
        None => discover_names(),
    };
    if names.is_empty() {
        eprintln!("no palantír watcher to stop");
        return 1;
    }
    let mut code = 0;
    for n in &names {
        match read_pid(n) {
            Some(pid) if is_alive(pid) => {
                #[cfg(unix)]
                unsafe { libc::kill(pid, libc::SIGTERM); }
                remove_pid(n);
                remove_snapshot(n);
                println!("stopped palantír '{n}' (pid {pid})");
            }
            _ => {
                eprintln!("palantír '{n}' is not running");
                remove_pid(n);
                remove_snapshot(n);
                code = 1;
            }
        }
    }
    code
}

// ── pipe injection ──────────────────────────────────────────────────────────

/// A watcher's snapshot is fresh enough to trust only if it was updated within
/// this window (a stale snapshot means a dead/hung daemon — don't surface it).
const SNAPSHOT_FRESH_SECS: i64 = 120;
/// Surface alerts from at most this far back, so the chat sees *current*
/// incidents, not last hour's.
const ALERT_FRESH_SECS: i64 = 600;
/// A stand-down (`clear`) surfaces only this briefly — long enough for the chat
/// to announce the all-clear after a prediction resolves, then it ages out so a
/// resolved incident doesn't linger as stale context. Much shorter than
/// `ALERT_FRESH_SECS`: an all-clear is only interesting right after it fires.
const CLEAR_FRESH_SECS: i64 = 90;

/// One-line observations for the chat Pipe: each fresh watcher that has a recent
/// alert — a live incident (up to `ALERT_FRESH_SECS`) or a just-fired stand-down
/// (the all-clear, up to `CLEAR_FRESH_SECS`). Quiet and stale watchers are
/// omitted so the system prompt (and the Pi prefill) stays lean. Text is
/// sanitized for safe injection into the system prompt.
pub fn recent_observations(now: i64) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(state_dir()) else { return out };
    for e in rd.flatten() {
        if e.path().extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = std::fs::read(e.path()) else { continue };
        let Ok(o) = crate::storage::json::parse(&bytes) else { continue };
        if now - o.get_i64("updated_at_unix").unwrap_or(0) > SNAPSHOT_FRESH_SECS {
            continue; // stale daemon
        }
        let Some(la) = o.get_object("last_alert") else { continue }; // quiet watcher
        let at = la.get_i64("at_unix").unwrap_or(0);
        // A live incident surfaces for ALERT_FRESH_SECS; a stand-down surfaces
        // only briefly as an all-clear so the chat can announce the resolution
        // (the badge already flips green via `active_incident`, but the chat must
        // be *told* — otherwise it can warn of a cascade and never report that it
        // cleared). Both age out; an old incident or a stale all-clear is dropped.
        let surface = if la.get_str("kind") == Some("clear") {
            now - at <= CLEAR_FRESH_SECS
        } else {
            active_incident(la, now)
        };
        if !surface {
            continue;
        }
        let name = sanitize(o.get_str("name").unwrap_or("?"));
        let msg = sanitize(la.get_str("message").unwrap_or(""));
        out.push(format!("[palantir:{name}] {msg} ({})", humanize_age(now - at)));
    }
    out.sort();
    out
}

/// JSON array of the LIVE watchers, for the web-UI header badge — one
/// `{name, status, alert}` per running daemon. `status` is `"alerting"` with the
/// message when the last alert is recent, else `"watching"`. Stale/stopped
/// watchers are omitted (the badge only shows what's actually running).
pub fn status_json(now: i64) -> String {
    let mut parts: Vec<String> = Vec::new();
    for name in discover_names() {
        match read_pid(&name) {
            Some(pid) if is_alive(pid) => {}
            _ => continue,
        }
        let (status, alert) = live_state(&name, now);
        let alert_json = match alert {
            Some(m) => format!("\"{}\"", json_escape(&m)),
            None => "null".to_string(),
        };
        parts.push(format!(
            "{{\"name\":\"{}\",\"status\":\"{status}\",\"alert\":{alert_json}}}",
            json_escape(&name)
        ));
    }
    format!("[{}]", parts.join(","))
}

/// A recent `last_alert` is an *active* incident — a red badge, a chat-Pipe
/// injection — for every kind except `clear`. A `clear` is the stand-down that
/// fires when a predicted cascade never arrives: the watcher is back to normal,
/// so it must return to green instead of staying red for the rest of the
/// freshness window. ("alert is recent" ≠ "incident is active".)
fn active_incident(la: &crate::storage::json::Object, now: i64) -> bool {
    now - la.get_i64("at_unix").unwrap_or(0) <= ALERT_FRESH_SECS
        && la.get_str("kind") != Some("clear")
}

/// (status, recent-alert message) for a running watcher from its snapshot.
fn live_state(name: &str, now: i64) -> (&'static str, Option<String>) {
    let Ok(bytes) = std::fs::read(snapshot_path(name)) else { return ("watching", None) };
    let Ok(o) = crate::storage::json::parse(&bytes) else { return ("watching", None) };
    if let Some(la) = o.get_object("last_alert") {
        if active_incident(la, now) {
            return ("alerting", la.get_str("message").map(sanitize));
        }
    }
    ("watching", None)
}

fn humanize_age(secs: i64) -> String {
    if secs < 0 { "just now".to_string() }
    else if secs < 60 { format!("{secs}s ago") }
    else { format!("{}m ago", secs / 60) }
}

/// Strip anything that could break or forge the injected block: angle brackets
/// (no fake tags), control chars, and an overlong line.
fn sanitize(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '<' && *c != '>' && !c.is_control())
        .take(200)
        .collect()
}

fn discover_names() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(state_dir()) {
        for e in rd.flatten() {
            if let Some(stem) = e.path().file_stem().and_then(|s| s.to_str()) {
                if e.path().extension().and_then(|x| x.to_str()) == Some("pid") {
                    out.push(stem.to_string());
                }
            }
        }
    }
    out.sort();
    out
}

fn describe(name: &str) -> String {
    let alive = read_pid(name).filter(|&p| is_alive(p));
    let run = match alive {
        Some(p) => format!("running pid {p}"),
        None => "stopped".to_string(),
    };
    // Pull the human summary from the snapshot, if any.
    let detail = std::fs::read(snapshot_path(name))
        .ok()
        .and_then(|b| crate::storage::json::parse(&b).ok())
        .map(|o| {
            let path = o.get_str("path").unwrap_or("?").to_string();
            let status = o.get_str("status").unwrap_or("?").to_string();
            let last = o
                .get_object("last_alert")
                .and_then(|la| la.get_str("message").map(str::to_string))
                .unwrap_or_else(|| "(no alert yet)".to_string());
            format!("{path}  [{status}]  last: {last}")
        })
        .unwrap_or_else(|| "(no snapshot)".to_string());
    format!("palantír '{name}'  [{run}]  {detail}")
}
