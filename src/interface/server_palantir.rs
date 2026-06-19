//! Palantír watcher start/stop POST handlers — split out of `server.rs` to keep
//! it under the 500-line cap. Lets the web-UI badge launch and stop logwatch
//! daemons without dropping to the CLI.
//!
//! Security: starting a watcher from the web turns a localhost ops tool into a
//! remotely-reachable capability when the server is bound non-loopback. The
//! endpoints inherit the server's token gate; two further limits keep the
//! surface tight:
//!   - the watched path is validated by [`resolve_watch_path`] ($HOME, /tmp, and
//!     read-only /var/log; `..`, symlink-escape, and sensitive subtrees refused);
//!   - no `--notify` sinks are accepted — `exec:` would be RCE and `webhook:`
//!     SSRF. The snapshot/badge is the only feedback channel from the web.

use crate::core::path_guard::resolve_watch_path;
use crate::interface::server::serve_json;
use crate::interface::server_http::read_body;
use crate::palantir::daemon;
use crate::palantir::sink::json_escape;
use crate::storage::json;

/// A validated watch request: everything needed to build the daemon argv, with
/// every rejection (sinks, bad path, bad sensitivity) already applied. Pure
/// aside from the path canonicalize inside [`resolve_watch_path`] — so the
/// security decisions are unit-testable without a socket or a real spawn.
#[derive(Debug)]
pub struct WatchPlan {
    /// Canonical, allowlisted watch target.
    pub path: String,
    /// Sanitized watcher name (matches what the badge will show).
    pub name: String,
    /// `low|med|high` if explicitly set, else `None` (daemon defaults to medium).
    pub sensitivity: Option<String>,
}

/// Parse and validate a `/api/palantir/watch` body. `Err` carries a
/// user-facing refusal message; `Ok` is ready to spawn.
pub fn plan_watch(body: &[u8]) -> Result<WatchPlan, String> {
    let obj = json::parse(body).map_err(|_| "invalid JSON body".to_string())?;

    // Sinks are CLI-only: exec: is arbitrary command execution, webhook: is SSRF.
    if obj.get_str("notify").is_some() || obj.get_str("sink").is_some() {
        return Err("alert sinks cannot be set from the web UI".into());
    }

    let path = match obj.get_str("path").map(str::trim) {
        Some(p) if !p.is_empty() => p,
        _ => return Err("missing \"path\"".into()),
    };
    let resolved = resolve_watch_path(path).map_err(|e| e.refusal_message())?;
    let path = resolved.to_string_lossy().into_owned();

    // Sensitivity is optional; default (Medium) means omitting the flag.
    let sensitivity = match obj.get_str("sensitivity") {
        Some(s) if matches!(s, "low" | "med" | "high") => Some(s.to_string()),
        Some(_) => return Err("sensitivity must be low|med|high".into()),
        None => None,
    };

    let name = daemon::watcher_name(&path, obj.get_str("name"));
    Ok(WatchPlan { path, name, sensitivity })
}

/// `POST /api/palantir/watch` — start a logwatch daemon on a file.
/// Body: `{"path":"…","name"?:"…","sensitivity"?:"low|med|high"}`.
pub(crate) fn handle_palantir_watch(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
) {
    let body = read_body(stream, req, buf, n);
    let plan = match plan_watch(&body) {
        Ok(p) => p,
        Err(msg) => return err(stream, &msg),
    };
    let exe = match std::env::current_exe() {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => return err(stream, "cannot locate the olorin binary"),
    };

    // `olorin palantir --alert <path> --daemon --name <name> [--sensitivity S]`.
    // The daemon double-forks + setsid, so exec::run returns once the launching
    // child exits; the watcher survives this request and the server itself.
    let mut argv: Vec<&str> =
        vec![&exe, "palantir", "--alert", &plan.path, "--daemon", "--name", &plan.name];
    if let Some(s) = &plan.sensitivity {
        argv.push("--sensitivity");
        argv.push(s);
    }

    match crate::interface::exec::run(&argv) {
        Ok(out) if out.exit_code == 0 => {
            serve_json(stream, &format!(r#"{{"ok":true,"name":"{}"}}"#, json_escape(&plan.name)));
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            err(stream, &format!("watcher failed to start: {}", stderr.trim()));
        }
        Err(e) => err(stream, &format!("spawn failed: {e}")),
    }
}

/// `POST /api/palantir/stop` — stop a watcher by name. Body: `{"name":"…"}`.
pub(crate) fn handle_palantir_stop(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
) {
    let body = read_body(stream, req, buf, n);
    let obj = match json::parse(&body) {
        Ok(o) => o,
        Err(_) => return err(stream, "invalid JSON body"),
    };
    let name = match obj.get_str("name").map(str::trim) {
        Some(nm) if !nm.is_empty() => nm,
        _ => return err(stream, "missing \\\"name\\\""),
    };
    if daemon::stop(Some(name)) == 0 {
        serve_json(stream, r#"{"ok":true}"#);
    } else {
        err(stream, "no such running watcher");
    }
}

fn err(stream: &mut std::net::TcpStream, msg: &str) {
    serve_json(stream, &format!(r#"{{"ok":false,"error":"{}"}}"#, json_escape(msg)));
}
