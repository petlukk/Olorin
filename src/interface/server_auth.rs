//! Network auth gate for the web server.
//!
//! Loopback binds (`127.0.0.1`, `::1`, `localhost`) need no auth — that's the
//! single-user-on-own-machine default. But the moment the server binds to a
//! non-loopback address (`OLORIN_BIND=0.0.0.0`, a LAN IP, …) it is reachable
//! by other hosts, and *every* endpoint dispatches through the full Pipe —
//! including the shell / http / write_file tools. Unauthenticated, that's
//! remote command execution on the host.
//!
//! So a non-loopback bind is **fail-closed**: `OLORIN_AUTH_TOKEN` must be set
//! or the server refuses to start, and every request must then present that
//! token via `Authorization: Bearer <t>`, an `olorin_auth=<t>` cookie, or a
//! `?token=<t>` query param. The query param lets a browser bootstrap the
//! cookie by visiting `/?token=…` once; same-origin `fetch`/`EventSource`/
//! WebSocket then carry the cookie automatically, so the UI needs no changes.

use std::sync::Arc;

/// Resolved auth policy, held for the server's lifetime. Cheap to clone
/// (the token is behind an `Arc`), so each connection thread gets a copy.
#[derive(Clone)]
pub struct AuthGate {
    /// `Some(token)` → every request must present it. `None` → loopback bind,
    /// no auth required.
    token: Option<Arc<String>>,
}

impl AuthGate {
    /// Decide the policy from the bind host + environment. Returns `Err` with
    /// an operator-facing message when a non-loopback bind has no token set —
    /// the caller prints it and exits *before* binding (fail-closed).
    pub fn resolve(bind_host: &str) -> Result<Self, String> {
        if is_loopback(bind_host) {
            return Ok(Self { token: None });
        }
        match std::env::var("OLORIN_AUTH_TOKEN") {
            Ok(t) if !t.is_empty() => Ok(Self { token: Some(Arc::new(t)) }),
            _ => Err(format!(
                "refusing to bind non-loopback address `{bind_host}` without auth.\n        \
                 Set OLORIN_AUTH_TOKEN=<secret> to expose the web UI on the network,\n        \
                 or bind 127.0.0.1 (the default) for local-only use."
            )),
        }
    }

    /// True when no auth is required (loopback bind).
    pub fn is_open(&self) -> bool {
        self.token.is_none()
    }

    /// Whether `request` (raw HTTP request head) carries the required token.
    /// Always true when the gate is open. Accepts if **any** presented
    /// credential matches — `Authorization: Bearer`, an `olorin_auth` cookie,
    /// or a `?token=` query param — each constant-time compared.
    ///
    /// Checking all three (not just the first present) is deliberate: a stale
    /// `olorin_auth` cookie from an earlier session must not shadow a fresh
    /// `?token=` bootstrap, or a browser holding an old cookie could never
    /// re-authenticate by pasting the correct URL (it would keep getting 401
    /// while sending both a wrong cookie and the right query token).
    pub fn authorized(&self, request: &str) -> bool {
        let Some(token) = &self.token else { return true; };
        let want = token.as_bytes();
        matches(bearer_token(request), want)
            || matches(cookie_token(request), want)
            || matches(query_token(request), want)
    }

    /// `Set-Cookie` header *value* to persist the token in the browser, when a
    /// valid `?token=` was presented on this request (used on the `GET /`
    /// response). `None` when the gate is open or the query token is absent /
    /// wrong.
    pub fn bootstrap_cookie(&self, request: &str) -> Option<String> {
        let token = self.token.as_ref()?;
        let q = query_token(request)?;
        if ct_eq(q.as_bytes(), token.as_bytes()) {
            // Reflect the configured token (already proven equal), not raw input.
            Some(format!("olorin_auth={token}; HttpOnly; SameSite=Strict; Path=/"))
        } else {
            None
        }
    }
}

/// Loopback if `localhost` or an IP literal whose `is_loopback()` holds
/// (`127.0.0.0/8`, `::1`). Anything unparseable is treated as non-loopback so
/// the gate fails closed.
fn is_loopback(host: &str) -> bool {
    let h = host.trim();
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    // Strip brackets from IPv6 literals like `[::1]`.
    let h = h.strip_prefix('[').and_then(|s| s.strip_suffix(']')).unwrap_or(h);
    h.parse::<std::net::IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
}

/// Constant-time check that a presented credential (if any) equals `want`.
fn matches(presented: Option<String>, want: &[u8]) -> bool {
    match presented {
        Some(p) => ct_eq(p.as_bytes(), want),
        None => false,
    }
}

fn bearer_token(request: &str) -> Option<String> {
    for line in request.lines() {
        if line.len() >= 14 && line[..14].eq_ignore_ascii_case("authorization:") {
            let val = line[14..].trim();
            // Case-insensitive "Bearer " prefix.
            if val.len() >= 7 && val[..7].eq_ignore_ascii_case("bearer ") {
                let t = val[7..].trim();
                if !t.is_empty() {
                    return Some(t.to_string());
                }
            }
        }
    }
    None
}

fn cookie_token(request: &str) -> Option<String> {
    for line in request.lines() {
        if line.len() >= 7 && line[..7].eq_ignore_ascii_case("cookie:") {
            for crumb in line[7..].split(';') {
                if let Some(v) = crumb.trim().strip_prefix("olorin_auth=") {
                    if !v.is_empty() {
                        return Some(v.to_string());
                    }
                }
            }
        }
    }
    None
}

fn query_token(request: &str) -> Option<String> {
    let first = request.lines().next()?;
    let url = first.split_whitespace().nth(1)?; // METHOD <url> HTTP/1.1
    let query = url.split('?').nth(1)?;
    for pair in query.split('&') {
        if let Some(v) = pair.strip_prefix("token=") {
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Length-checked constant-time byte comparison. The length check leaks token
/// length only, which is not secret-bearing; the byte loop is data-independent.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}
