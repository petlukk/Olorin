//! MEDIUM finding regression: the web server must fail-closed when bound
//! off-loopback. A non-loopback bind without OLORIN_AUTH_TOKEN must refuse to
//! start, and once a token is set every request must present it (Bearer
//! header, cookie, or `?token=` bootstrap) before any dispatch.

use olorin::interface::server_auth::AuthGate;

const TOKEN: &str = "s3cret-token-xyz";

/// Loopback binds are open — the single-user-on-own-machine default. This
/// path never reads the env var, so it's race-free against the env test below.
#[test]
fn loopback_binds_are_open() {
    for host in ["127.0.0.1", "127.0.0.5", "::1", "[::1]", "localhost", "LOCALHOST"] {
        let gate = AuthGate::resolve(host).unwrap_or_else(|e| panic!("{host} should be open: {e}"));
        assert!(gate.is_open(), "{host} must be open");
        // An open gate authorizes everything, including a bare request.
        assert!(gate.authorized("GET / HTTP/1.1\r\nHost: x\r\n\r\n"), "{host}");
    }
}

/// All env-var-dependent assertions in one test so the process-global
/// OLORIN_AUTH_TOKEN mutation can't race a sibling.
#[test]
fn non_loopback_is_fail_closed_then_token_enforced() {
    // ── No token → refuse to start on every non-loopback shape ──────────────
    std::env::remove_var("OLORIN_AUTH_TOKEN");
    for host in ["0.0.0.0", "192.168.1.5", "10.0.0.1", "garbage-host"] {
        assert!(
            AuthGate::resolve(host).is_err(),
            "{host} without a token must be refused"
        );
    }

    // ── Token set → gate constructed, not open ──────────────────────────────
    std::env::set_var("OLORIN_AUTH_TOKEN", TOKEN);
    let gate = AuthGate::resolve("0.0.0.0").expect("token present → resolves");
    assert!(!gate.is_open());

    // Missing / wrong credentials are rejected on every channel.
    assert!(!gate.authorized("GET / HTTP/1.1\r\nHost: x\r\n\r\n"), "no creds");
    assert!(!gate.authorized("GET / HTTP/1.1\r\nAuthorization: Bearer nope\r\n\r\n"));
    assert!(!gate.authorized("GET / HTTP/1.1\r\nCookie: olorin_auth=nope\r\n\r\n"));
    assert!(!gate.authorized("GET /?token=nope HTTP/1.1\r\nHost: x\r\n\r\n"));
    // Near-miss (prefix of the real token) must not pass.
    assert!(!gate.authorized("GET / HTTP/1.1\r\nAuthorization: Bearer s3cret-token-xy\r\n\r\n"));

    // Correct token accepted via each accepted channel.
    assert!(gate.authorized(&format!("POST /api/generate HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\n\r\n")));
    assert!(gate.authorized(&format!("GET / HTTP/1.1\r\nAuthorization: bearer {TOKEN}\r\n\r\n")), "case-insensitive scheme");
    assert!(gate.authorized(&format!("GET / HTTP/1.1\r\nCookie: theme=dark; olorin_auth={TOKEN}\r\n\r\n")));
    assert!(gate.authorized(&format!("GET /?token={TOKEN} HTTP/1.1\r\nHost: x\r\n\r\n")));

    // ── Browser bootstrap cookie: only for a valid query token ──────────────
    let cookie = gate
        .bootstrap_cookie(&format!("GET /?token={TOKEN} HTTP/1.1\r\nHost: x\r\n\r\n"))
        .expect("valid query token yields a Set-Cookie value");
    assert!(cookie.contains(&format!("olorin_auth={TOKEN}")));
    assert!(cookie.contains("HttpOnly") && cookie.contains("SameSite=Strict"));
    assert!(gate.bootstrap_cookie("GET /?token=wrong HTTP/1.1\r\n\r\n").is_none());
    assert!(gate.bootstrap_cookie("GET / HTTP/1.1\r\n\r\n").is_none(), "no query token → no cookie");

    std::env::remove_var("OLORIN_AUTH_TOKEN");
}

/// An open (loopback) gate never offers a bootstrap cookie and authorizes
/// even when a `?token=` is present but no token is configured.
#[test]
fn open_gate_never_sets_cookie() {
    let gate = AuthGate::resolve("127.0.0.1").unwrap();
    assert!(gate.bootstrap_cookie("GET /?token=anything HTTP/1.1\r\n\r\n").is_none());
}
