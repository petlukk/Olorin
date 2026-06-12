//! Robustness wave three — server abuse.
//!
//! The web server (std::net, thread-per-connection) is reachable on the
//! network when `OLORIN_BIND` is non-loopback (the wifi/Pi mode) — every
//! request runs through `AuthGate::authorized` *before* dispatch, so the auth
//! parser is attacker-reachable pre-authentication. Oracle here is a
//! property: *no attacker-controlled request head may panic a connection
//! thread, and only the configured token may authorize.*
//!
//! S3 (auth-parser panic on a non-UTF-8-boundary header slice) is FIXED — see
//! `s3_*` below. S1 (Content-Length eager allocation → memory amplification)
//! is FIXED — `read_body` now grows with the bytes that actually arrive (see
//! `s1_*`). S2 (unbounded thread-per-connection) is FIXED by the accept-loop
//! concurrency cap (`OLORIN_MAX_CONN`); the loop itself isn't unit-reachable,
//! so it's verified by inspection + the Pi gate.

use olorin::interface::server_auth::AuthGate;
use olorin::interface::server_http::read_body;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

const TOKEN: &str = "wave3-secret-token";

/// Build a token-enforcing gate without racing sibling tests on the global
/// env var: this whole file's env-dependent assertions live in one test.
fn enforced_gate() -> AuthGate {
    std::env::set_var("OLORIN_AUTH_TOKEN", TOKEN);
    let gate = AuthGate::resolve("0.0.0.0").expect("token set → resolves");
    assert!(!gate.is_open());
    gate
}

#[test]
fn s3_multibyte_header_lines_do_not_panic_auth() {
    // FINDING S3 (FIXED): `bearer_token`/`cookie_token` sliced fixed byte
    // ranges (`line[..14]`, `line[..7]`, `val[..7]`) on attacker-controlled
    // header lines. A multibyte char straddling byte 7 or 14 made the slice
    // panic the connection thread — pre-auth, on an exposed server. The parser
    // must now treat such a request as simply unauthorized, never panic.
    let gate = enforced_gate();

    // Each line places a 2-byte char (`ñ` = C3 B1) so the byte boundary at 7
    // and 14 lands mid-char — the historical panic points.
    let malicious = [
        "GET / HTTP/1.1\r\nabcdefñ: x\r\n\r\n",                 // cookie-token line[..7]
        "GET / HTTP/1.1\r\naaaaaaaaaaaaañ: x\r\n\r\n",          // bearer-token line[..14]
        "GET / HTTP/1.1\r\nCookie: ñ\r\n\r\n",
        "GET / HTTP/1.1\r\nAuthorizatioñ: Bearer x\r\n\r\n",
        "GET /?token=ñ HTTP/1.1\r\nHost: x\r\n\r\n",
        "\u{0}\u{0}\u{0}ñ\r\n\r\n",                              // garbage head
    ];
    for req in malicious {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gate.authorized(req)
        }));
        assert!(
            outcome.is_ok(),
            "auth parser panicked on a malformed header (S3 regression): {req:?}"
        );
        // And of course none of these garbage requests authorize.
        assert!(!outcome.unwrap(), "garbage request must not authorize: {req:?}");
    }

    // Sanity: the real token still works after the fix (boundary-safe prefix
    // checks must not change behaviour for valid ASCII headers).
    assert!(gate.authorized(&format!(
        "GET / HTTP/1.1\r\nAuthorization: Bearer {TOKEN}\r\n\r\n"
    )));
    assert!(gate.authorized(&format!(
        "GET / HTTP/1.1\r\nCookie: olorin_auth={TOKEN}\r\n\r\n"
    )));
    assert!(!gate.authorized(
        "GET / HTTP/1.1\r\nAuthorization: Bearer wrong\r\n\r\n"
    ));

    std::env::remove_var("OLORIN_AUTH_TOKEN");
}

#[test]
fn s3_open_gate_also_survives_malformed_heads() {
    // A loopback (open) gate authorizes everything and reads no env — but it
    // must still never panic on a malformed head (the bootstrap_cookie path
    // also parses the query string).
    let gate = AuthGate::resolve("127.0.0.1").unwrap();
    for req in ["\u{0}ñ", "GET", "", "GET /?token=ñ HTTP/1.1\r\n\r\n", "ñ: ñ\r\n\r\n"] {
        let a = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gate.authorized(req)));
        assert!(a.is_ok(), "open gate panicked on {req:?}");
        assert!(a.unwrap(), "open gate authorizes everything");
        let c = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gate.bootstrap_cookie(req)));
        assert!(c.is_ok(), "bootstrap_cookie panicked on {req:?}");
    }
}

/// Drive `read_body` over a real loopback socket: write `request`, close the
/// write half, and return what the server side reads. Mirrors the header read
/// in `handle_connection` (read until CRLFCRLF) before calling `read_body`.
fn body_over_socket(request: &[u8]) -> Vec<u8> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();

    let mut client = TcpStream::connect(addr).unwrap();
    client.write_all(request).unwrap();
    client.shutdown(std::net::Shutdown::Write).unwrap(); // signal end-of-body

    let (mut srv, _) = listener.accept().unwrap();
    srv.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();

    let mut buf = [0u8; 8192];
    let mut n = 0;
    loop {
        match srv.read(&mut buf[n..]) {
            Ok(0) | Err(_) => break,
            Ok(r) => n += r,
        }
        if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") || n == buf.len() {
            break;
        }
    }
    let req = std::str::from_utf8(&buf[..n]).unwrap();
    read_body(&mut srv, req, &buf, n)
}

#[test]
fn s1_lying_content_length_does_not_force_a_large_read() {
    // FINDING S1 (FIXED): a request declaring a 128 MB body but sending only a
    // few bytes must NOT allocate 128 MB — `read_body` grows with the bytes
    // that actually arrive and stops at EOF. Returns just the 5 sent bytes.
    let req = b"POST /api/x HTTP/1.1\r\nContent-Length: 134217728\r\n\r\nHELLO";
    let body = body_over_socket(req);
    assert_eq!(body, b"HELLO", "only the bytes that actually arrived");
    assert!(body.len() < 1024, "no eager allocation to the declared length");
}

#[test]
fn s1_full_body_still_read_completely() {
    // The fix must not under-read a legitimate body: declared length matches
    // what's sent, so the whole body comes back intact.
    let req = b"POST /api/x HTTP/1.1\r\nContent-Length: 11\r\n\r\nhello world";
    let body = body_over_socket(req);
    assert_eq!(body, b"hello world");
}

#[test]
fn s1_over_cap_content_length_is_rejected() {
    // The existing OLORIN_MAX_UPLOAD cap still rejects an over-cap declared
    // length outright (returns empty), before reading anything.
    let huge = format!(
        "POST /api/x HTTP/1.1\r\nContent-Length: 999999999999\r\n\r\n"
    );
    let body = body_over_socket(huge.as_bytes());
    assert!(body.is_empty(), "over-cap Content-Length must be refused");
}
