//! Palantír v2 — alert sinks (`--notify stdout|webhook:URL|exec:CMD`).

use olorin::palantir::sink::{webhook_body, Sink};
use olorin::palantir::watch::Alert;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

fn confirmed() -> Alert {
    Alert::Confirmed { trigger_at: 100, at: 110, errors: 5 }
}

// ── parsing ─────────────────────────────────────────────────────────────────

#[test]
fn parse_accepts_each_sink_form() {
    assert!(matches!(Sink::parse("stdout"), Ok(Sink::Stdout)));
    assert!(matches!(Sink::parse("webhook:https://x/y"), Ok(Sink::Webhook(u)) if u == "https://x/y"));
    assert!(matches!(Sink::parse("exec:notify-send hi"), Ok(Sink::Exec(c)) if c == "notify-send hi"));
}

#[test]
fn parse_rejects_bad_specs() {
    assert!(Sink::parse("webhook:").is_err(), "empty URL must be rejected");
    assert!(Sink::parse("exec:").is_err(), "empty command must be rejected");
    assert!(Sink::parse("pigeon").is_err(), "unknown sink must be rejected");
}

// ── payload + severity ──────────────────────────────────────────────────────

#[test]
fn webhook_body_is_valid_json_with_routing_fields() {
    let body = webhook_body(&confirmed());
    assert!(body.contains("\"text\":\""), "has a Slack-compatible text field: {body}");
    assert!(body.contains("\"source\":\"olorin-palantir\""));
    assert!(body.contains("\"kind\":\"confirmed\""));
    assert!(body.contains("\"severity\":\"critical\""));
    // The rendered text carries an em dash and emoji — must not break the JSON.
    assert!(body.matches('"').count() % 2 == 0, "unbalanced quotes: {body}");
}

#[test]
fn kind_and_severity_map_per_variant() {
    assert_eq!(Alert::Predicted { at: 0, eta: None, window: 45 }.kind(), "predicted");
    assert_eq!(Alert::Predicted { at: 0, eta: None, window: 45 }.severity(), "warning");
    assert_eq!(confirmed().severity(), "critical");
    assert_eq!(Alert::Clear { trigger_at: 0, window: 45 }.severity(), "info");
}

// ── exec sink ───────────────────────────────────────────────────────────────

#[test]
fn exec_sink_runs_the_command_with_alert_in_env() {
    let out = format!("/tmp/olorin_palantir_exec_{}.txt", std::process::id());
    let _ = std::fs::remove_file(&out);
    Sink::parse(&format!("exec:printf '%s|%s' \"$PALANTIR_KIND\" \"$PALANTIR_SEVERITY\" > {out}"))
        .unwrap()
        .deliver(&confirmed());
    let got = std::fs::read_to_string(&out).expect("exec sink should have written the file");
    assert_eq!(got, "confirmed|critical");
    let _ = std::fs::remove_file(&out);
}

// ── webhook sink (against a local listener) ─────────────────────────────────

#[test]
fn webhook_sink_posts_the_json_body() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let server = thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = [0u8; 8192];
        let n = sock.read(&mut buf).unwrap();
        let _ = sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
        String::from_utf8_lossy(&buf[..n]).into_owned()
    });

    Sink::parse(&format!("webhook:http://127.0.0.1:{port}/hook"))
        .unwrap()
        .deliver(&confirmed());

    let req = server.join().unwrap();
    assert!(req.starts_with("POST "), "must be a POST: {req}");
    assert!(req.contains("content-type: application/json"));
    assert!(req.contains("\"source\":\"olorin-palantir\""));
    assert!(req.contains("\"severity\":\"critical\""), "body missing in request: {req}");
}
