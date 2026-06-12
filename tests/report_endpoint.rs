//! `/api/report` over a real localhost socket pair (same harness as the
//! term-WS tests): a valid multi-file payload returns one self-contained
//! HTML attachment from the deterministic pipeline; bad payloads fail
//! closed with a 400. No DispatchContext, no model.

use olorin::interface::server_analyze::handle_report;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

fn socket_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

fn b64(data: &[u8]) -> String {
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let v = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(A[(v >> 18) as usize & 63] as char);
        out.push(A[(v >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { A[(v >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { A[v as usize & 63] as char } else { '=' });
    }
    out
}

/// Run one request through handle_report and return the raw response.
fn roundtrip(body: &str) -> String {
    olorin::kernels::ffi::init().unwrap();
    let (mut client, mut server) = socket_pair();
    let request = format!(
        "POST /api/report HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len(),
    );
    client.write_all(request.as_bytes()).unwrap();
    client.flush().unwrap();

    // Mimic the routing layer: one read for the head, then the handler.
    let mut buf = vec![0u8; 64 * 1024];
    let n = server.read(&mut buf).unwrap();
    let req = String::from_utf8_lossy(&buf[..n]).into_owned();
    handle_report(&mut server, &req, &buf[..n], n);
    drop(server);

    let mut response = String::new();
    client.read_to_string(&mut response).unwrap();
    response
}

#[test]
fn valid_payload_returns_html_attachment() {
    let log = {
        let mut s = String::new();
        for m in 0..120 {
            s.push_str(&format!("2026-06-12T08:{:02}:00 INFO ok\n", m % 60));
        }
        s
    };
    let csv = "time,event\n2026-06-12T08:10:00,deploy\n2026-06-12T08:40:00,deploy\n\
               2026-06-12T09:10:00,deploy\n";
    let body = format!(
        "{{\"files\":[{{\"name\":\"app.log\",\"b64\":\"{}\"}},\
         {{\"name\":\"deploys.csv\",\"b64\":\"{}\"}}]}}",
        b64(log.as_bytes()), b64(csv.as_bytes()),
    );
    let resp = roundtrip(&body);
    assert!(resp.starts_with("HTTP/1.1 200 OK"), "status: {}", &resp[..60.min(resp.len())]);
    assert!(resp.contains("Content-Disposition: attachment"), "attachment header");
    assert!(resp.contains("<!DOCTYPE html>"), "HTML body");
    assert!(resp.contains("via eatime"), "per-file section present");
    assert!(!resp.contains("<script"), "no JS in the artifact");
}

#[test]
fn bad_json_fails_closed_with_400() {
    let resp = roundtrip("{not json");
    assert!(resp.starts_with("HTTP/1.1 400"), "status: {}", &resp[..60.min(resp.len())]);
}

#[test]
fn empty_file_list_fails_closed_with_400() {
    let resp = roundtrip("{\"files\":[]}");
    assert!(resp.starts_with("HTTP/1.1 400"), "status: {}", &resp[..60.min(resp.len())]);
}
