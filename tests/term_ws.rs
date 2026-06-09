//! Integration tests for the WebSocket terminal handler: clean-disconnect
//! teardown (no thread/session leak — review fix #1) and the blocked-input
//! feedback frame (review fix #5). Drives handle_term_ws over a real localhost
//! socket pair against a real PTY session.

use olorin::interface::pty::PtySession;
use olorin::interface::term_stream::{handle_term_ws, term_sessions};
use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn upgrade_req(id: u32) -> String {
    // Sec-WebSocket-Key is the RFC 6455 example; handle_term_ws only needs it
    // present to produce the 101 accept response.
    format!(
        "GET /api/term/{id}/ws HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\
         Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
}

/// Register a fresh PTY session under `id`; return a connected (client, server)
/// socket pair ready for handle_term_ws to run on the server end.
fn setup(id: u32) -> (TcpStream, TcpStream) {
    olorin::kernels::ffi::init().unwrap();
    let session = PtySession::new(80, 24).expect("open pty");
    term_sessions().lock().unwrap().insert(id, Arc::new(Mutex::new(session)));
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let client = TcpStream::connect(addr).unwrap();
    let (server, _) = listener.accept().unwrap();
    (client, server)
}

/// Build a masked client Text frame (payload must be <= 125 bytes).
fn mask_text_frame(payload: &[u8]) -> Vec<u8> {
    let mask = [0xA5u8, 0x5A, 0x33, 0xCC];
    let mut f = vec![0x81u8, 0x80 | payload.len() as u8];
    f.extend_from_slice(&mask);
    for (i, &b) in payload.iter().enumerate() {
        f.push(b ^ mask[i & 3]);
    }
    f
}

/// Read one server frame's payload as text (server frames are unmasked).
fn read_frame_text(stream: &mut TcpStream) -> std::io::Result<String> {
    let mut h = [0u8; 2];
    stream.read_exact(&mut h)?;
    let mut len = (h[1] & 0x7f) as usize;
    if len == 126 {
        let mut b = [0u8; 2];
        stream.read_exact(&mut b)?;
        len = u16::from_be_bytes(b) as usize;
    } else if len == 127 {
        let mut b = [0u8; 8];
        stream.read_exact(&mut b)?;
        len = u64::from_be_bytes(b) as usize;
    }
    let mut p = vec![0u8; len];
    stream.read_exact(&mut p)?;
    Ok(String::from_utf8_lossy(&p).into_owned())
}

/// Consume the 101 handshake response byte-by-byte, stopping exactly at the
/// header terminator so the first WS frame stays intact in the socket buffer.
fn read_handshake(stream: &mut TcpStream) {
    let mut acc = Vec::new();
    let mut b = [0u8; 1];
    while stream.read_exact(&mut b).is_ok() {
        acc.push(b[0]);
        if acc.ends_with(b"\r\n\r\n") {
            break;
        }
    }
    assert!(String::from_utf8_lossy(&acc).contains("101"), "no 101 upgrade");
}

#[test]
fn ws_clean_disconnect_terminates_and_frees_session() {
    let id = 50001;
    let (mut client, server) = setup(id);
    let req = upgrade_req(id);
    let (tx, rx) = mpsc::channel();
    let h = std::thread::spawn(move || {
        let mut s = server;
        handle_term_ws(&mut s, &req, id);
        let _ = tx.send(());
    });
    client.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    read_handshake(&mut client);
    // Simulate the browser tab closing.
    client.shutdown(Shutdown::Both).unwrap();
    drop(client);
    // The handler must return promptly — a spin/leak would hang this.
    rx.recv_timeout(Duration::from_secs(5))
        .expect("handle_term_ws did not terminate on disconnect");
    h.join().unwrap();
    // Session slot freed (and the bash child SIGTERMed via PtySession::Drop).
    assert!(
        !term_sessions().lock().unwrap().contains_key(&id),
        "session leaked after disconnect"
    );
}

#[test]
fn ws_blocked_input_emits_blocked_frame() {
    let id = 50002;
    let (mut client, server) = setup(id);
    let req = upgrade_req(id);
    let h = std::thread::spawn(move || {
        let mut s = server;
        handle_term_ws(&mut s, &req, id);
    });
    client.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    read_handshake(&mut client);
    // A safety-blocked command (see pty_guard.rs::guard_blocks_rm_rf).
    client.write_all(&mask_text_frame(b"rm -rf /\n")).unwrap();
    let mut saw_blocked = false;
    for _ in 0..400 {
        match read_frame_text(&mut client) {
            Ok(t) if t.contains("\"blocked\"") => {
                saw_blocked = true;
                break;
            }
            Ok(_) => continue,
            Err(_) => break,
        }
    }
    assert!(saw_blocked, "no blocked frame after safety-blocked input");
    client.shutdown(Shutdown::Both).ok();
    h.join().unwrap();
    term_sessions().lock().unwrap().remove(&id);
}
