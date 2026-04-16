//! WhatsApp bridge — subprocess communicating via JSONL.

use std::sync::{Arc, Mutex};
use crate::core::router::DispatchContext;
use crate::interface::exec;
use crate::interface::server::{extract_json_string, escape_json};

/// Start the WhatsApp bridge subprocess and run the JSONL message loop.
pub fn run_whatsapp(model_arg: Option<&str>, draft_arg: Option<&str>, draft_k: Option<usize>) {
    let bridge_path = find_bridge();

    let home        = std::env::var("HOME").unwrap_or_default();
    let session_dir = format!("{home}/.olorin/wa_session");
    std::fs::create_dir_all(&session_dir).ok();

    let child = match exec::spawn(&[&bridge_path, "--session-dir", &session_dir]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[olorin] failed to start WhatsApp bridge: {e}");
            std::process::exit(1);
        }
    };

    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let ctx     = Arc::new(Mutex::new(DispatchContext::new(api_key, model_arg, draft_arg, draft_k)));

    eprintln!("[olorin] WhatsApp bridge started (pid={})", child.pid);
    eprintln!("[olorin] Waiting for bridge connection...");

    // Pull fds out before forgetting Child so Drop doesn't close them
    let pid       = child.pid;
    let stdout_fd = child.stdout_fd;
    let stdin_fd  = child.stdin_fd;
    std::mem::forget(child);

    wa_message_loop(stdout_fd, stdin_fd, pid, ctx);
}

fn wa_message_loop(
    stdout_fd: i32,
    stdin_fd:  i32,
    pid:       i32,
    ctx:       Arc<Mutex<DispatchContext>>,
) {
    let mut line_buf = String::new();
    let mut byte     = [0u8; 1];

    loop {
        // Read one line from bridge stdout
        line_buf.clear();
        loop {
            let n = unsafe {
                libc::read(stdout_fd, byte.as_mut_ptr() as *mut libc::c_void, 1)
            };
            if n <= 0 {
                eprintln!("[olorin] Bridge stdout closed.");
                unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) };
                unsafe { libc::close(stdout_fd); libc::close(stdin_fd); }
                return;
            }
            if byte[0] == b'\n' { break; }
            line_buf.push(byte[0] as char);
        }

        if line_buf.is_empty() { continue; }

        let msg_type = extract_json_string(&line_buf, "type").unwrap_or_default();
        match msg_type.as_str() {
            "connected" => {
                eprintln!("[olorin] WhatsApp connected!");
            }
            "message" => {
                let text = extract_json_string(&line_buf, "text").unwrap_or_default();
                let jid  = extract_json_string(&line_buf, "jid").unwrap_or_default();
                if text.is_empty() || jid.is_empty() { continue; }

                let response = {
                    let mut guard = ctx.lock().unwrap();
                    guard.dispatch(&text)
                };

                let reply_text = escape_json(&response.text);
                let reply_jid  = escape_json(&jid);
                let reply = format!(
                    "{{\"type\":\"send\",\"jid\":\"{reply_jid}\",\"text\":\"{reply_text}\"}}\n"
                );
                let bytes = reply.as_bytes();
                let mut written = 0;
                while written < bytes.len() {
                    let n = unsafe {
                        libc::write(
                            stdin_fd,
                            bytes[written..].as_ptr() as *const libc::c_void,
                            bytes.len() - written,
                        )
                    };
                    if n <= 0 { break; }
                    written += n as usize;
                }
            }
            _ => {}
        }
    }
}

/// Placeholder for streaming teleport flow (QR display + bridge lifecycle).
/// Task 4 will replace this with the real implementation.
pub fn teleport_loop_streaming(
    _ctx: &mut crate::core::router::DispatchContext,
    tx: std::sync::mpsc::Sender<crate::core::router::StreamEvent>,
) {
    let msg = "WhatsApp bridge not available.".to_string();
    let _ = tx.send(crate::core::router::StreamEvent::Token(msg.clone()));
    let _ = tx.send(crate::core::router::StreamEvent::Done { full_text: msg });
}

fn find_bridge() -> String {
    if let Ok(p) = std::env::var("OLORIN_BRIDGE") {
        return p;
    }
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.parent()
            .map(|p| p.join("bridge/wa-bridge"))
            .unwrap_or_default();
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "bridge/wa-bridge".to_string()
}
