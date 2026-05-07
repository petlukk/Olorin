//! WhatsApp bridge — subprocess communicating via JSONL.

use crate::core::router::{DispatchContext, Response, StreamEvent};
use crate::interface::spawner::{default_spawner, ChildProcess};
use crate::interface::server::{extract_json_string, escape_json};

/// Sentinel returned by strip_trigger for the /teleport command.
pub const TRIGGER_TELEPORT: &str = "__teleport__";

/// Check if a message matches an Olorin trigger. Returns the stripped text
/// (trigger prefix removed) or None if no trigger matched.
/// Returns TRIGGER_TELEPORT for the /teleport command.
pub fn strip_trigger(text: &str) -> Option<&str> {
    let trimmed = text.trim();

    if trimmed.eq_ignore_ascii_case("/teleport") {
        return Some(TRIGGER_TELEPORT);
    }

    let lower = trimmed.as_bytes();

    // @olorin or !olorin prefix (case-insensitive)
    for prefix in &[b"@olorin " as &[u8], b"!olorin "] {
        if lower.len() > prefix.len()
            && lower[..prefix.len()].eq_ignore_ascii_case(prefix)
        {
            return Some(trimmed[prefix.len()..].trim());
        }
    }

    // "olorin " prefix (case-insensitive, must have space after)
    let olorin = b"olorin ";
    if lower.len() > olorin.len()
        && lower[..olorin.len()].eq_ignore_ascii_case(olorin)
    {
        return Some(trimmed[olorin.len()..].trim());
    }

    None
}

/// Spawn the bridge subprocess. Returns the child on success.
fn spawn_bridge() -> Result<Box<dyn ChildProcess>, String> {
    let bridge_path = find_bridge();
    if !std::path::Path::new(&bridge_path).exists() {
        return Err(format!(
            "WhatsApp bridge not found at '{bridge_path}'. \
             Build it with: cd bridge && go build -o wa-bridge"
        ));
    }

    let home = std::env::var("HOME").unwrap_or_default();
    let session_dir = format!("{home}/.olorin/wa_session");
    std::fs::create_dir_all(&session_dir).ok();

    default_spawner()
        .spawn(&[&bridge_path, "--session-dir", &session_dir])
        .map_err(|e| format!("Failed to start WhatsApp bridge: {e}"))
}

/// Handle a single inbound message from the bridge. Returns true if the
/// loop should exit (i.e., /teleport received).
fn handle_wa_message(
    child: &dyn ChildProcess,
    text: &str,
    jid: &str,
    ctx: &mut DispatchContext,
) -> bool {
    match strip_trigger(text) {
        Some(TRIGGER_TELEPORT) => {
            let reply = format!(
                "{{\"type\":\"send\",\"jid\":\"{}\",\"text\":\"{}\"}}",
                escape_json(jid),
                escape_json("Olorin has returned to local. Goodbye!")
            );
            let _ = child.write_line(&reply);
            std::thread::sleep(std::time::Duration::from_millis(500));
            true
        }
        Some(stripped) => {
            let response = ctx.dispatch(stripped);
            let reply = format!(
                "{{\"type\":\"send\",\"jid\":\"{}\",\"text\":\"{}\"}}",
                escape_json(jid),
                escape_json(&response.text)
            );
            let _ = child.write_line(&reply);
            false
        }
        None => false,
    }
}

/// Clear the teleported flags on context.
fn clear_teleported(ctx: &mut DispatchContext) {
    ctx.teleported = false;
    if let Some(ref flag) = ctx.server_teleported {
        flag.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Set the teleported flags on context.
fn set_teleported(ctx: &mut DispatchContext) {
    ctx.teleported = true;
    if let Some(ref flag) = ctx.server_teleported {
        flag.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Run the teleport loop (non-streaming, for REPL / `dispatch` path).
/// Spawns bridge, processes messages, returns when /teleport is received
/// from WhatsApp or the bridge dies.
pub fn teleport_loop(ctx: &mut DispatchContext) -> Response {
    if ctx.teleported {
        return Response::text(
            "Olorin is already on WhatsApp. Send /teleport there to return."
        );
    }
    let child = match spawn_bridge() {
        Ok(c) => c,
        Err(msg) => return Response::text(msg),
    };

    eprintln!("[olorin] WhatsApp bridge started (pid={})", child.id());
    set_teleported(ctx);

    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        if child.read_line(&mut line_buf).unwrap_or(0) == 0 {
            eprintln!("[olorin] Bridge disconnected.");
            break;
        }
        if line_buf.is_empty() { continue; }

        let msg_type = extract_json_string(&line_buf, "type").unwrap_or_default();
        match msg_type.as_str() {
            "connected" => eprintln!("[olorin] WhatsApp connected!"),
            "qr" => eprintln!("[olorin] QR code — check terminal."),
            "message" => {
                let text = extract_json_string(&line_buf, "text").unwrap_or_default();
                let jid = extract_json_string(&line_buf, "jid").unwrap_or_default();
                if text.is_empty() || jid.is_empty() { continue; }
                if handle_wa_message(&*child, &text, &jid, ctx) { break; }
            }
            _ => {}
        }
    }

    clear_teleported(ctx);
    Response::text("Olorin has returned from WhatsApp.")
}

/// Streaming teleport loop — sends QR code and status as StreamEvents
/// into the chat bubble via SSE, then enters the blocking message loop.
/// Called from `dispatch_streaming` when CMD_TELEPORT is detected.
pub fn teleport_loop_streaming(
    ctx: &mut DispatchContext,
    tx: std::sync::mpsc::Sender<StreamEvent>,
) {
    if ctx.teleported {
        let msg = "Olorin is already on WhatsApp. Send /teleport there to return."
            .to_string();
        let _ = tx.send(StreamEvent::Token(msg.clone()));
        let _ = tx.send(StreamEvent::Done { full_text: msg });
        return;
    }
    let child = match spawn_bridge() {
        Ok(c) => c,
        Err(msg) => {
            let _ = tx.send(StreamEvent::Token(msg.clone()));
            let _ = tx.send(StreamEvent::Done { full_text: msg });
            return;
        }
    };

    eprintln!("[olorin] WhatsApp bridge started (pid={})", child.id());
    set_teleported(ctx);

    let _ = tx.send(StreamEvent::Token(
        "Teleporting to WhatsApp...\n".to_string()
    ));

    let mut line_buf = String::new();
    let connected = false;

    // Phase 1: Wait for QR / connected events, stream them to chat bubble
    loop {
        line_buf.clear();
        if child.read_line(&mut line_buf).unwrap_or(0) == 0 {
            eprintln!("[olorin] Bridge disconnected during handshake.");
            clear_teleported(ctx);
            let msg = "Bridge disconnected before connecting.".to_string();
            let _ = tx.send(StreamEvent::Token(msg.clone()));
            let _ = tx.send(StreamEvent::Done { full_text: msg });
            return;
        }
        if line_buf.is_empty() { continue; }

        let msg_type = extract_json_string(&line_buf, "type").unwrap_or_default();
        match msg_type.as_str() {
            "qr" => {
                if let Some(ascii) = extract_json_string(&line_buf, "ascii") {
                    let _ = tx.send(StreamEvent::Token(ascii));
                } else {
                    let _ = tx.send(StreamEvent::Token(
                        "QR code displayed in terminal. Scan with WhatsApp.\n".to_string()
                    ));
                }
            }
            "connected" => {
                let _ = tx.send(StreamEvent::Token(
                    "Connected! Olorin is now on WhatsApp.\n\
                     Send /teleport there to return.\n".to_string()
                ));
                break;
            }
            "message" if !connected => {
                let _ = tx.send(StreamEvent::Token(
                    "Connected! Olorin is now on WhatsApp.\n\
                     Send /teleport there to return.\n".to_string()
                ));
                let text = extract_json_string(&line_buf, "text").unwrap_or_default();
                let jid = extract_json_string(&line_buf, "jid").unwrap_or_default();
                if !text.is_empty() && !jid.is_empty() {
                    if handle_wa_message(&*child, &text, &jid, ctx) {
                        clear_teleported(ctx);
                        let msg = "Olorin has returned from WhatsApp.".to_string();
                        let _ = tx.send(StreamEvent::Token(msg.clone()));
                        let _ = tx.send(StreamEvent::Done { full_text: msg });
                        return;
                    }
                }
                break;
            }
            _ => {}
        }
    }

    // SSE Done — the chat bubble is complete. The loop continues silently.
    let _ = tx.send(StreamEvent::Done {
        full_text: "Olorin is on WhatsApp.".to_string(),
    });

    // Phase 2: Blocking message loop (no more SSE events)
    loop {
        line_buf.clear();
        if child.read_line(&mut line_buf).unwrap_or(0) == 0 {
            eprintln!("[olorin] Bridge disconnected.");
            break;
        }
        if line_buf.is_empty() { continue; }

        let msg_type = extract_json_string(&line_buf, "type").unwrap_or_default();
        if msg_type == "message" {
            let text = extract_json_string(&line_buf, "text").unwrap_or_default();
            let jid = extract_json_string(&line_buf, "jid").unwrap_or_default();
            eprintln!("[olorin] WA msg: jid={jid} text={text}");
            if text.is_empty() || jid.is_empty() { continue; }
            if handle_wa_message(&*child, &text, &jid, ctx) { break; }
        }
    }

    clear_teleported(ctx);
}

/// Start the WhatsApp bridge as standalone (--whatsapp flag).
pub fn run_whatsapp(model_arg: Option<&str>) {
    let child = match spawn_bridge() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[olorin] failed to start WhatsApp bridge: {e}");
            std::process::exit(1);
        }
    };

    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let ctx = std::sync::Arc::new(std::sync::Mutex::new(
        DispatchContext::new(api_key, model_arg),
    ));

    eprintln!("[olorin] WhatsApp bridge started (pid={})", child.id());
    eprintln!("[olorin] Waiting for bridge connection...");

    wa_message_loop(child, ctx);
}

fn wa_message_loop(
    child: Box<dyn ChildProcess>,
    ctx: std::sync::Arc<std::sync::Mutex<DispatchContext>>,
) {
    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        if child.read_line(&mut line_buf).unwrap_or(0) == 0 {
            eprintln!("[olorin] Bridge stdout closed.");
            let _ = child.wait();
            return;
        }
        if line_buf.is_empty() { continue; }

        let msg_type = extract_json_string(&line_buf, "type").unwrap_or_default();
        match msg_type.as_str() {
            "connected" => eprintln!("[olorin] WhatsApp connected!"),
            "message" => {
                let text = extract_json_string(&line_buf, "text").unwrap_or_default();
                let jid = extract_json_string(&line_buf, "jid").unwrap_or_default();
                if text.is_empty() || jid.is_empty() { continue; }

                let response = {
                    let mut guard = ctx.lock().unwrap_or_else(|e| e.into_inner());
                    guard.dispatch(&text)
                };

                let reply = format!(
                    "{{\"type\":\"send\",\"jid\":\"{}\",\"text\":\"{}\"}}",
                    escape_json(&jid),
                    escape_json(&response.text)
                );
                let _ = child.write_line(&reply);
            }
            _ => {}
        }
    }
}

fn find_bridge() -> String {
    if let Ok(p) = std::env::var("OLORIN_BRIDGE") {
        return p;
    }
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .map(|p| p.join("bridge/wa-bridge"))
            .unwrap_or_default();
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "bridge/wa-bridge".to_string()
}
