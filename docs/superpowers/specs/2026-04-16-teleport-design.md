# /teleport — WhatsApp Bridge for Olorin

## Purpose

Allow Olorin to "teleport" to WhatsApp mid-session: the user types `/teleport` in the chat UI or REPL, Olorin starts responding on WhatsApp, and local interfaces go dormant. Typing `/teleport` on WhatsApp brings Olorin back. Conversation context (messages, recall, vault) carries across seamlessly.

## Architecture: Blocking Teleport (Approach A)

Single-presence model. When teleported, the WhatsApp message loop runs on the thread that holds the `DispatchContext` mutex, blocking local interfaces. An `Arc<AtomicBool>` flag lets the web server respond with a dormant message without waiting for the mutex.

```
Local input ──► pre_inference() ──► CMD_TELEPORT intercepted
                                         │
                                         ▼
                              Set AtomicBool teleported=true
                                         │
                                         ▼
                              Spawn Go bridge subprocess
                              (exec::spawn, --session-dir ~/.olorin/wa_session)
                                         │
                                         ▼
                              Read QR/connected events
                              Return QR art + status to chat bubble
                                         │
                                         ▼
                              Enter wa_message_loop (blocks this thread)
                                  │                    │
                          trigger match?          /teleport received?
                                  │                    │
                              dispatch()          Kill bridge, break loop
                              send response       Set teleported=false
                                                  Return to caller
```

## State Changes

### DispatchContext (router.rs)

New field:
- `teleported: bool` — checked in `pre_inference()` for early dormant return

### Server (server.rs)

New shared state:
- `Arc<AtomicBool>` teleported flag — checked in `handle_generate` and `handle_command` before acquiring the `DispatchContext` mutex

Flow while teleported:
- `handle_generate`: SSE response with `data: {"token":"Olorin is on WhatsApp. Send /teleport there to return."}` then `data: [DONE]`
- `handle_command`: JSON `{"output":"Olorin is on WhatsApp. Send /teleport there to return.","success":false}`

## Command Routing

`CMD_TELEPORT` (ID 27) is already registered in `dispatch.rs` within the tool range (`CMD_TOOL_FIRST..CMD_TOOL_LAST`). It gets intercepted in `pre_inference()` before the general tool handler, similar to how `CMD_RECALL` and `CMD_TASKS` are handled.

```rust
// In pre_inference(), after meta commands, before tool range:
if cmd_id == dispatch::CMD_TELEPORT {
    return Err(self.handle_teleport(/* teleported flag */));
}
```

## WhatsApp Message Loop

### Bridge Interaction

Reuses the existing Go bridge at `bridge/main.go` unchanged. Rust spawns it via `exec::spawn` with `--session-dir ~/.olorin/wa_session`.

### JSONL Protocol

Inbound (bridge to Rust):
- `{"type":"connected"}` — bridge ready
- `{"type":"qr","data":"..."}` — QR code data for first-time pairing
- `{"type":"message","jid":"...","sender":"...","sender_name":"...","text":"...","timestamp":N,"is_from_me":false}`

Outbound (Rust to bridge):
- `{"type":"send","jid":"...","text":"..."}`

### Trigger Matching

Olorin responds only to messages matching one of:
- `@olorin <text>` (case-insensitive)
- `!olorin <text>`
- `olorin <text>` (followed by space)
- `/teleport` (always matched — the return command)

The trigger prefix is stripped before passing to `dispatch()`. Non-matching messages are silently ignored. Messages with `is_from_me: true` are ignored.

### QR Code Handling

On first login, the Go bridge emits `{"type":"qr","data":"...","ascii":"..."}` where `ascii` contains a pre-rendered Unicode half-block QR code. The streaming teleport loop sends this as a `StreamEvent::Token` into the SSE connection, rendering it directly in the chat bubble (monospace `pre-wrap` rendering).

The Go bridge renders the QR using the same half-block technique it uses for stderr, but returns it as a string field in the JSON event. This avoids needing a QR library in Rust (zero-deps rule).

Subsequent connections reuse the stored session in `~/.olorin/wa_session/device.db` and skip QR.

### Exit Conditions

The loop exits when:
1. `/teleport` is received from WhatsApp
2. Bridge subprocess dies (disconnect, crash)
3. Read error on bridge stdout

On exit: `teleported` is set to `false`, bridge process is killed, mutex is released.

## File Changes

| File | Change |
|------|--------|
| `bridge/main.go` | Add `renderQRToString()`, emit `"ascii"` field in QR JSON event |
| `src/core/router.rs` | Add `teleported: bool` + `server_teleported` fields. `dispatch_streaming` handles CMD_TELEPORT specially for QR streaming |
| `src/interface/whatsapp.rs` | `teleport_loop()` for non-streaming, `teleport_loop_streaming()` for SSE with QR in chat bubble. Trigger matching |
| `src/interface/server.rs` | Add `Arc<AtomicBool>` teleported flag. Check in `handle_generate` and `handle_command` before mutex lock |
| `src/core/router_tools.rs` | `/help` text updated, `handle_teleport()` wired |

No changes to: command registration (`dispatch.rs`), frontend (`chat.html`), build system (`build.rs`, `Cargo.toml`).

## Testing

Integration tests in `tests/`:

1. **No bridge binary** — `/teleport` returns clear error message, `teleported` stays false
2. **Dormant message** — set `teleported = true`, verify `dispatch()` returns dormant message without entering inference
3. **Teleport return** — verify `/teleport` from WhatsApp exits the loop and clears the flag
