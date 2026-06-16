//! Config + API-key POST handlers for the web server — split out of
//! `server.rs` to keep that file under the 500-line cap.

use std::sync::{Arc, Mutex};
use crate::core::router::DispatchContext;
use crate::interface::server::serve_json;
use crate::interface::server_http::read_body;

pub(crate) fn handle_config_update(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str = std::str::from_utf8(&body_bytes).unwrap_or("");
    ctx.lock().unwrap_or_else(|e| e.into_inner()).update_config(body_str);
    let config = ctx.lock().unwrap_or_else(|e| e.into_inner()).get_config();
    serve_json(stream, &config);
}

pub(crate) fn handle_config_apikey(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let key = std::str::from_utf8(&body_bytes).unwrap_or("").trim();
    if key.is_empty() {
        serve_json(stream, r#"{"ok":false,"error":"empty key"}"#);
        return;
    }
    ctx.lock().unwrap_or_else(|e| e.into_inner()).store_api_key(key);
    serve_json(stream, r#"{"ok":true}"#);
}
