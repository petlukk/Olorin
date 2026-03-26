//! WhatsApp channel via a bridge subprocess (whatsmeow or similar).
//!
//! The bridge binary handles the WhatsApp Web protocol.
//! We communicate via JSON lines on stdin/stdout:
//!
//! Inbound:  {"type":"message","jid":"...","sender":"...","sender_name":"...","text":"...","timestamp":N}
//! Outbound: {"type":"send","jid":"...","text":"..."}
//! Control:  {"type":"connected"} / {"type":"qr","data":"..."}

use crate::channel::types::{GroupChannel, InboundMessage};
use crate::error::{Error, Result};
use async_trait::async_trait;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex};

/// WhatsApp channel backed by a bridge subprocess.
pub struct WhatsAppChannel {
    name: String,
    rx: Mutex<mpsc::Receiver<InboundMessage>>,
    tx_handle: mpsc::Sender<String>,
    connected: Arc<AtomicBool>,
    _child: Child,
}

impl WhatsAppChannel {
    /// Start the WhatsApp bridge subprocess and connect.
    /// `bridge_path` is the path to the bridge binary.
    /// `session_dir` is the directory for WhatsApp session data.
    pub async fn start(bridge_path: &str, session_dir: &str) -> Result<Self> {
        let mut child = Command::new(bridge_path)
            .arg("--session-dir")
            .arg(session_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| Error::Channel(format!("failed to start bridge: {e}")))?;

        let stdout = child.stdout.take()
            .ok_or_else(|| Error::Channel("no stdout from bridge".into()))?;
        let stdin = child.stdin.take()
            .ok_or_else(|| Error::Channel("no stdin to bridge".into()))?;

        let (msg_tx, msg_rx) = mpsc::channel::<InboundMessage>(256);
        let (send_tx, mut send_rx) = mpsc::channel::<String>(256);
        let connected = Arc::new(AtomicBool::new(false));

        // Reader task: parse JSON lines from bridge stdout
        let conn_flag = connected.clone();
        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                    match val.get("type").and_then(|t| t.as_str()) {
                        Some("connected") => {
                            conn_flag.store(true, Ordering::Relaxed);
                            tracing::info!("WhatsApp bridge connected");
                        }
                        Some("qr") => {
                            // QR code is rendered by the bridge to stderr.
                            // Log that we received it for debugging.
                            tracing::info!("QR code received — check terminal for scannable code");
                        }
                        Some("message") => {
                            if let Ok(msg) = serde_json::from_value::<InboundMessage>(val) {
                                let _ = msg_tx.send(msg).await;
                            }
                        }
                        _ => {
                            tracing::debug!("bridge: {line}");
                        }
                    }
                }
            }
        });

        // Writer task: send JSON lines to bridge stdin
        tokio::spawn(async move {
            let mut stdin = stdin;
            while let Some(line) = send_rx.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                let _ = stdin.flush().await;
            }
        });

        Ok(Self {
            name: "whatsapp".into(),
            rx: Mutex::new(msg_rx),
            tx_handle: send_tx,
            connected,
            _child: child,
        })
    }
}

#[async_trait]
impl GroupChannel for WhatsAppChannel {
    fn name(&self) -> &str {
        &self.name
    }

    async fn recv(&self) -> Option<InboundMessage> {
        self.rx.lock().await.recv().await
    }

    async fn send(&self, jid: &str, content: &str) {
        let msg = serde_json::json!({
            "type": "send",
            "jid": jid,
            "text": content,
        });
        let _ = self.tx_handle.send(msg.to_string()).await;
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    async fn disconnect(&self) {
        self.connected.store(false, Ordering::Relaxed);
        // Bridge process will be killed when _child is dropped
    }
}
