//! WhatsApp channel via a bridge subprocess (whatsmeow or similar).
//!
//! Uses exec::spawn (raw fork+exec) instead of std::process::Command
//! to avoid pidfd@GLIBC_2.39. The bridge binary handles the WhatsApp
//! Web protocol. Communication via JSON lines on stdin/stdout.

use crate::channel::types::{GroupChannel, InboundMessage};
use crate::error::{Error, Result};
use crate::exec;
use async_trait::async_trait;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// WhatsApp channel backed by a bridge subprocess.
pub struct WhatsAppChannel {
    name: String,
    rx: Mutex<mpsc::Receiver<InboundMessage>>,
    tx_handle: mpsc::Sender<String>,
    connected: Arc<AtomicBool>,
}

impl WhatsAppChannel {
    /// Start the WhatsApp bridge subprocess and connect.
    pub async fn start(bridge_path: &str, session_dir: &str) -> Result<Self> {
        let child = exec::spawn(&[bridge_path, "--session-dir", session_dir])
            .map_err(|e| Error::Channel(format!("failed to start bridge: {e}")))?;

        let child = Arc::new(std::sync::Mutex::new(child));
        let (msg_tx, msg_rx) = mpsc::channel::<InboundMessage>(256);
        let (send_tx, mut send_rx) = mpsc::channel::<String>(256);
        let connected = Arc::new(AtomicBool::new(false));

        // Reader task: parse JSON lines from bridge stdout
        let reader_child = child.clone();
        let conn_flag = connected.clone();
        tokio::task::spawn_blocking(move || {
            let mut line = String::new();
            loop {
                let child = reader_child.lock().unwrap();
                match child.read_line(&mut line) {
                    Ok(0) => break, // EOF
                    Ok(_) => {}
                    Err(_) => break,
                }
                drop(child);

                if line.is_empty() {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                    match val.get("type").and_then(|t| t.as_str()) {
                        Some("connected") => {
                            conn_flag.store(true, Ordering::Relaxed);
                        }
                        Some("message") => {
                            if let Ok(msg) = serde_json::from_value::<InboundMessage>(val) {
                                let _ = msg_tx.blocking_send(msg);
                            }
                        }
                        _ => {}
                    }
                }
            }
        });

        // Writer task: send JSON lines to bridge stdin
        let writer_child = child.clone();
        tokio::task::spawn_blocking(move || {
            while let Some(line) = send_rx.blocking_recv() {
                let child = writer_child.lock().unwrap();
                if child.write_line(&line).is_err() {
                    break;
                }
            }
        });

        Ok(Self {
            name: "whatsapp".into(),
            rx: Mutex::new(msg_rx),
            tx_handle: send_tx,
            connected,
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
    }
}
