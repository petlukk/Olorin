//! Alert sinks for the logwatch palantír. Each `--notify` flag selects where
//! an alert goes; the default is stdout. Delivery never blocks the watcher and
//! never aborts on a sink error — a dead webhook must not stop the alerts that
//! follow it.
//!
//! - `stdout`        — the rendered line (default).
//! - `webhook:URL`   — HTTP POST a JSON body via curl. Slack/PagerDuty/Discord
//!                     take it directly; point it at a WhatsApp webhook service
//!                     (CallMeBot, Twilio) for phone alerts.
//! - `exec:CMD`      — run `CMD` per alert with the alert in the environment
//!                     (`PALANTIR_ALERT`/`_KIND`/`_SEVERITY`). Wire it to the
//!                     local wa-bridge, `ntfy`, `mail`, desktop notifications…

use super::watch::Alert;
use std::process::Command;

pub enum Sink {
    Stdout,
    Webhook(String),
    Exec(String),
}

impl Sink {
    /// Parse one `--notify` spec.
    pub fn parse(spec: &str) -> Result<Sink, String> {
        if spec == "stdout" {
            return Ok(Sink::Stdout);
        }
        if let Some(url) = spec.strip_prefix("webhook:") {
            if url.is_empty() {
                return Err("webhook: needs a URL (webhook:https://…)".to_string());
            }
            return Ok(Sink::Webhook(url.to_string()));
        }
        if let Some(cmd) = spec.strip_prefix("exec:") {
            if cmd.is_empty() {
                return Err("exec: needs a command (exec:notify-send …)".to_string());
            }
            return Ok(Sink::Exec(cmd.to_string()));
        }
        Err(format!(
            "unknown --notify sink: {spec} (expected stdout | webhook:URL | exec:CMD)"
        ))
    }

    pub fn deliver(&self, alert: &Alert) {
        match self {
            Sink::Stdout => println!("{}", alert.render()),
            Sink::Webhook(url) => deliver_webhook(url, alert),
            Sink::Exec(cmd) => deliver_exec(cmd, alert),
        }
    }
}

/// JSON body for a webhook: `text` is the rendered line (Slack reads this);
/// `source`/`kind`/`severity` let richer receivers route on the structured
/// fields. Slack ignores the extra keys.
pub fn webhook_body(alert: &Alert) -> String {
    format!(
        "{{\"text\":\"{}\",\"source\":\"olorin-palantir\",\"kind\":\"{}\",\"severity\":\"{}\"}}",
        json_escape(&alert.render()),
        alert.kind(),
        alert.severity(),
    )
}

fn deliver_webhook(url: &str, alert: &Alert) {
    let body = webhook_body(alert);
    let out = Command::new("curl")
        .arg("-sS")
        .arg("--max-time").arg("10")
        .arg("-X").arg("POST")
        .arg("-H").arg("content-type: application/json")
        .arg("-d").arg(&body)
        .arg(url)
        .output();
    if let Ok(o) = out {
        if !o.status.success() {
            eprintln!(
                "[palantír] webhook delivery to {url} failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            );
        }
    } else {
        eprintln!("[palantír] webhook delivery failed: curl not found");
    }
}

fn deliver_exec(cmd: &str, alert: &Alert) {
    #[cfg(unix)]
    let (shell, flag) = ("sh", "-c");
    #[cfg(windows)]
    let (shell, flag) = ("cmd", "/C");
    let status = Command::new(shell)
        .arg(flag)
        .arg(cmd)
        .env("PALANTIR_ALERT", alert.render())
        .env("PALANTIR_KIND", alert.kind())
        .env("PALANTIR_SEVERITY", alert.severity())
        .status();
    match status {
        Ok(s) if !s.success() => {
            eprintln!("[palantír] exec sink `{cmd}` exited {}", s.code().unwrap_or(-1))
        }
        Err(e) => eprintln!("[palantír] exec sink `{cmd}` failed to start: {e}"),
        _ => {}
    }
}

/// Minimal JSON string escaper — quotes, backslashes, and control chars. Kept
/// local so the palantír subsystem doesn't reach into the web layer.
pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
