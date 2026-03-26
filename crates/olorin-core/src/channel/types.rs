//! Types for multi-group messaging channels (WhatsApp, Telegram, etc).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// An inbound message from a group channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    pub jid: String,
    pub sender: String,
    pub sender_name: String,
    pub text: String,
    pub timestamp: u64,
    #[serde(default)]
    pub is_from_me: bool,
}

/// A multi-group channel that routes messages by JID.
#[async_trait]
pub trait GroupChannel: Send + Sync {
    fn name(&self) -> &str;
    async fn recv(&self) -> Option<InboundMessage>;
    async fn send(&self, jid: &str, content: &str);
    fn is_connected(&self) -> bool;
    async fn disconnect(&self);
}

/// Check if a message matches a trigger pattern.
/// Returns true if the message should be processed by an agent.
pub fn matches_trigger(text: &str, trigger: &str) -> bool {
    if trigger.is_empty() {
        return true; // no trigger = process everything
    }
    let lower = text.to_lowercase();
    let trigger_lower = trigger.to_lowercase();
    // Match: @trigger, !trigger, or trigger at start of message
    lower.contains(&format!("@{trigger_lower}"))
        || lower.contains(&format!("!{trigger_lower}"))
        || lower.starts_with(&trigger_lower)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_at_mention() {
        assert!(matches_trigger("hey @eaclaw help me", "eaclaw"));
    }

    #[test]
    fn test_trigger_bang() {
        assert!(matches_trigger("!eaclaw what time is it", "eaclaw"));
    }

    #[test]
    fn test_trigger_start() {
        assert!(matches_trigger("eaclaw do something", "eaclaw"));
    }

    #[test]
    fn test_trigger_case_insensitive() {
        assert!(matches_trigger("@EACLAW hello", "eaclaw"));
    }

    #[test]
    fn test_trigger_no_match() {
        assert!(!matches_trigger("hello world", "eaclaw"));
    }

    #[test]
    fn test_trigger_empty() {
        assert!(matches_trigger("anything goes", ""));
    }
}
