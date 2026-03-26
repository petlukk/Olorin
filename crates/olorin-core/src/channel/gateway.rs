//! Gateway: routes messages from group channels to per-group agents.
//!
//! The gateway owns the channel connection and dispatches messages
//! based on JID. Each group gets its own VectorStore, history log,
//! and conversation context.

use crate::channel::types::{matches_trigger, InboundMessage};
use crate::config::Config;
use crate::error::Result;
use crate::persist::HistoryLog;
use crate::recall::VectorStore;
use crate::safety::SafetyLayer;
use std::collections::HashMap;

/// Per-group agent state.
pub struct GroupAgent {
    pub jid: String,
    pub store: VectorStore,
    pub history: HistoryLog,
    pub message_count: usize,
}

impl GroupAgent {
    /// Create a new group agent, replaying history into the vector store.
    pub fn new(jid: &str) -> Result<Self> {
        let history = HistoryLog::for_group(jid)?;
        let mut store = VectorStore::with_capacity(1024);
        let replayed = history.replay_into(&mut store)?;
        tracing::info!(jid, replayed, "group agent initialized");
        Ok(Self {
            jid: jid.to_string(),
            store,
            history,
            message_count: 0,
        })
    }
}

/// Gateway that routes group channel messages.
pub struct Gateway {
    trigger: String,
    groups: HashMap<String, GroupAgent>,
    safety: SafetyLayer,
}

impl Gateway {
    pub fn new(config: &Config) -> Self {
        Self {
            trigger: config.agent_name.clone(),
            groups: HashMap::new(),
            safety: SafetyLayer::new(),
        }
    }

    /// Register a group for message handling.
    pub fn register_group(&mut self, jid: &str) -> Result<()> {
        if !self.groups.contains_key(jid) {
            let agent = GroupAgent::new(jid)?;
            self.groups.insert(jid.to_string(), agent);
        }
        Ok(())
    }

    /// Process an inbound message. Returns a response if the message
    /// matches the trigger and passes safety checks, or None to skip.
    pub fn process_inbound(&mut self, msg: &InboundMessage) -> Option<ProcessedMessage> {
        // Trigger check (~20 ns)
        if !matches_trigger(&msg.text, &self.trigger) {
            return None;
        }

        // Safety scan (~2 µs)
        let scan = self.safety.scan_input(&msg.text);
        if scan.injection_found {
            return Some(ProcessedMessage {
                jid: msg.jid.clone(),
                action: Action::Blocked("potential injection detected".into()),
            });
        }
        if scan.leaks_found {
            return Some(ProcessedMessage {
                jid: msg.jid.clone(),
                action: Action::Blocked("message contains potential secrets".into()),
            });
        }

        // Get or create group agent
        if !self.groups.contains_key(&msg.jid) {
            if let Err(e) = self.register_group(&msg.jid) {
                tracing::error!(jid = %msg.jid, error = %e, "failed to register group");
                return None;
            }
        }
        let agent = self.groups.get_mut(&msg.jid).unwrap();

        // Index message for recall
        let recall_text = format!("{}: {}", msg.sender_name, msg.text);
        agent.store.insert(&recall_text);
        agent.message_count += 1;

        // Persist
        if let Err(e) = agent.history.append(msg) {
            tracing::warn!(error = %e, "failed to persist message");
        }

        // Get recent context via recall
        let context = agent.store.recall(&msg.text, 10);
        let context_text: Vec<String> = context.iter()
            .map(|r| r.text.clone())
            .collect();

        Some(ProcessedMessage {
            jid: msg.jid.clone(),
            action: Action::Forward {
                text: msg.text.clone(),
                sender_name: msg.sender_name.clone(),
                context: context_text,
            },
        })
    }

    /// Record an agent response for a group.
    pub fn record_response(&mut self, jid: &str, response: &str) {
        if let Some(agent) = self.groups.get_mut(jid) {
            agent.store.insert(response);
            if let Err(e) = agent.history.append_text(response) {
                tracing::warn!(error = %e, "failed to persist response");
            }
        }
    }

    /// Number of registered groups.
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    /// List registered group JIDs.
    pub fn group_jids(&self) -> Vec<&str> {
        self.groups.keys().map(|s| s.as_str()).collect()
    }
}

/// Result of processing an inbound message.
pub struct ProcessedMessage {
    pub jid: String,
    pub action: Action,
}

/// Action to take after processing a message.
pub enum Action {
    /// Forward to LLM with context.
    Forward {
        text: String,
        sender_name: String,
        context: Vec<String>,
    },
    /// Message was blocked by safety scan.
    Blocked(String),
}
