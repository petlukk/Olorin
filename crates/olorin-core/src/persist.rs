//! Append-only JSONL persistence for conversation history.
//!
//! Each group gets a `history.jsonl` file under `~/.eaclaw/groups/{jid}/`.
//! On startup, replay into VectorStore. During runtime, append-only writes.

use crate::channel::types::InboundMessage;
use crate::error::{Error, Result};
use crate::recall::VectorStore;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Manages JSONL history for a single group.
pub struct HistoryLog {
    path: PathBuf,
}

impl HistoryLog {
    /// Create a history log for a group JID.
    /// Creates the directory structure if needed.
    pub fn for_group(jid: &str) -> Result<Self> {
        let dir = group_dir(jid)?;
        fs::create_dir_all(&dir)?;
        Ok(Self {
            path: dir.join("history.jsonl"),
        })
    }

    /// Create a history log at a specific path (for testing).
    pub fn at_path(path: PathBuf) -> Self {
        Self { path }
    }

    /// Append a message to the log.
    pub fn append(&self, msg: &InboundMessage) -> Result<()> {
        let line = serde_json::to_string(msg)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Append a raw text entry (for agent responses).
    pub fn append_text(&self, text: &str) -> Result<()> {
        let entry = serde_json::json!({"text": text, "type": "assistant"});
        let line = serde_json::to_string(&entry)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Replay history into a VectorStore.
    /// Returns the number of entries replayed.
    pub fn replay_into(&self, store: &mut VectorStore) -> Result<usize> {
        if !self.path.exists() {
            return Ok(0);
        }
        let file = fs::File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut count = 0;
        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            // Try to extract text from either message or assistant entry
            let text = extract_text(&line);
            if let Some(text) = text {
                if !text.trim().is_empty() {
                    store.insert(&text);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Number of lines in the log file.
    pub fn entry_count(&self) -> usize {
        if !self.path.exists() {
            return 0;
        }
        fs::File::open(&self.path)
            .map(|f| BufReader::new(f).lines().count())
            .unwrap_or(0)
    }

    /// Path to the history file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Extract text content from a JSONL line.
fn extract_text(line: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    // InboundMessage format: {"text": "...", "sender_name": "...", ...}
    // Assistant format: {"text": "...", "type": "assistant"}
    v.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
}

/// Get the directory for a group's data.
pub fn group_dir(jid: &str) -> Result<PathBuf> {
    let base = home::home_dir()
        .ok_or_else(|| Error::Config("cannot determine home directory".into()))?;
    // Sanitize JID for filesystem: replace @ and . with _
    let safe_name: String = jid.chars().map(|c| match c {
        '@' | '.' | ':' => '_',
        c => c,
    }).collect();
    Ok(base.join(".eaclaw").join("groups").join(safe_name))
}

/// List all registered group JIDs (directories under ~/.eaclaw/groups/).
pub fn list_groups() -> Result<Vec<String>> {
    let base = home::home_dir()
        .ok_or_else(|| Error::Config("cannot determine home directory".into()))?;
    let groups_dir = base.join(".eaclaw").join("groups");
    if !groups_dir.exists() {
        return Ok(Vec::new());
    }
    let mut groups = Vec::new();
    for entry in fs::read_dir(&groups_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                groups.push(name.to_string());
            }
        }
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_log() -> (HistoryLog, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_history.jsonl");
        (HistoryLog::at_path(path), dir)
    }

    #[test]
    fn test_append_and_replay() {
        let (log, _dir) = tmp_log();
        let msg = InboundMessage {
            jid: "test@g.us".into(),
            sender: "user1".into(),
            sender_name: "Alice".into(),
            text: "hello from whatsapp".into(),
            timestamp: 1000,
            is_from_me: false,
        };
        log.append(&msg).unwrap();
        log.append_text("I can help with that!").unwrap();

        assert_eq!(log.entry_count(), 2);

        let mut store = VectorStore::with_capacity(10);
        let count = log.replay_into(&mut store).unwrap();
        assert_eq!(count, 2);
        assert_eq!(store.len(), 2);

        let results = store.recall("whatsapp", 1);
        assert!(!results.is_empty());
        assert!(results[0].text.contains("whatsapp"));
    }

    #[test]
    fn test_replay_empty() {
        let (log, _dir) = tmp_log();
        let mut store = VectorStore::new();
        let count = log.replay_into(&mut store).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_extract_text() {
        let msg = r#"{"text":"hello","sender":"user","jid":"g@g.us","sender_name":"U","timestamp":0}"#;
        assert_eq!(extract_text(msg), Some("hello".into()));

        let asst = r#"{"text":"response","type":"assistant"}"#;
        assert_eq!(extract_text(asst), Some("response".into()));

        assert_eq!(extract_text("not json"), None);
    }

    #[test]
    fn test_group_dir_sanitizes_jid() {
        let dir = group_dir("group@g.us").unwrap();
        let name = dir.file_name().unwrap().to_str().unwrap();
        assert!(!name.contains('@'));
        assert!(!name.contains('.'));
    }
}
