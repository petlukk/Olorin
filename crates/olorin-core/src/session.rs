use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SessionToken {
    pub active_channel: String,
    pub vault_id: String,
    pub seq_len: usize,
    pub context_window_start: usize,
    pub last_msg_hash: String,
    pub model: String,
    pub timestamp: u64,
    pub ttl: u64,
}

impl SessionToken {
    pub fn new(channel: &str, vault_id: &str, model: &str) -> Self {
        Self {
            active_channel: channel.to_string(),
            vault_id: vault_id.to_string(),
            seq_len: 0,
            context_window_start: 0,
            last_msg_hash: String::new(),
            model: model.to_string(),
            timestamp: now_epoch(),
            ttl: 86400, // 24 hours
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &Path) -> Result<Option<Self>, std::io::Error> {
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read_to_string(path)?;
        let token: Self = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if token.is_expired() {
            return Ok(None);
        }
        Ok(Some(token))
    }

    pub fn is_expired(&self) -> bool {
        now_epoch() > self.timestamp + self.ttl
    }
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_session_token_roundtrip() {
        let token = SessionToken::new("whatsapp", "group_abc", "bitnet-2b");
        let path = std::env::temp_dir().join("test_session.json");
        token.save(&path).unwrap();
        let loaded = SessionToken::load(&path).unwrap().unwrap();
        assert_eq!(loaded.vault_id, "group_abc");
        assert_eq!(loaded.active_channel, "whatsapp");
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_session_token_expiry() {
        let mut token = SessionToken::new("web", "group_123", "bitnet-2b");
        token.timestamp = 0; // very old
        token.ttl = 1;
        assert!(token.is_expired());
    }

    #[test]
    fn test_session_load_missing_file() {
        let result = SessionToken::load(Path::new("/tmp/nonexistent_session.json")).unwrap();
        assert!(result.is_none());
    }
}
