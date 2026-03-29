//! Session state — tracks active conversation channel, vault, model, and TTL.
//!
//! Serialized as JSON using crate::storage::json (no serde dependency).

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::storage::json::{self, Object, Value};

/// Active session token — persists across restarts within TTL.
#[derive(Debug, Clone)]
pub struct SessionToken {
    pub active_channel:       String,
    pub vault_id:             String,
    pub seq_len:              usize,
    pub context_window_start: usize,
    pub last_msg_hash:        String,
    pub model:                String,
    pub timestamp:            u64,
    pub ttl:                  u64,
}

impl SessionToken {
    pub fn new(channel: &str, vault_id: &str, model: &str) -> Self {
        Self {
            active_channel:       channel.to_string(),
            vault_id:             vault_id.to_string(),
            seq_len:              0,
            context_window_start: 0,
            last_msg_hash:        String::new(),
            model:                model.to_string(),
            timestamp:            now_epoch(),
            ttl:                  86400,
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), std::io::Error> {
        let mut obj = Object::new();
        obj.set("active_channel",       Value::Str(self.active_channel.clone()));
        obj.set("vault_id",             Value::Str(self.vault_id.clone()));
        obj.set("seq_len",              Value::I64(self.seq_len as i64));
        obj.set("context_window_start", Value::I64(self.context_window_start as i64));
        obj.set("last_msg_hash",        Value::Str(self.last_msg_hash.clone()));
        obj.set("model",                Value::Str(self.model.clone()));
        obj.set("timestamp",            Value::I64(self.timestamp as i64));
        obj.set("ttl",                  Value::I64(self.ttl as i64));
        let s = json::serialize(&obj);
        std::fs::write(path, s)
    }

    pub fn load(path: &Path) -> Result<Option<Self>, std::io::Error> {
        if !path.exists() { return Ok(None); }
        let data = std::fs::read(path)?;
        let obj = json::parse(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        let token = Self {
            active_channel:       obj.get_str("active_channel").unwrap_or("").to_string(),
            vault_id:             obj.get_str("vault_id").unwrap_or("").to_string(),
            seq_len:              obj.get_i64("seq_len").unwrap_or(0) as usize,
            context_window_start: obj.get_i64("context_window_start").unwrap_or(0) as usize,
            last_msg_hash:        obj.get_str("last_msg_hash").unwrap_or("").to_string(),
            model:                obj.get_str("model").unwrap_or("").to_string(),
            timestamp:            obj.get_i64("timestamp").unwrap_or(0) as u64,
            ttl:                  obj.get_i64("ttl").unwrap_or(86400) as u64,
        };

        if token.is_expired() { return Ok(None); }
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
