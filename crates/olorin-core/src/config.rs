use crate::error::{Error, Result};
use crate::safety::shell_guard::{self, ShellPolicy};
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum Backend {
    Anthropic,
    Local,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub api_key: String,
    pub model: String,
    pub agent_name: String,
    pub max_turns: usize,
    pub command_prefix: String,
    /// Optional identity text prepended to the system prompt.
    pub identity: Option<String>,
    /// Allowed HTTP hosts. Empty = allow all.
    pub allowed_hosts: Vec<String>,
    pub backend: Backend,
    pub model_path: Option<String>,
    pub ctx_size: usize,
    pub batch_size: usize,
    pub threads: usize,
    pub mlock: bool,
    pub shell_policy: ShellPolicy,
}

impl Config {
    /// Load configuration from environment variables.
    pub fn from_env() -> Result<Self> {
        let backend = match env::var("EACLAW_BACKEND").as_deref() {
            Ok("local") => Backend::Local,
            _ => Backend::Anthropic,
        };

        let api_key = if backend == Backend::Anthropic {
            env::var("ANTHROPIC_API_KEY")
                .map_err(|_| Error::Config("ANTHROPIC_API_KEY not set".into()))?
                .trim()
                .to_string()
        } else {
            env::var("ANTHROPIC_API_KEY").unwrap_or_default()
                .trim()
                .to_string()
        };

        let model = env::var("ANTHROPIC_MODEL")
            .unwrap_or_else(|_| "claude-sonnet-4-20250514".into());

        let agent_name = env::var("AGENT_NAME")
            .unwrap_or_else(|_| "eaclaw".into());

        let max_turns = env::var("MAX_TURNS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);

        let command_prefix = env::var("COMMAND_PREFIX")
            .unwrap_or_else(|_| "/".into());

        let identity = load_identity();
        let allowed_hosts = load_allowed_hosts();

        let model_path = env::var("EACLAW_MODEL_PATH").ok().or_else(|| {
            home::home_dir().map(|h| {
                h.join(".eaclaw/models/qwen2.5-3b-instruct-q4_k_m.gguf")
                    .to_string_lossy()
                    .into_owned()
            })
        });

        let ctx_size = env::var("EACLAW_CTX_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048);

        let batch_size = env::var("EACLAW_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);

        let threads = env::var("EACLAW_THREADS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            });

        let mlock = env::var("EACLAW_MLOCK")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let shell_policy = shell_guard::load_shell_policy();

        Ok(Self {
            api_key,
            model,
            agent_name,
            max_turns,
            command_prefix,
            identity,
            allowed_hosts,
            backend,
            model_path,
            ctx_size,
            batch_size,
            threads,
            mlock,
            shell_policy,
        })
    }
}

/// Load identity from EACLAW_IDENTITY env var path or ~/.eaclaw/identity.md.
fn load_identity() -> Option<String> {
    let path = if let Ok(p) = env::var("EACLAW_IDENTITY") {
        PathBuf::from(p)
    } else {
        dirs()?.join("identity.md")
    };
    std::fs::read_to_string(&path).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Load allowed hosts from EACLAW_ALLOWED_HOSTS env var (comma-separated)
/// or ~/.eaclaw/allowed_hosts.txt (one per line).
fn load_allowed_hosts() -> Vec<String> {
    if let Ok(hosts) = env::var("EACLAW_ALLOWED_HOSTS") {
        return hosts
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect();
    }
    if let Some(path) = dirs().map(|d| d.join("allowed_hosts.txt")) {
        if let Ok(content) = std::fs::read_to_string(&path) {
            return content
                .lines()
                .map(|l| l.trim().to_lowercase())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .collect();
        }
    }
    Vec::new()
}

/// Returns ~/.eaclaw/ directory path.
fn dirs() -> Option<PathBuf> {
    home::home_dir().map(|h| h.join(".eaclaw"))
}
