# Config File + Web UI Model Selector

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TOML config file for persistent settings (API key, default model, recall level) and a Web UI model selector dropdown.

**Architecture:** `~/.olorin/config.toml` parsed at startup with a hand-rolled TOML scanner (no serde). CLI flags override config. Web UI gets a `/api/models` endpoint listing available models and a `/api/model` POST to switch at runtime. DispatchContext holds the active model name and can hot-swap Engine.

**Tech Stack:** Existing `storage/json.rs` pattern for the TOML parser (line-oriented key=value). No new deps.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/core/config.rs` | Create | Parse `~/.olorin/config.toml`, expose typed Config struct |
| `src/core/mod.rs` | Modify | Add `pub mod config;` |
| `src/core/router.rs` | Modify | Accept Config, expose model swap, model_name getter |
| `src/inference/generate.rs` | Modify | `list_models()` returning available model aliases + paths |
| `src/interface/server.rs` | Modify | `/api/models` GET, `/api/model` POST, pass config to DispatchContext |
| `src/interface/terminal.rs` | Modify | Pass config to DispatchContext |
| `src/main.rs` | Modify | Load config, CLI overrides, pass to run functions |
| `web/chat.html` | Modify | Model dropdown in hyprbar, fetch/switch model |
| `tests/config.rs` | Create | Config parsing tests |

---

### Task 1: Config parser

**Files:**
- Create: `src/core/config.rs`
- Modify: `src/core/mod.rs`
- Create: `tests/config.rs`

- [ ] **Step 1: Write failing tests**

```rust
// tests/config.rs
use olorin::core::config::Config;

#[test]
fn test_parse_empty() {
    let cfg = Config::parse("");
    assert!(cfg.anthropic_api_key.is_none());
    assert!(cfg.default_model.is_none());
    assert_eq!(cfg.recall_level, 0);
}

#[test]
fn test_parse_all_fields() {
    let toml = r#"
anthropic_api_key = "sk-ant-test123"
default_model = "llama8b"
recall_level = 3
"#;
    let cfg = Config::parse(toml);
    assert_eq!(cfg.anthropic_api_key.as_deref(), Some("sk-ant-test123"));
    assert_eq!(cfg.default_model.as_deref(), Some("llama8b"));
    assert_eq!(cfg.recall_level, 3);
}

#[test]
fn test_parse_comments_and_whitespace() {
    let toml = "# comment\n  recall_level = 5  \n";
    let cfg = Config::parse(toml);
    assert_eq!(cfg.recall_level, 5);
}

#[test]
fn test_load_missing_file() {
    let cfg = Config::load("/tmp/nonexistent_olorin_config.toml");
    assert!(cfg.anthropic_api_key.is_none());
}
```

- [ ] **Step 2: Implement Config**

```rust
// src/core/config.rs
pub struct Config {
    pub anthropic_api_key: Option<String>,
    pub default_model: Option<String>,
    pub recall_level: usize,
    pub port: u16,
}

impl Config {
    pub fn load(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(s) => Self::parse(&s),
            Err(_) => Self::default(),
        }
    }

    pub fn parse(toml: &str) -> Self { /* line-by-line key = "value" */ }

    pub fn default() -> Self {
        Config { anthropic_api_key: None, default_model: None, recall_level: 0, port: 8080 }
    }
}
```

- [ ] **Step 3: Run tests, commit**

---

### Task 2: Wire config into startup

**Files:**
- Modify: `src/main.rs`
- Modify: `src/core/router.rs`
- Modify: `src/interface/server.rs`
- Modify: `src/interface/terminal.rs`

- [ ] **Step 1: Load config in main.rs**

```rust
let config = Config::load(&format!("{}/.olorin/config.toml", std::env::var("HOME").unwrap_or_default()));
// CLI flags override config:
let model_arg = get_opt(&args, "--model").or(config.default_model.as_deref());
let api_key = std::env::var("ANTHROPIC_API_KEY").ok().or(config.anthropic_api_key.clone());
let port = get_opt(&args, "--port").and_then(|s| s.parse().ok()).unwrap_or(config.port);
```

- [ ] **Step 2: Pass api_key and model_arg through run functions**

Server and terminal `run()` already take `model_arg`. Add `api_key` parameter instead of reading env var internally.

- [ ] **Step 3: DispatchContext uses config recall_level as default**

- [ ] **Step 4: Run tests, commit**

---

### Task 3: List available models

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Add `list_models()` function**

```rust
pub fn list_models() -> Vec<(&'static str, String, bool)> {
    // Returns (alias, path, exists) for each known model
    let home = std::env::var("HOME").unwrap_or_default();
    let dir = Path::new(&home).join(".olorin/models");
    ALIASES.iter().map(|&(alias, filename)| {
        let p = dir.join(filename);
        let exists = p.exists();
        (alias, p.to_string_lossy().to_string(), exists)
    }).collect()
}
```

- [ ] **Step 2: Test, commit**

---

### Task 4: Model swap at runtime

**Files:**
- Modify: `src/core/router.rs`

- [ ] **Step 1: Add `/model` slash command handler**

`/model` → show current model name.
`/model llama8b` → reload Engine with new model path.

- [ ] **Step 2: Add `model_name()` getter to DispatchContext**

- [ ] **Step 3: Test via pipe tests, commit**

---

### Task 5: Server API endpoints

**Files:**
- Modify: `src/interface/server.rs`

- [ ] **Step 1: `GET /api/models`**

Returns JSON array of available models:
```json
[{"alias":"bitnet","path":"...","available":true,"active":true},...]
```

- [ ] **Step 2: `POST /api/model`**

Accepts `{"model":"llama8b"}`, calls DispatchContext model swap, returns new model info.

- [ ] **Step 3: Update `GET /api/model` to include active model name from DispatchContext**

- [ ] **Step 4: Build, test, commit**

---

### Task 6: Web UI model dropdown

**Files:**
- Modify: `web/chat.html`

- [ ] **Step 1: Replace static model text in hyprbar with dropdown**

```html
<select id="hb-model-select" class="c-mauve">...</select>
```

- [ ] **Step 2: Populate dropdown from `GET /api/models` on page load**

Only show models where `available: true`. Mark active model as selected.

- [ ] **Step 3: On dropdown change, POST `/api/model` to switch**

Show loading state while model loads (can take seconds for large models).

- [ ] **Step 4: Manual test in browser, commit**

---

### Task 7: Integration test

- [ ] **Step 1: Test config → startup → model selection → switch → verify**

```
1. Create temp config.toml with default_model = "bitnet"
2. Start olorin, verify bitnet loaded
3. /model llama → verify switch
4. /model → verify shows current
```

- [ ] **Step 2: Commit all**
