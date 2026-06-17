//! Tests for runtime config API — get/update/partial update.
//!
//! `DispatchContext::new` auto-loads the production vault + model from
//! `$HOME/.olorin/…`. Each test isolates `HOME` to a fresh tmpdir so
//! real user state (saved API keys, downloaded models) doesn't leak in.

use olorin::core::router::DispatchContext;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

// Kernel init reads HOME too; run it once under the real HOME so the
// loaded .so handles stay valid when later tests rotate HOME.
static KERNELS_INIT: OnceLock<()> = OnceLock::new();
// Serialize tests — they mutate the process-global HOME env var.
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct IsolatedHome {
    _guard: std::sync::MutexGuard<'static, ()>,
    path: PathBuf,
    old_home: Option<String>,
}

impl IsolatedHome {
    fn new(tag: &str) -> Self {
        KERNELS_INIT.get_or_init(|| olorin::kernels::ffi::init().unwrap());
        let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("olorin-cfg-{pid}-{tag}"));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        let old_home = std::env::var("HOME").ok();
        std::env::set_var("HOME", &path);
        Self { _guard: guard, path, old_home }
    }
}

impl Drop for IsolatedHome {
    fn drop(&mut self) {
        match &self.old_home {
            Some(h) => std::env::set_var("HOME", h),
            None => std::env::remove_var("HOME"),
        }
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[test]
fn get_config_returns_defaults() {
    let _h = IsolatedHome::new("defaults");
    let ctx = DispatchContext::new(None, None);
    let json = ctx.get_config();
    assert!(json.contains("\"temperature\":"), "missing temperature: {json}");
    assert!(json.contains("\"has_api_key\":false"), "missing has_api_key: {json}");
}

#[test]
fn get_config_no_engine_returns_none_model() {
    let _h = IsolatedHome::new("none-model");
    let ctx = DispatchContext::new(None, None);
    let json = ctx.get_config();
    assert!(json.contains("\"model\":\"none\""), "expected model:none: {json}");
}

#[test]
fn update_config_system_prompt() {
    let _h = IsolatedHome::new("sys-prompt");
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"system_prompt": "Be helpful."}"#);
    let json = ctx.get_config();
    assert!(json.contains("Be helpful."), "system_prompt not updated: {json}");
}

#[test]
fn update_config_recall_level() {
    let _h = IsolatedHome::new("recall-level");
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"recall_level": 5}"#);
    let json = ctx.get_config();
    assert!(json.contains("\"recall_level\":5"), "recall_level not updated: {json}");
}

#[test]
fn update_config_partial_preserves_other_fields() {
    let _h = IsolatedHome::new("partial-update");
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"recall_level": 7}"#);
    let after = ctx.get_config();
    // system_prompt should still be present (default)
    assert!(after.contains("\"system_prompt\":"), "system_prompt lost: {after}");
    assert!(after.contains("\"recall_level\":7"), "recall_level not set: {after}");
    assert!(after.contains("\"has_api_key\":false"), "has_api_key changed: {after}");
}

#[test]
fn store_api_key_creates_client() {
    let _h = IsolatedHome::new("store-key");
    let mut ctx = DispatchContext::new(None, None);
    assert!(ctx.get_config().contains("\"has_api_key\":false"), "should start without key");
    ctx.store_api_key("sk-ant-test-key");
    assert!(ctx.get_config().contains("\"has_api_key\":true"), "key not stored");
}

#[test]
fn update_cloud_model_with_client() {
    let _h = IsolatedHome::new("cloud-model");
    let mut ctx = DispatchContext::new(Some("sk-test".to_string()), None);
    ctx.update_config(r#"{"cloud_model": "claude-sonnet-4-6"}"#);
    let json = ctx.get_config();
    assert!(json.contains("claude-sonnet-4-6"), "cloud_model not updated: {json}");
}

/// Regression: once an API key is applied, a failing cloud request must
/// surface the real error (so the user can see *why* it failed), not the
/// generic "No LLM backend available" message — which made a configured-
/// but-failing backend indistinguishable from no backend at all.
///
/// The fake key can never authenticate, so `client.generate` always returns
/// Err (HTTP 401 with network, connection error without). Either way the
/// dispatch must report a cloud failure, never the no-backend message.
#[test]
fn cloud_failure_surfaces_real_error_not_no_backend() {
    let _h = IsolatedHome::new("cloud-fail-surfaces");
    let mut ctx = DispatchContext::new(None, None);
    ctx.store_api_key("sk-ant-invalid-test-key");
    assert!(ctx.get_config().contains("\"has_api_key\":true"), "key not applied");

    let resp = ctx.dispatch("hello there");
    assert!(
        resp.text.contains("Cloud inference failed"),
        "should surface the real cloud error, got: {}", resp.text
    );
    assert!(
        !resp.text.contains("No LLM backend available"),
        "must not mask a configured backend as missing, got: {}", resp.text
    );
}

/// Regression: the default cloud-fallback model must be a currently-served
/// model id. `claude-3-5-haiku-latest` aliased `claude-3-5-haiku-20241022`,
/// retired 2026-02-19, so the API rejected it with "model: claude-3-5-haiku-latest"
/// the moment a key was configured. The default is now `claude-haiku-4-5`.
#[test]
fn default_cloud_model_is_currently_served() {
    let _h = IsolatedHome::new("default-cloud-model");
    let mut ctx = DispatchContext::new(None, None);
    ctx.store_api_key("sk-ant-test-key");
    let json = ctx.get_config();
    assert!(
        json.contains("\"cloud_model\":\"claude-haiku-4-5\""),
        "default cloud model should be claude-haiku-4-5, got: {json}"
    );
    assert!(
        !json.contains("claude-3-5-haiku"),
        "retired haiku-3.5 id must not appear, got: {json}"
    );
}

#[test]
fn store_api_key_then_update_cloud_model() {
    let _h = IsolatedHome::new("store-then-update");
    let mut ctx = DispatchContext::new(None, None);
    ctx.store_api_key("sk-ant-test-key");
    ctx.update_config(r#"{"cloud_model": "claude-sonnet-4-6"}"#);
    let json = ctx.get_config();
    assert!(json.contains("\"has_api_key\":true"), "key missing: {json}");
    assert!(json.contains("claude-sonnet-4-6"), "cloud_model not updated: {json}");
}
