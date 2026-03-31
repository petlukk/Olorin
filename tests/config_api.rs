//! Tests for runtime config API — get/update/partial update.

use olorin::core::router::DispatchContext;

#[test]
fn get_config_returns_defaults() {
    olorin::kernels::ffi::init().unwrap();
    let ctx = DispatchContext::new(None, None);
    let json = ctx.get_config();
    assert!(json.contains("\"temperature\":"), "missing temperature: {json}");
    assert!(json.contains("\"has_api_key\":false"), "missing has_api_key: {json}");
}

#[test]
fn get_config_no_engine_returns_none_model() {
    olorin::kernels::ffi::init().unwrap();
    let ctx = DispatchContext::new(None, None);
    let json = ctx.get_config();
    assert!(json.contains("\"model\":\"none\""), "expected model:none: {json}");
}

#[test]
fn update_config_system_prompt() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"system_prompt": "Be helpful."}"#);
    let json = ctx.get_config();
    assert!(json.contains("Be helpful."), "system_prompt not updated: {json}");
}

#[test]
fn update_config_recall_level() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"recall_level": 5}"#);
    let json = ctx.get_config();
    assert!(json.contains("\"recall_level\":5"), "recall_level not updated: {json}");
}

#[test]
fn update_config_partial_preserves_other_fields() {
    olorin::kernels::ffi::init().unwrap();
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
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    assert!(ctx.get_config().contains("\"has_api_key\":false"), "should start without key");
    ctx.store_api_key("sk-ant-test-key");
    assert!(ctx.get_config().contains("\"has_api_key\":true"), "key not stored");
}

#[test]
fn update_cloud_model_with_client() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(Some("sk-test".to_string()), None);
    ctx.update_config(r#"{"cloud_model": "claude-sonnet-4-6"}"#);
    let json = ctx.get_config();
    assert!(json.contains("claude-sonnet-4-6"), "cloud_model not updated: {json}");
}

#[test]
fn store_api_key_then_update_cloud_model() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.store_api_key("sk-ant-test-key");
    ctx.update_config(r#"{"cloud_model": "claude-sonnet-4-6"}"#);
    let json = ctx.get_config();
    assert!(json.contains("\"has_api_key\":true"), "key missing: {json}");
    assert!(json.contains("claude-sonnet-4-6"), "cloud_model not updated: {json}");
}
