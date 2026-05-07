//! Tests for `--strict` mode: DispatchContext::new_strict() and the
//! refusal behavior at the LLM-fallback edge.

use olorin::core::router::{DispatchContext, STRICT_REFUSAL};

#[test]
fn strict_context_refuses_llm_fallback() {
    olorin::kernels::ffi::init().expect("kernel init");
    let mut ctx = DispatchContext::new_strict(None);
    // "hello world" matches no slash, no intent, no rune → would normally
    // fall through to the LLM. In strict mode that fallback is refused
    // with the known message.
    let resp = ctx.dispatch("hello world");
    assert_eq!(
        resp.text, STRICT_REFUSAL,
        "strict mode must refuse LLM fallback with the known message; got: {}",
        resp.text
    );
    assert!(!resp.blocked, "refusal is not a safety block, just a mode refusal");
}

#[test]
fn strict_context_still_runs_slash_commands() {
    olorin::kernels::ffi::init().expect("kernel init");
    let mut ctx = DispatchContext::new_strict(None);
    // /help is deterministic — works regardless of strict mode.
    let resp = ctx.dispatch("/help");
    assert!(resp.text.contains("Commands:"), "help should run in strict: {}", resp.text);
    assert!(resp.text.contains("strict"), "help should announce strict mode: {}", resp.text);
}

#[test]
fn strict_context_runs_calc_intent() {
    olorin::kernels::ffi::init().expect("kernel init");
    let mut ctx = DispatchContext::new_strict(None);
    // /calc is a tool — deterministic path, no LLM needed.
    let resp = ctx.dispatch("/calc 2+2");
    assert!(resp.text.contains('4'), "calc should produce 4 in strict: {}", resp.text);
    assert!(!resp.text.contains(STRICT_REFUSAL),
        "deterministic path must not hit the refusal: {}", resp.text);
}

#[test]
fn non_strict_context_still_loads_normally() {
    // Regression: the new field doesn't break the non-strict constructor.
    olorin::kernels::ffi::init().expect("kernel init");
    let ctx = DispatchContext::new(None, Some("nonexistent-model-name-for-test"));
    let sp = ctx.system_prompt_for_test();
    assert!(sp.contains("eacrunch") || sp.contains("eajson"),
        "non-strict context must keep the runes block in its system prompt");
}
