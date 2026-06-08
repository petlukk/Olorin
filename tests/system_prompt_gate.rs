//! Guard the architecture gate on the chat system prompt.
//!
//! aarch64 (the Pi) can't emit `<tool_call>` XML on the NEON forward pass
//! (0/16), so it ships a minimal identity prompt instead of the ~2.6 KB runes
//! tools block — cutting ~30s of dead prefill per chat turn. x86_64 emits tool
//! calls (16/16) and keeps the full block for autonomous tool-calling. This
//! test runs on both CI arches and asserts each gets the right prompt.

use olorin::core::router::DispatchContext;

#[test]
fn chat_system_prompt_is_arch_gated() {
    olorin::kernels::ffi::init().unwrap();
    // Strict build: no engine, no API key — just exercises the prompt gate.
    let ctx = DispatchContext::new_strict(None);
    let sp = ctx.system_prompt_for_test();

    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(
            sp,
            olorin::core::llm::MINIMAL_SYSTEM_PROMPT,
            "aarch64 must use the minimal prompt — NEON can't emit tool calls",
        );
        assert!(
            !sp.contains("<tools>"),
            "aarch64 must not ship the dead-weight tools block",
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        assert!(
            sp.contains("<tools>"),
            "x86_64 keeps the full runes block for autonomous tool-calling",
        );
    }
}
