//! End-to-end wiring tests for the LLM tool-call path on runes.

use olorin::runes;

#[test]
fn runes_prompt_block_contains_eacrunch_name_and_description() {
    let block = runes::runes_prompt_block();
    assert!(block.contains("<tools>"), "missing opening <tools> tag");
    assert!(block.contains("</tools>"), "missing closing </tools> tag");
    assert!(
        block.contains("- eacrunch:"),
        "rune name bullet missing from prompt block"
    );
    assert!(
        block.to_lowercase().contains("csv"),
        "eacrunch description (which mentions csv) missing from block"
    );
    assert!(
        block.contains("<tool_call>"),
        "tool_call usage example missing from block"
    );
    assert!(
        block.contains("untrusted=\"true\""),
        "untrusted delimiter guidance missing — required for file-derived output"
    );
}

#[test]
fn runes_prompt_block_is_stable_across_calls() {
    let a = runes::runes_prompt_block();
    let b = runes::runes_prompt_block();
    // Same pointer: confirms OnceLock caching (no per-call rebuild).
    assert!(std::ptr::eq(a.as_ptr(), b.as_ptr()));
}

#[test]
fn dispatch_context_new_system_prompt_contains_rune_block() {
    let ctx = olorin::core::router::DispatchContext::new(None, None, None, None);
    let sp = ctx.system_prompt_for_test();
    assert!(sp.contains("- eacrunch:"), "rune block not composed into system_prompt");
    assert!(sp.contains("<tools>"), "tools opener missing in composed prompt");
}
