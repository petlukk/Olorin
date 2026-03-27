//! End-to-end integration tests: Vault + Tool-call detection
//!
//! Run with: cargo test -p olorin-core --test integration_vault_and_tools -- --nocapture

// ============================================================
// VAULT: Full lifecycle — create, write, search, reopen
// ============================================================

#[test]
fn test_vault_full_lifecycle() {
    use olorin_core::vault::{Vault, EachachaCrypto, find_chacha_lib};

    let dir = tempfile::tempdir().unwrap();
    let vault_path = dir.path().join("lifecycle.vault");
    let key = [42u8; 32];

    // Phase 1: Create vault and write a conversation
    {
        let mut vault = Vault::create(&vault_path, &key, Box::new(EachachaCrypto::new(find_chacha_lib().expect("libchacha20.so not found")))).unwrap();

        vault.append_message("User: How do I optimize x86 SIMD code for AVX-512?").unwrap();
        vault.append_message("Olorin: Use 512-bit registers with zmm0-zmm31. Key intrinsics include _mm512_fmadd_ps for fused multiply-add.").unwrap();
        vault.flush().unwrap();

        vault.append_message("User: What about ARM NEON? How does it compare?").unwrap();
        vault.append_message("Olorin: NEON uses 128-bit registers (q0-q31). For float32, that's 4 elements per vector vs 16 in AVX-512.").unwrap();
        vault.flush().unwrap();

        vault.append_message("User: Explain Rust's ownership model.").unwrap();
        vault.append_message("Olorin: Rust enforces ownership rules at compile time. Each value has exactly one owner.").unwrap();
        vault.flush().unwrap();

        assert_eq!(vault.block_count(), 3);
        println!("[vault] Created with 3 encrypted blocks");
    }

    // Phase 2: Reopen and search
    {
        let mut vault = Vault::open(&vault_path, &key, Box::new(EachachaCrypto::new(find_chacha_lib().expect("libchacha20.so not found")))).unwrap();
        assert_eq!(vault.block_count(), 3);

        let results = vault.search("AVX-512 SIMD optimization x86", 3).unwrap();
        assert!(!results.is_empty(), "Search should find results for AVX-512");
        let top_text = String::from_utf8_lossy(&results[0].text);
        assert!(
            top_text.contains("AVX-512") || top_text.contains("x86") || top_text.contains("zmm"),
            "Top result should be about x86: {:.80}", top_text
        );
        println!("[vault] Search 'AVX-512': score={:.3}, match=yes", results[0].score);

        let arm_results = vault.search("ARM NEON registers 128-bit", 3).unwrap();
        assert!(!arm_results.is_empty());
        let arm_text = String::from_utf8_lossy(&arm_results[0].text);
        assert!(arm_text.contains("NEON") || arm_text.contains("ARM") || arm_text.contains("128"));
        println!("[vault] Search 'ARM NEON': score={:.3}, match=yes", arm_results[0].score);

        let rust_results = vault.search("ownership borrow Rust compile time", 3).unwrap();
        assert!(!rust_results.is_empty());
        println!("[vault] Search 'Rust ownership': score={:.3}, match=yes", rust_results[0].score);
    }

    // Phase 3: Reopen, append, search again
    {
        let mut vault = Vault::open(&vault_path, &key, Box::new(EachachaCrypto::new(find_chacha_lib().expect("libchacha20.so not found")))).unwrap();
        vault.append_message("User: Back to x86 — what about cache line alignment?").unwrap();
        vault.append_message("Olorin: x86 cache lines are 64 bytes. Align to 64-byte boundaries.").unwrap();
        vault.flush().unwrap();
        assert_eq!(vault.block_count(), 4);

        let results = vault.search("x86 cache line 64 bytes alignment", 4).unwrap();
        let top_text = String::from_utf8_lossy(&results[0].text);
        assert!(top_text.contains("cache") || top_text.contains("64"));
        println!("[vault] After append: search 'cache line' score={:.3}", results[0].score);
    }

    // Phase 4: decrypt_last_n for /teleport greeting
    {
        let mut vault = Vault::open(&vault_path, &key, Box::new(EachachaCrypto::new(find_chacha_lib().expect("libchacha20.so not found")))).unwrap();
        let last = vault.decrypt_last_n(2).unwrap();
        assert_eq!(last.len(), 2);
        let block2 = String::from_utf8_lossy(&last[0]);
        let block3 = String::from_utf8_lossy(&last[1]);
        println!("[vault] Last 2 blocks for /teleport:");
        println!("  [-2]: {:.60}...", block2);
        println!("  [-1]: {:.60}...", block3);
    }

    println!("=== VAULT LIFECYCLE PASSED ===");
}

// ============================================================
// TOOL-CALL DETECTION
// ============================================================

#[test]
fn test_tool_call_streaming_detection() {
    use olorin_core::llm::tool_parse::{StringToolCallDetector, StrDetectResult};

    let mut detector = StringToolCallDetector::new(512);

    let chunks = [
        "Let me check. ",
        "<tool_",
        "call>",
        r#"{"name":"calc","arguments":{"expr":"2+3"}}"#,
        "</tool_call>",
    ];

    let mut text_parts = Vec::new();
    let mut tool_json = None;

    for chunk in &chunks {
        match detector.feed(chunk) {
            StrDetectResult::Text(t) => text_parts.push(t),
            StrDetectResult::ToolCall(json) => tool_json = Some(json),
            StrDetectResult::Buffering => {}
            StrDetectResult::Aborted(buf) => text_parts.push(buf),
        }
    }

    let text = text_parts.join("");
    println!("[tools] Streamed text: '{}'", text.trim());
    assert!(text.contains("Let me check"), "Should capture text: {}", text);

    assert!(tool_json.is_some(), "Should detect tool call");
    let json = tool_json.unwrap();
    println!("[tools] Tool call JSON: {}", json);
    assert!(json.contains("calc"), "Should reference 'calc'");
    assert!(json.contains("2+3"), "Should contain expression");

    println!("=== TOOL-CALL STREAMING PASSED ===");
}

#[test]
fn test_tool_call_extract_from_full_response() {
    use olorin_core::llm::tool_parse::extract_tool_calls;
    use olorin_core::llm::ContentBlock;

    let text = r#"I'll look that up. <tool_call>{"name":"http","arguments":{"url":"https://example.com"}}</tool_call>"#;
    let response = extract_tool_calls(text);

    let mut has_text = false;
    let mut has_tool = false;

    for block in &response.content {
        match block {
            ContentBlock::Text { text } => {
                assert!(text.contains("look that up"), "Text: {}", text);
                has_text = true;
                println!("[tools] Extracted text: '{}'", text.trim());
            }
            ContentBlock::ToolUse { name, input, .. } => {
                assert_eq!(name, "http");
                assert_eq!(input["url"], "https://example.com");
                has_tool = true;
                println!("[tools] Tool: name={}, url={}", name, input["url"]);
            }
            _ => {}
        }
    }

    assert!(has_text, "Should have text block");
    assert!(has_tool, "Should have tool_use block");
    println!("=== TOOL-CALL EXTRACTION PASSED ===");
}

#[test]
fn test_no_false_positive_tool_calls() {
    use olorin_core::llm::tool_parse::{StringToolCallDetector, StrDetectResult};

    let mut detector = StringToolCallDetector::new(512);

    let chunks = [
        "The tool is useful. ",
        "You can use <angle brackets> freely. ",
        "This is not a tool_call.",
    ];

    for chunk in &chunks {
        match detector.feed(chunk) {
            StrDetectResult::ToolCall(_) => panic!("False positive tool call!"),
            _ => {}
        }
    }

    println!("[tools] No false positives in normal text");
    println!("=== NO FALSE POSITIVE PASSED ===");
}
