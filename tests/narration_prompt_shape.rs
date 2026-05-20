//! Regression: the narration prompt must not embed a trailing instruction
//! that the model can mistake for content to continue.
//!
//! Background — observed on Pi 5 with gemma-4-e2b-q3kffnimpl on 2026-05-20:
//! For repetitive-pattern rune outputs (eatime 24 hour-buckets, eajson 28
//! key-lines), the model echoed the trailing instruction template
//! "In 1-2 plain-English sentences, tell me what stands out about this
//! data."  Sometimes hallucinated markdown bolding around it
//! (**...**) and repeated it multiple times.  Root cause: the user prompt
//! redundantly repeated the system prompt's instruction at the end.
//!
//! Fix: keep the user prompt to just `Output of <rune>:\n\n<data>` and
//! let the system prompt carry all instruction load.  This test locks
//! that contract in.

use olorin::runes::{build_narration_prompt, OutputSafety, RuneResult};

fn make_result(answer: &str) -> RuneResult {
    RuneResult {
        answer:     answer.to_string(),
        details:    None,
        success:    true,
        timing_us:  0,
        structured: false,
    }
}

#[test]
fn prompt_does_not_carry_trailing_instruction() {
    let r = make_result("rows: 30\ncategory: stuff\n");
    let p = build_narration_prompt("eajson", OutputSafety::Trusted, r)
        .expect("narration prompt expected for plain-text rune output");

    // The old prompt ended with these template fragments — make sure
    // they never reappear, by-string, anywhere in the user message.
    let banned = [
        "In 1-2 plain-English sentences",
        "tell me what stands out",
        "Do not repeat the raw numbers",
        "surface insights",
    ];
    for needle in banned {
        assert!(
            !p.contains(needle),
            "prompt contains banned trailing-instruction fragment `{needle}`:\n{p}"
        );
    }
}

#[test]
fn prompt_starts_with_a_clear_label() {
    // The label format `Output of \`<name>\`:` is what the system prompt
    // expects to see; keeping it as the first non-empty line gives the
    // model an unambiguous start-of-content marker.
    let r = make_result("any data");
    let p = build_narration_prompt("eatime", OutputSafety::Trusted, r).unwrap();
    assert!(
        p.starts_with("Output of `eatime`:"),
        "prompt should start with the rune label, got:\n{p}"
    );
}

#[test]
fn structured_output_skips_narration() {
    // --json outputs were already skipped before the fix; keep that
    // contract locked.
    let mut r = make_result(r#"{"schema_version":1,"rune":"eatime"}"#);
    r.structured = true;
    let p = build_narration_prompt("eatime", OutputSafety::Trusted, r);
    assert!(
        p.is_none(),
        "structured rune output must not produce a narration prompt; got {p:?}"
    );
}

#[test]
fn rune_data_is_preserved_verbatim_in_user_prompt() {
    // Whatever the kernel emitted, the model must see it without paraphrase
    // or wrapping that hides the original bytes.
    let data = "rows: 99\nfoo:   42\nbar:   13\n";
    let r = make_result(data);
    let p = build_narration_prompt("eacrunch", OutputSafety::Trusted, r).unwrap();
    assert!(p.contains(data), "rune output must be passed through verbatim:\n{p}");
}

#[test]
fn long_repetitive_output_does_not_get_trailing_template() {
    // Simulates the eatime case that caused the original bug: 24
    // hour-bucket lines.  Confirm none of the old template fragments
    // sneak back in for long outputs either.
    let mut data = String::from("bytes:       130.1 KB\ntimestamps:  1867\nscan:        0 ms\n\nhour-of-day:\n");
    for h in 0..24 {
        data.push_str(&format!("  {h:02}:00       {}  ( {:.2}%)\n",
            if h == 7 { 1214 } else { 0 },
            if h == 7 { 65.0 } else { 0.0 }));
    }
    data.push_str("\npeak: 07:00 (1214 timestamps)\n");

    let r = make_result(&data);
    let p = build_narration_prompt("eatime", OutputSafety::Trusted, r).unwrap();

    assert!(!p.contains("plain-English"),  "leaked template: plain-English");
    assert!(!p.contains("stands out"),     "leaked template: stands out");
    assert!(!p.contains("surface insights"),"leaked template: surface insights");
    assert!(p.contains("hour-of-day"),     "data must be preserved");
}
