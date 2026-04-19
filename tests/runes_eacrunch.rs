use olorin::runes::{RuneResult, OutputSafety};

#[test]
fn rune_result_defaults() {
    let r = RuneResult {
        answer: "hello".into(),
        details: None,
        success: true,
        timing_us: 42,
    };
    assert!(r.success);
    assert_eq!(r.answer, "hello");
    assert_eq!(r.timing_us, 42);
}

#[test]
fn output_safety_variants() {
    let t = OutputSafety::Trusted;
    let u = OutputSafety::UntrustedQuoted;
    assert!(matches!(t, OutputSafety::Trusted));
    assert!(matches!(u, OutputSafety::UntrustedQuoted));
}
