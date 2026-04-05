//! Safety pipeline — injection detection and secret leak prevention.
//!
//! Merges sanitizer + leak_detector into one module.
//! Uses Eä SIMD kernels: scan_safety_fused, scan_injection_prefixes, scan_leak_prefixes.
//!
//! Hot path: ScanResult has `blocked` + `has_leak` for fast checks.
//! No allocations if input is empty.

use crate::kernels::ffi;

// ── Public API ────────────────────────────────────────────────────────────────

/// Result of a safety scan.
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub blocked: bool,
    pub has_leak: bool,
    pub details: Vec<SafetyWarning>,
}

#[derive(Debug, Clone)]
pub struct SafetyWarning {
    pub kind: WarningKind,
    pub pattern: &'static str,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningKind {
    Injection,
    SecretLeak,
}

/// Stateless scan: runs fused SIMD kernel then verifies candidates in Rust.
/// Allocates only when warnings are found.
pub fn scan(input: &[u8]) -> ScanResult {
    if input.is_empty() {
        return ScanResult { blocked: false, has_leak: false, details: Vec::new() };
    }

    let len = input.len() as i32;
    let n_blocks = (input.len() + 15) / 16;

    let mut inject_masks = vec![0i32; n_blocks];
    let mut leak_masks   = vec![0i32; n_blocks];
    let mut n_out        = 0i32;

    unsafe {
        ffi::scan_safety_fused(
            input.as_ptr(),
            len,
            inject_masks.as_mut_ptr(),
            leak_masks.as_mut_ptr(),
            &mut n_out,
        );
    }

    let mut details = Vec::new();

    // Verify injection candidates
    for_each_candidate(&inject_masks, n_out as usize, |pos| {
        verify_injection_at(input, pos, &mut details);
    });
    let simd_covered = if input.len() >= 17 {
        ((input.len() - 17) / 16 + 1) * 16
    } else {
        0
    };
    for pos in simd_covered..input.len() {
        verify_injection_at(input, pos, &mut details);
    }

    let injection_found = details.iter().any(|w| w.kind == WarningKind::Injection);

    // Verify leak candidates
    let mut checked = std::collections::HashSet::new();
    for_each_candidate(&leak_masks, n_out as usize, |pos| {
        if checked.insert(pos) {
            verify_leak_at(input, pos, &mut details);
        }
    });
    let leak_simd_covered = (input.len() / 16) * 16;
    for pos in leak_simd_covered..input.len() {
        if checked.insert(pos) {
            verify_leak_at(input, pos, &mut details);
        }
    }

    let has_leak = details.iter().any(|w| w.kind == WarningKind::SecretLeak);

    ScanResult {
        blocked: injection_found || has_leak,
        has_leak,
        details,
    }
}

/// Outbound safety scan — only checks for secret leaks.
/// Injection patterns are expected in LLM output (ChatML headers etc.)
/// and must NOT trigger blocking.
pub fn scan_outbound(input: &[u8]) -> ScanResult {
    if input.is_empty() {
        return ScanResult { blocked: false, has_leak: false, details: Vec::new() };
    }

    let len = input.len() as i32;
    let n_blocks = (input.len() + 15) / 16;

    let mut inject_masks = vec![0i32; n_blocks];
    let mut leak_masks   = vec![0i32; n_blocks];
    let mut n_out        = 0i32;

    unsafe {
        ffi::scan_safety_fused(
            input.as_ptr(),
            len,
            inject_masks.as_mut_ptr(),
            leak_masks.as_mut_ptr(),
            &mut n_out,
        );
    }

    let mut details = Vec::new();

    // Only verify leak candidates — skip injection entirely
    let mut checked = std::collections::HashSet::new();
    for_each_candidate(&leak_masks, n_out as usize, |pos| {
        if checked.insert(pos) {
            verify_leak_at(input, pos, &mut details);
        }
    });
    let leak_simd_covered = (input.len() / 16) * 16;
    for pos in leak_simd_covered..input.len() {
        if checked.insert(pos) {
            verify_leak_at(input, pos, &mut details);
        }
    }

    let has_leak = !details.is_empty();

    ScanResult {
        blocked: has_leak,
        has_leak,
        details,
    }
}

// ── Injection patterns ────────────────────────────────────────────────────────

const INJECTION_PATTERNS: &[(&[u8], &str)] = &[
    (b"ignore previous",      "override previous instructions"),
    (b"ignore all previous",  "override ALL previous instructions"),
    (b"disregard",            "potential instruction override"),
    (b"forget everything",    "attempt to reset context"),
    (b"you are now",          "role change attempt"),
    (b"act as",               "role manipulation"),
    (b"pretend to be",        "role manipulation"),
    (b"system:",              "system message injection"),
    (b"assistant:",           "assistant response injection"),
    (b"user:",                "user message injection"),
    (b"<|",                   "special token injection"),
    (b"|>",                   "special token injection"),
    (b"[INST]",               "instruction token injection"),
    (b"[/INST]",              "instruction token injection"),
    (b"new instructions",     "new instruction attempt"),
    (b"updated instructions", "instruction update attempt"),
];

fn verify_injection_at(text: &[u8], pos: usize, out: &mut Vec<SafetyWarning>) {
    for &(pattern, desc) in INJECTION_PATTERNS {
        if matches_case_insensitive(text, pos, pattern) {
            out.push(SafetyWarning {
                kind: WarningKind::Injection,
                pattern: desc,
                position: pos,
            });
            return;
        }
    }
}

fn matches_case_insensitive(text: &[u8], pos: usize, pattern: &[u8]) -> bool {
    if pos + pattern.len() > text.len() { return false; }
    text[pos..pos + pattern.len()]
        .iter()
        .zip(pattern.iter())
        .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
}

// ── Leak patterns ─────────────────────────────────────────────────────────────

struct LeakPattern {
    prefix: &'static [u8],
    min_total_len: usize,
    description: &'static str,
    validate: fn(&[u8]) -> bool,
}

const LEAK_PATTERNS: &[LeakPattern] = &[
    LeakPattern { prefix: b"sk-ant-api",   min_total_len: 20, description: "Anthropic API key",          validate: valid_alnum_dash },
    LeakPattern { prefix: b"sk-proj-",     min_total_len: 20, description: "OpenAI API key (project)",   validate: valid_alnum_dash },
    LeakPattern { prefix: b"sk-",          min_total_len: 20, description: "OpenAI API key",             validate: valid_alnum_dash },
    LeakPattern { prefix: b"AKIA",         min_total_len: 20, description: "AWS access key",             validate: valid_upper_alnum },
    LeakPattern { prefix: b"ghp_",         min_total_len: 40, description: "GitHub personal token",      validate: valid_alnum_underscore },
    LeakPattern { prefix: b"gho_",         min_total_len: 40, description: "GitHub OAuth token",         validate: valid_alnum_underscore },
    LeakPattern { prefix: b"ghu_",         min_total_len: 40, description: "GitHub user token",          validate: valid_alnum_underscore },
    LeakPattern { prefix: b"ghs_",         min_total_len: 40, description: "GitHub server token",        validate: valid_alnum_underscore },
    LeakPattern { prefix: b"ghr_",         min_total_len: 40, description: "GitHub refresh token",       validate: valid_alnum_underscore },
    LeakPattern { prefix: b"github_pat_",  min_total_len: 40, description: "GitHub fine-grained PAT",    validate: valid_alnum_underscore },
    LeakPattern { prefix: b"xoxb-",        min_total_len: 15, description: "Slack bot token",            validate: valid_alnum_dash },
    LeakPattern { prefix: b"xoxp-",        min_total_len: 15, description: "Slack user token",           validate: valid_alnum_dash },
    LeakPattern { prefix: b"xoxa-",        min_total_len: 15, description: "Slack app token",            validate: valid_alnum_dash },
    LeakPattern { prefix: b"SG.",          min_total_len: 40, description: "SendGrid API key",           validate: valid_alnum_dash_dot },
    LeakPattern { prefix: b"sk_live_",     min_total_len: 24, description: "Stripe live API key",        validate: valid_alnum_underscore },
    LeakPattern { prefix: b"sk_test_",     min_total_len: 24, description: "Stripe test API key",        validate: valid_alnum_underscore },
    LeakPattern { prefix: b"sess_",        min_total_len: 32, description: "Session token",              validate: valid_alnum_underscore },
    LeakPattern { prefix: b"-----BEGIN",   min_total_len: 10, description: "PEM private key",            validate: |_| true },
    LeakPattern { prefix: b"AIza",         min_total_len: 39, description: "Google API key",             validate: valid_alnum_dash },
    LeakPattern { prefix: b"Bearer ",      min_total_len: 27, description: "Bearer token",               validate: |_| true },
];

fn valid_alnum_dash(tail: &[u8]) -> bool {
    tail.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}
fn valid_upper_alnum(tail: &[u8]) -> bool {
    tail.iter().all(|&b| b.is_ascii_uppercase() || b.is_ascii_digit())
}
fn valid_alnum_underscore(tail: &[u8]) -> bool {
    tail.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_')
}
fn valid_alnum_dash_dot(tail: &[u8]) -> bool {
    tail.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
}

fn verify_leak_at(text: &[u8], pos: usize, out: &mut Vec<SafetyWarning>) {
    for pat in LEAK_PATTERNS {
        let prefix = pat.prefix;
        if pos + prefix.len() > text.len() { continue; }
        if &text[pos..pos + prefix.len()] != prefix { continue; }
        let remaining = &text[pos + prefix.len()..];
        let tail_needed = pat.min_total_len.saturating_sub(prefix.len());
        if remaining.len() < tail_needed { continue; }
        let tail_end = remaining.iter().position(|&b| b.is_ascii_whitespace())
            .unwrap_or(remaining.len());
        let tail = &remaining[..tail_end];
        if tail.len() < tail_needed { continue; }
        if (pat.validate)(tail) {
            out.push(SafetyWarning {
                kind: WarningKind::SecretLeak,
                pattern: pat.description,
                position: pos,
            });
            return;
        }
    }
}

// ── ChatML hallucination detection ───────────────────────────────────────────

const CHATML_PATTERNS: &[&[u8]] = &[
    b"<|im_start|>",
    b"<|im_end|>",
    b"<|end_header_id|>",
    b"<|start_header_id|>",
    b"<|eot_id|>",
    b"<|turn>",
    b"[INST]",
    b"[/INST]",
];

/// Returns true if the token looks like a ChatML/prompt header hallucination.
/// Used for aggressive trimming during streaming: if true, stop generation.
pub fn is_chatml_hallucination(token: &str) -> bool {
    let lower = token.as_bytes();
    for pat in CHATML_PATTERNS {
        if lower.len() >= pat.len() {
            let matches = lower[..pat.len()]
                .iter()
                .zip(pat.iter())
                .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase());
            if matches {
                return true;
            }
        }
    }
    false
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Iterate over bit-set positions in SIMD mask blocks.
fn for_each_candidate(masks: &[i32], n_blocks: usize, mut f: impl FnMut(usize)) {
    let count = n_blocks.min(masks.len());
    for block in 0..count {
        let mut m = masks[block] as u32;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            let pos = block * 16 + bit;
            f(pos);
            m &= m - 1;
        }
    }
}
