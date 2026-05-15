//! Safety pipeline — injection detection (score-based, multi-language) and secret leak prevention.
//!
//! Inbound scan combines:
//! 1. Fused SIMD candidate finder (`scan_safety_fused`) — used by leak detection.
//! 2. Score-based injection matcher over two normalized forms of the input:
//!    spaceful (alnum runs separated by single spaces) and spaceless
//!    (alnum-only).  Each weighted pattern hit accumulates a score; the
//!    input is blocked when total score >= `INJECT_THRESHOLD` (2).
//!
//! Pattern weights: 2 = strong (single match blocks); 1 = weak (needs a second
//! signal).  This makes everyday speech ("forget about that") harmless while
//! still catching the "act as [privileged role]" / "Acting as a system admin"
//! shapes via two-weak-signal aggregation.

use crate::kernels::ffi;

// ── Public API ────────────────────────────────────────────────────────────────

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

const INJECT_THRESHOLD: u32 = 2;

/// Inbound safety scan: leak detection (SIMD-assisted) + score-based injection match.
pub fn scan(input: &[u8]) -> ScanResult {
    if input.is_empty() {
        return ScanResult { blocked: false, has_leak: false, details: Vec::new() };
    }

    let mut details = Vec::new();
    run_leak_scan(input, &mut details);
    let has_leak = !details.is_empty();

    let mut score = 0u32;
    let mut hit_descs: Vec<&str> = Vec::with_capacity(8);

    // Pass 1: Special tokens — case-insensitive byte match on raw input.
    // These contain non-alnum chars that normalization would strip.
    for pat in SPECIAL_PATTERNS {
        if let Some(pos) = find_case_insensitive(input, pat.bytes) {
            if !hit_descs.contains(&pat.desc) {
                hit_descs.push(pat.desc);
                score = score.saturating_add(pat.score);
                details.push(SafetyWarning {
                    kind: WarningKind::Injection,
                    pattern: pat.desc,
                    position: pos,
                });
            }
        }
    }

    // Pass 2: Normalized matching against the weighted pattern set.
    let spaceful = normalize_spaceful(input);
    let spaceless: Vec<u8> = spaceful.iter().copied().filter(|&b| b != b' ').collect();
    for pat in normalized_patterns() {
        if hit_descs.contains(&pat.desc) {
            continue;
        }
        let matched = if pat.has_space {
            substring(&spaceful, &pat.spaceful) || substring(&spaceless, &pat.spaceless)
        } else {
            word_match(&spaceful, &pat.spaceful)
        };
        if matched {
            hit_descs.push(pat.desc);
            score = score.saturating_add(pat.score);
            details.push(SafetyWarning {
                kind: WarningKind::Injection,
                pattern: pat.desc,
                position: 0,
            });
        }
    }

    let blocked = score >= INJECT_THRESHOLD || has_leak;
    ScanResult { blocked, has_leak, details }
}

/// Outbound safety scan — only checks for secret leaks.
/// Injection patterns are expected in LLM output (ChatML headers etc.)
/// and must NOT trigger blocking.
pub fn scan_outbound(input: &[u8]) -> ScanResult {
    if input.is_empty() {
        return ScanResult { blocked: false, has_leak: false, details: Vec::new() };
    }

    let mut details = Vec::new();
    run_leak_scan(input, &mut details);

    for &(needle, desc) in OUTBOUND_ONLY_LEAKS {
        if let Some(pos) = find_needle(input, needle) {
            details.push(SafetyWarning {
                kind: WarningKind::SecretLeak,
                pattern: desc,
                position: pos,
            });
        }
    }

    let has_leak = !details.is_empty();
    ScanResult { blocked: has_leak, has_leak, details }
}

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

// ── Injection patterns ────────────────────────────────────────────────────────

struct WeightedPattern {
    pattern: &'static str,
    score: u32,
    desc: &'static str,
}

const WEIGHTED_PATTERNS: &[WeightedPattern] = &[
    // English strong (2) — multi-word injection language with low benign overlap.
    WeightedPattern { pattern: "ignore previous",         score: 2, desc: "ignore previous instructions" },
    WeightedPattern { pattern: "ignore all previous",     score: 2, desc: "ignore all previous instructions" },
    WeightedPattern { pattern: "ignored previous",        score: 2, desc: "ignored previous (variant)" },
    WeightedPattern { pattern: "ignoring previous",       score: 2, desc: "ignoring previous (variant)" },
    WeightedPattern { pattern: "forget everything",       score: 2, desc: "reset context" },
    WeightedPattern { pattern: "forget all",              score: 2, desc: "reset context (variant)" },
    WeightedPattern { pattern: "forgetting all",          score: 2, desc: "reset context (variant)" },
    WeightedPattern { pattern: "you are now",             score: 2, desc: "role change" },
    WeightedPattern { pattern: "were now",                score: 2, desc: "role change (past)" },
    WeightedPattern { pattern: "pretend to be",           score: 2, desc: "role manipulation" },
    WeightedPattern { pattern: "pretending to be",        score: 2, desc: "role manipulation (variant)" },
    WeightedPattern { pattern: "new instructions",        score: 2, desc: "new instruction attempt" },
    WeightedPattern { pattern: "updated instructions",    score: 2, desc: "instruction update" },
    WeightedPattern { pattern: "these are updated",       score: 2, desc: "updated instruction phrasing" },

    // English weak (1) — single-word triggers. Word-boundary matched.
    WeightedPattern { pattern: "ignore",       score: 1, desc: "ignore (weak)" },
    WeightedPattern { pattern: "ignored",      score: 1, desc: "ignored (weak)" },
    WeightedPattern { pattern: "ignoring",     score: 1, desc: "ignoring (weak)" },
    WeightedPattern { pattern: "disregard",    score: 2, desc: "disregard" },
    WeightedPattern { pattern: "disregards",   score: 2, desc: "disregards (variant)" },
    WeightedPattern { pattern: "disregarded",  score: 2, desc: "disregarded (variant)" },
    WeightedPattern { pattern: "disregarding", score: 2, desc: "disregarding (variant)" },
    WeightedPattern { pattern: "forget",       score: 1, desc: "forget (weak)" },
    WeightedPattern { pattern: "forgetting",   score: 1, desc: "forgetting (weak)" },
    WeightedPattern { pattern: "forgot",       score: 1, desc: "forgot (weak)" },
    WeightedPattern { pattern: "act as",       score: 1, desc: "act as (weak)" },
    WeightedPattern { pattern: "acting as",    score: 1, desc: "acting as (weak)" },
    WeightedPattern { pattern: "pretend",      score: 1, desc: "pretend (weak)" },
    WeightedPattern { pattern: "pretending",   score: 1, desc: "pretending (weak)" },
    WeightedPattern { pattern: "system",       score: 1, desc: "system (weak)" },

    // Swedish strong (2) — multi-word.
    WeightedPattern { pattern: "ignorera tidigare",         score: 2, desc: "ignorera tidigare" },
    WeightedPattern { pattern: "ignorera alla tidigare",    score: 2, desc: "ignorera alla tidigare" },
    WeightedPattern { pattern: "glöm allt",                 score: 2, desc: "glöm allt" },
    WeightedPattern { pattern: "du är nu",                  score: 2, desc: "du är nu (role change)" },
    WeightedPattern { pattern: "låtsas vara",               score: 2, desc: "låtsas vara" },
    WeightedPattern { pattern: "agera som",                 score: 2, desc: "agera som" },
    WeightedPattern { pattern: "nya instruktioner",         score: 2, desc: "nya instruktioner" },
    WeightedPattern { pattern: "uppdaterade instruktioner", score: 2, desc: "uppdaterade instruktioner" },
    WeightedPattern { pattern: "strunta i",                 score: 2, desc: "strunta i (disregard)" },
    WeightedPattern { pattern: "föregående instruktioner",  score: 2, desc: "föregående instruktioner" },

    // Swedish weak (1).
    WeightedPattern { pattern: "ignorera", score: 1, desc: "ignorera (weak)" },
    WeightedPattern { pattern: "glöm",     score: 1, desc: "glöm (weak)" },
];

struct SpecialPattern {
    bytes: &'static [u8],
    score: u32,
    desc: &'static str,
}

const SPECIAL_PATTERNS: &[SpecialPattern] = &[
    SpecialPattern { bytes: b"system:",     score: 2, desc: "system: prefix" },
    SpecialPattern { bytes: b"assistant:",  score: 2, desc: "assistant: prefix" },
    SpecialPattern { bytes: b"user:",       score: 2, desc: "user: prefix" },
    SpecialPattern { bytes: b"<|",          score: 2, desc: "ChatML token start" },
    SpecialPattern { bytes: b"|>",          score: 2, desc: "ChatML token end" },
    SpecialPattern { bytes: b"[INST]",      score: 2, desc: "INST tag" },
    SpecialPattern { bytes: b"[/INST]",     score: 2, desc: "/INST tag" },
];

const CHATML_PATTERNS: &[&[u8]] = &[
    b"<|im_start|>",
    b"<|im_end|>",
    b"<|end_header_id|>",
    b"<|start_header_id|>",
    b"<|eot_id|>",
    b"[INST]",
    b"[/INST]",
];

// ── Normalization ─────────────────────────────────────────────────────────────

/// Lowercase via Unicode rules, keep alphanumerics, replace any run of
/// non-alphanumerics (whitespace + punctuation) with a single ASCII space.
/// Invalid UTF-8 falls back to empty (caller treats as no-match).
fn normalize_spaceful(input: &[u8]) -> Vec<u8> {
    let s = match std::str::from_utf8(input) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let lower = s.to_lowercase();
    let mut out = Vec::with_capacity(lower.len());
    let mut last_was_space = true;
    for c in lower.chars() {
        if c.is_alphanumeric() {
            let mut buf = [0u8; 4];
            let bytes = c.encode_utf8(&mut buf).as_bytes();
            out.extend_from_slice(bytes);
            last_was_space = false;
        } else if !last_was_space {
            out.push(b' ');
            last_was_space = true;
        }
    }
    if out.last() == Some(&b' ') {
        out.pop();
    }
    out
}

struct NormalizedPattern {
    spaceful: Vec<u8>,
    spaceless: Vec<u8>,
    score: u32,
    desc: &'static str,
    has_space: bool,
}

fn normalized_patterns() -> &'static [NormalizedPattern] {
    static CACHE: std::sync::OnceLock<Vec<NormalizedPattern>> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| {
        WEIGHTED_PATTERNS
            .iter()
            .map(|p| {
                let spaceful = normalize_spaceful(p.pattern.as_bytes());
                let has_space = spaceful.contains(&b' ');
                let spaceless: Vec<u8> = if has_space {
                    spaceful.iter().copied().filter(|&b| b != b' ').collect()
                } else {
                    Vec::new()
                };
                NormalizedPattern {
                    spaceful,
                    spaceless,
                    score: p.score,
                    desc: p.desc,
                    has_space,
                }
            })
            .collect()
    })
}

// ── Matchers ──────────────────────────────────────────────────────────────────

fn substring(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    (0..=haystack.len() - needle.len()).any(|i| &haystack[i..i + needle.len()] == needle)
}

/// Match a single-token pattern at word boundaries within a spaceful normalized
/// haystack (words separated by single ASCII spaces).  Pattern must be
/// alnum-only (no spaces).
fn word_match(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    let end = haystack.len() - needle.len();
    for i in 0..=end {
        if &haystack[i..i + needle.len()] != needle {
            continue;
        }
        let before_ok = i == 0 || haystack[i - 1] == b' ';
        let after_pos = i + needle.len();
        let after_ok = after_pos == haystack.len() || haystack[after_pos] == b' ';
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn find_case_insensitive(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).find(|&i| {
        haystack[i..i + needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
    })
}

// ── Leak detection (unchanged from v1.1.x) ────────────────────────────────────

const OUTBOUND_ONLY_LEAKS: &[(&[u8], &'static str)] = &[
    (b"OLRN",     "Olorin vault magic"),
    (b".olorin/", "Olorin internal path"),
];

fn find_needle(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len())
        .find(|&i| &haystack[i..i + needle.len()] == needle)
}

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

fn run_leak_scan(input: &[u8], details: &mut Vec<SafetyWarning>) {
    let len = input.len() as i32;
    let n_blocks = (input.len() + 15) / 16;
    let mut inject_masks = vec![0i32; n_blocks];
    let mut leak_masks = vec![0i32; n_blocks];
    let mut n_out = 0i32;

    unsafe {
        ffi::scan_safety_fused(
            input.as_ptr(),
            len,
            inject_masks.as_mut_ptr(),
            leak_masks.as_mut_ptr(),
            &mut n_out,
        );
    }

    let mut checked = std::collections::HashSet::new();
    for_each_candidate(&leak_masks, n_out as usize, |pos| {
        if checked.insert(pos) {
            verify_leak_at(input, pos, details);
        }
    });
    let simd_covered = (input.len() / 16) * 16;
    for pos in simd_covered..input.len() {
        if checked.insert(pos) {
            verify_leak_at(input, pos, details);
        }
    }
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
