use crate::kernels::{fused_safety, leak_scanner};
use crate::safety::{SafetyWarning, WarningKind};


/// Leak patterns to verify at candidate positions.
/// Each entry: (prefix, min_length, validator_fn_name, description).
struct LeakPattern {
    prefix: &'static [u8],
    min_total_len: usize,
    description: &'static str,
    validate: fn(&[u8]) -> bool,
}

const LEAK_PATTERNS: &[LeakPattern] = &[
    LeakPattern {
        prefix: b"sk-ant-api",
        min_total_len: 20,
        description: "Anthropic API key",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"sk-proj-",
        min_total_len: 20,
        description: "OpenAI API key (project)",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"sk-",
        min_total_len: 20,
        description: "OpenAI API key",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"AKIA",
        min_total_len: 20,
        description: "AWS access key",
        validate: validate_upper_alnum,
    },
    LeakPattern {
        prefix: b"ghp_",
        min_total_len: 40,
        description: "GitHub personal access token",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"gho_",
        min_total_len: 40,
        description: "GitHub OAuth token",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"ghu_",
        min_total_len: 40,
        description: "GitHub user token",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"ghs_",
        min_total_len: 40,
        description: "GitHub server token",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"ghr_",
        min_total_len: 40,
        description: "GitHub refresh token",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"github_pat_",
        min_total_len: 40,
        description: "GitHub fine-grained PAT",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"xoxb-",
        min_total_len: 15,
        description: "Slack bot token",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"xoxp-",
        min_total_len: 15,
        description: "Slack user token",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"xoxa-",
        min_total_len: 15,
        description: "Slack app token",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"SG.",
        min_total_len: 40,
        description: "SendGrid API key",
        validate: validate_alnum_dash_dot,
    },
    LeakPattern {
        prefix: b"sk_live_",
        min_total_len: 24,
        description: "Stripe live API key",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"sk_test_",
        min_total_len: 24,
        description: "Stripe test API key",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"sess_",
        min_total_len: 32,
        description: "Session token",
        validate: validate_alnum_underscore,
    },
    LeakPattern {
        prefix: b"-----BEGIN",
        min_total_len: 10,
        description: "PEM private key",
        validate: |_| true,
    },
    LeakPattern {
        prefix: b"AIza",
        min_total_len: 39,
        description: "Google API key",
        validate: validate_alnum_dash,
    },
    LeakPattern {
        prefix: b"Bearer ",
        min_total_len: 27,
        description: "Bearer token",
        validate: |_| true,
    },
];

fn validate_alnum_dash(tail: &[u8]) -> bool {
    tail.iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn validate_upper_alnum(tail: &[u8]) -> bool {
    tail.iter()
        .all(|&b| b.is_ascii_uppercase() || b.is_ascii_digit())
}

fn validate_alnum_underscore(tail: &[u8]) -> bool {
    tail.iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'_')
}

fn validate_alnum_dash_dot(tail: &[u8]) -> bool {
    tail.iter()
        .all(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
}

pub struct LeakDetector;

impl LeakDetector {
    pub fn new() -> Self {
        Self
    }

    /// SIMD filter + scalar verify for secret patterns.
    pub fn scan(&self, text: &[u8]) -> Vec<SafetyWarning> {
        if text.is_empty() {
            return vec![];
        }

        let masks = leak_scanner::scan_prefixes(text);
        let mut warnings = Vec::new();
        let mut checked = std::collections::HashSet::new();

        leak_scanner::for_each_candidate(&masks, |pos| {
            if !checked.insert(pos) {
                return;
            }
            self.verify_at(text, pos, &mut warnings);
        });

        // Handle tail bytes
        let simd_covered = (text.len() / 16) * 16;
        for pos in simd_covered..text.len() {
            self.verify_at(text, pos, &mut warnings);
        }

        warnings
    }

    /// Verify leak candidates from pre-computed fused masks.
    pub fn verify_from_masks(&self, masks: &[i32], text: &[u8]) -> Vec<SafetyWarning> {
        if text.is_empty() {
            return vec![];
        }

        let mut warnings = Vec::new();
        let mut checked = std::collections::HashSet::new();

        fused_safety::for_each_candidate(masks, |pos| {
            if !checked.insert(pos) {
                return;
            }
            self.verify_at(text, pos, &mut warnings);
        });

        // Handle tail bytes
        let simd_covered = (text.len() / 16) * 16;
        for pos in simd_covered..text.len() {
            self.verify_at(text, pos, &mut warnings);
        }

        warnings
    }

    fn verify_at(&self, text: &[u8], pos: usize, warnings: &mut Vec<SafetyWarning>) {
        for pattern in LEAK_PATTERNS {
            let prefix = pattern.prefix;
            if pos + prefix.len() > text.len() {
                continue;
            }
            if &text[pos..pos + prefix.len()] != prefix {
                continue;
            }
            // Check minimum total length
            let remaining = &text[pos + prefix.len()..];
            let tail_needed = pattern.min_total_len.saturating_sub(prefix.len());
            if remaining.len() < tail_needed {
                continue;
            }
            // Find end of token (next whitespace or end)
            let tail_end = remaining
                .iter()
                .position(|&b| b.is_ascii_whitespace())
                .unwrap_or(remaining.len());
            let tail = &remaining[..tail_end];
            if tail.len() < tail_needed {
                continue;
            }
            if (pattern.validate)(tail) {
                warnings.push(SafetyWarning {
                    kind: WarningKind::SecretLeak,
                    pattern: pattern.description.to_string(),
                    position: pos,
                });
                return; // One warning per position
            }
        }
    }
}

impl Default for LeakDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text() {
        let d = LeakDetector::new();
        let warnings = d.scan(b"Hello, this is a normal message with no secrets.");
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_openai_key() {
        let d = LeakDetector::new();
        let input = b"my key is sk-1234567890abcdefghij1234567890ab";
        let warnings = d.scan(input);
        assert!(!warnings.is_empty());
        assert_eq!(warnings[0].kind, WarningKind::SecretLeak);
    }

    #[test]
    fn test_aws_key() {
        let d = LeakDetector::new();
        let input = b"AWS key: AKIAIOSFODNN7EXAMPLE";
        let warnings = d.scan(input);
        assert!(!warnings.is_empty());
        assert!(warnings[0].pattern.contains("AWS"));
    }

    #[test]
    fn test_github_token() {
        let d = LeakDetector::new();
        let input = b"token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn";
        let warnings = d.scan(input);
        assert!(!warnings.is_empty());
        assert!(warnings[0].pattern.contains("GitHub"));
    }

    #[test]
    fn test_pem_key() {
        let d = LeakDetector::new();
        let input = b"-----BEGIN RSA PRIVATE KEY-----\nMIIEpA...";
        let warnings = d.scan(input);
        assert!(!warnings.is_empty());
        assert!(warnings[0].pattern.contains("PEM"));
    }

    #[test]
    fn test_short_sk_no_match() {
        let d = LeakDetector::new();
        // "sk-" followed by too few chars shouldn't match
        let input = b"sk-short";
        let warnings = d.scan(input);
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_empty() {
        let d = LeakDetector::new();
        assert!(d.scan(b"").is_empty());
    }
}
