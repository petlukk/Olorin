use super::byte_classifier::FLAG_WS;

/// SIMD-accelerated argument tokenizer.
/// Uses byte_classifier to find whitespace in a single SIMD pass,
/// then extracts token boundaries from the flag buffer.
pub struct ArgTokenizer {
    flags: Vec<u8>,
}

/// Token boundaries as byte offsets into the original input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenSpan {
    pub start: usize,
    pub end: usize,
}

impl ArgTokenizer {
    pub fn new() -> Self {
        Self { flags: Vec::new() }
    }

    pub fn with_capacity(max_len: usize) -> Self {
        Self {
            flags: vec![0u8; max_len],
        }
    }

    /// Tokenize input into up to `max_tokens` spans.
    /// The last token captures everything remaining (like splitn).
    /// Returns token spans as byte ranges into the input.
    pub fn tokenize<'a>(&mut self, input: &'a [u8], max_tokens: usize) -> Vec<&'a [u8]> {
        if input.is_empty() || max_tokens == 0 {
            return Vec::new();
        }

        let len = input.len();

        // Resize flag buffer (reuses allocation)
        self.flags.resize(len, 0);

        // SIMD classify pass
        unsafe {
            super::ffi::classify_bytes(input.as_ptr(), self.flags.as_mut_ptr(), len as i32);
        }

        // Scan flags for token boundaries
        let mut tokens: Vec<&'a [u8]> = Vec::with_capacity(max_tokens);
        let mut i = 0;

        // Skip leading whitespace
        while i < len && self.flags[i] == FLAG_WS {
            i += 1;
        }

        while i < len && tokens.len() < max_tokens {
            let start = i;

            if tokens.len() == max_tokens - 1 {
                // Last token: capture everything remaining
                tokens.push(&input[start..]);
                break;
            }

            // Advance past non-whitespace
            while i < len && self.flags[i] != FLAG_WS {
                i += 1;
            }
            tokens.push(&input[start..i]);

            // Skip whitespace between tokens
            while i < len && self.flags[i] == FLAG_WS {
                i += 1;
            }
        }

        tokens
    }

    /// Convenience: tokenize a str, returning str slices.
    pub fn tokenize_str<'a>(&mut self, input: &'a str, max_tokens: usize) -> Vec<&'a str> {
        let byte_tokens = self.tokenize(input.as_bytes(), max_tokens);
        byte_tokens
            .into_iter()
            .map(|b| unsafe { std::str::from_utf8_unchecked(b) })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let mut tok = ArgTokenizer::new();
        assert!(tok.tokenize(b"", 3).is_empty());
    }

    #[test]
    fn test_single_token() {
        let mut tok = ArgTokenizer::new();
        let result = tok.tokenize(b"hello", 3);
        assert_eq!(result, vec![b"hello".as_slice()]);
    }

    #[test]
    fn test_two_tokens() {
        let mut tok = ArgTokenizer::new();
        let result = tok.tokenize(b"hello world", 3);
        assert_eq!(result, vec![b"hello".as_slice(), b"world".as_slice()]);
    }

    #[test]
    fn test_splitn_behavior() {
        let mut tok = ArgTokenizer::new();
        // With max_tokens=2, second token captures remainder
        let result = tok.tokenize(b"write key some long value", 3);
        assert_eq!(
            result,
            vec![
                b"write".as_slice(),
                b"key".as_slice(),
                b"some long value".as_slice()
            ]
        );
    }

    #[test]
    fn test_leading_whitespace() {
        let mut tok = ArgTokenizer::new();
        // With max_tokens=3, gets individual tokens (trailing WS stripped)
        let result = tok.tokenize(b"  hello  world  ", 3);
        assert_eq!(result, vec![b"hello".as_slice(), b"world".as_slice()]);
        // With max_tokens=2, last token captures remainder including trailing WS
        let result2 = tok.tokenize(b"  hello  world  ", 2);
        assert_eq!(result2, vec![b"hello".as_slice(), b"world  ".as_slice()]);
    }

    #[test]
    fn test_max_one() {
        let mut tok = ArgTokenizer::new();
        let result = tok.tokenize(b"hello world", 1);
        assert_eq!(result, vec![b"hello world".as_slice()]);
    }

    #[test]
    fn test_str_convenience() {
        let mut tok = ArgTokenizer::new();
        let result = tok.tokenize_str("echo hello world", 2);
        assert_eq!(result, vec!["echo", "hello world"]);
    }

    #[test]
    fn test_reuse_buffer() {
        let mut tok = ArgTokenizer::with_capacity(64);
        // First call
        let r1 = tok.tokenize(b"abc def", 3);
        assert_eq!(r1, vec![b"abc".as_slice(), b"def".as_slice()]);
        // Second call reuses buffer
        let r2 = tok.tokenize(b"longer input string here", 2);
        assert_eq!(
            r2,
            vec![b"longer".as_slice(), b"input string here".as_slice()]
        );
    }

    #[test]
    fn test_matches_splitn() {
        let mut tok = ArgTokenizer::new();
        let inputs = &[
            ("list", 3),
            ("read mykey", 3),
            ("write key some value with spaces", 3),
            ("get path.to.field", 2),
        ];
        for &(input, n) in inputs {
            let simd: Vec<&str> = tok.tokenize_str(input, n);
            let scalar: Vec<&str> = input.splitn(n, ' ').collect();
            assert_eq!(simd, scalar, "mismatch for input={input:?}, n={n}");
        }
    }
}
