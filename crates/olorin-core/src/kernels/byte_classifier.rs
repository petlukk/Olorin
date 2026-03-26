use super::ffi;

pub const FLAG_WS: u8 = 1;
pub const FLAG_LETTER: u8 = 2;
pub const FLAG_DIGIT: u8 = 4;
pub const FLAG_PUNCT: u8 = 8;
pub const FLAG_NONASCII: u8 = 16;

/// Classify every byte in `text` into flag bits.
/// Returns a Vec<u8> of flags, one per input byte.
pub fn classify(text: &[u8]) -> Vec<u8> {
    let len = text.len();
    let mut flags = vec![0u8; len];
    if len > 0 {
        unsafe {
            ffi::classify_bytes(text.as_ptr(), flags.as_mut_ptr(), len as i32);
        }
    }
    flags
}

/// Scalar reference implementation for testing.
pub fn classify_scalar(text: &[u8]) -> Vec<u8> {
    text.iter()
        .map(|&b| {
            if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                FLAG_WS
            } else if b.is_ascii_alphabetic() {
                FLAG_LETTER
            } else if b.is_ascii_digit() {
                FLAG_DIGIT
            } else if b > 127 {
                FLAG_NONASCII
            } else if (33..=126).contains(&b)
                && !b.is_ascii_alphabetic()
                && !b.is_ascii_digit()
            {
                FLAG_PUNCT
            } else {
                0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(classify(b""), Vec::<u8>::new());
    }

    #[test]
    fn test_basic_classification() {
        let input = b"Hello, World! 123\n";
        let simd = classify(input);
        let scalar = classify_scalar(input);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_all_whitespace() {
        let input = b"   \t\n\r  ";
        let result = classify(input);
        assert!(result.iter().all(|&f| f == FLAG_WS));
    }

    #[test]
    fn test_sub_vector_length() {
        let input = b"abc123!@#";
        let simd = classify(input);
        let scalar = classify_scalar(input);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_exact_vector_length() {
        let input = b"abcdefghijklmnop"; // 16 bytes
        let simd = classify(input);
        let scalar = classify_scalar(input);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_over_vector_length() {
        let input = b"abcdefghijklmnopq"; // 17 bytes
        let simd = classify(input);
        let scalar = classify_scalar(input);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_nonascii() {
        let input = "café résumé".as_bytes();
        let simd = classify(input);
        let scalar = classify_scalar(input);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_1kb_random() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let simd = classify(&input);
        let scalar = classify_scalar(&input);
        assert_eq!(simd, scalar);
    }
}
