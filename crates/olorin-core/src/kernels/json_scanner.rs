use super::ffi;

const STRUCTURAL: &[u8] = b"{}[]:,\"\\";

/// Count structural JSON characters in `text`.
pub fn count_structural(text: &[u8]) -> i32 {
    let len = text.len();
    if len == 0 {
        return 0;
    }
    let mut count: i32 = 0;
    unsafe {
        ffi::count_json_structural(text.as_ptr(), len as i32, &mut count);
    }
    count
}

/// Extract positions and types of structural JSON characters.
/// Returns (positions, types) vectors.
pub fn extract_structural(text: &[u8]) -> (Vec<i32>, Vec<u8>) {
    let len = text.len();
    if len == 0 {
        return (vec![], vec![]);
    }
    // Worst case: every byte is structural
    let mut positions = vec![0i32; len];
    let mut types = vec![0u8; len];
    let mut count: i32 = 0;
    unsafe {
        ffi::extract_json_structural(
            text.as_ptr(),
            len as i32,
            positions.as_mut_ptr(),
            types.as_mut_ptr(),
            &mut count,
        );
    }
    let n = count as usize;
    positions.truncate(n);
    types.truncate(n);
    (positions, types)
}

/// Scalar reference for counting.
pub fn count_structural_scalar(text: &[u8]) -> i32 {
    text.iter().filter(|b| STRUCTURAL.contains(b)).count() as i32
}

/// Scalar reference for extraction.
pub fn extract_structural_scalar(text: &[u8]) -> (Vec<i32>, Vec<u8>) {
    let mut positions = Vec::new();
    let mut types = Vec::new();
    for (i, &b) in text.iter().enumerate() {
        if STRUCTURAL.contains(&b) {
            positions.push(i as i32);
            types.push(b);
        }
    }
    (positions, types)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(count_structural(b""), 0);
        assert_eq!(extract_structural(b""), (vec![], vec![]));
    }

    #[test]
    fn test_simple_json() {
        let input = br#"{"key": "val"}"#;
        let count = count_structural(input);
        let scalar_count = count_structural_scalar(input);
        assert_eq!(count, scalar_count);

        let (pos, types) = extract_structural(input);
        let (spos, stypes) = extract_structural_scalar(input);
        assert_eq!(pos, spos);
        assert_eq!(types, stypes);
    }

    #[test]
    fn test_no_structural() {
        let input = b"hello world 12345";
        assert_eq!(count_structural(input), 0);
        assert_eq!(extract_structural(input), (vec![], vec![]));
    }

    #[test]
    fn test_exact_16_bytes() {
        let input = b"{\"a\":1,\"b\":2}   "; // 16 bytes
        let simd = count_structural(input);
        let scalar = count_structural_scalar(input);
        assert_eq!(simd, scalar);
    }

    #[test]
    fn test_17_bytes() {
        let input = b"{\"a\":1,\"b\":2}    "; // 17 bytes
        let (pos, types) = extract_structural(input);
        let (spos, stypes) = extract_structural_scalar(input);
        assert_eq!(pos, spos);
        assert_eq!(types, stypes);
    }
}
