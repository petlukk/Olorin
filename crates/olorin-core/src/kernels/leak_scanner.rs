use super::ffi;

/// Scan for leak pattern two-byte prefixes using SIMD.
/// Returns bitmasks — one i32 per 16-byte block.
/// Bit N is set if bytes N and N+1 match a known secret pattern prefix pair.
pub fn scan_prefixes(text: &[u8]) -> Vec<i32> {
    let len = text.len();
    if len == 0 {
        return vec![];
    }
    let max_blocks = (len + 15) / 16;
    let mut masks = vec![0i32; max_blocks];
    let mut n_blocks: i32 = 0;
    unsafe {
        ffi::scan_leak_prefixes(
            text.as_ptr(),
            len as i32,
            masks.as_mut_ptr(),
            &mut n_blocks,
        );
    }
    masks.truncate(n_blocks as usize);
    masks
}

/// Iterate candidate positions from SIMD bitmasks.
/// Calls `f(position)` for each set bit.
pub fn for_each_candidate(masks: &[i32], mut f: impl FnMut(usize)) {
    for (block_idx, &mask) in masks.iter().enumerate() {
        let mut m = mask as u32;
        let base = block_idx * 16;
        while m != 0 {
            let bit = m.trailing_zeros() as usize;
            f(base + bit);
            m &= m - 1;
        }
    }
}

/// Two-byte pairs for the leak filter.
const LEAK_PAIRS: &[(u8, u8)] = &[
    (b's', b'k'), (b'A', b'K'), (b'g', b'h'), (b'g', b'i'),
    (b'x', b'o'), (b'S', b'G'), (b's', b'e'), (b'-', b'-'),
    (b'A', b'I'), (b'B', b'e'),
];

/// Scalar reference: return positions of two-byte prefix matches.
pub fn scan_prefixes_scalar(text: &[u8]) -> Vec<usize> {
    let mut positions = Vec::new();
    if text.len() < 2 {
        return positions;
    }
    for i in 0..text.len() - 1 {
        let c = text[i];
        let n = text[i + 1];
        for &(first, second) in LEAK_PAIRS {
            if c == first && n == second {
                positions.push(i);
                break;
            }
        }
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(scan_prefixes(b""), Vec::<i32>::new());
    }

    #[test]
    fn test_no_matches() {
        let input = b"hello world 1234567890123456";
        let masks = scan_prefixes(input);
        let mut found = vec![];
        for_each_candidate(&masks, |pos| found.push(pos));
        let scalar = scan_prefixes_scalar(input);
        assert_eq!(found, scalar);
    }

    #[test]
    fn test_with_sk_prefix() {
        let input = b"my key is sk-1234567890abcdef";
        let masks = scan_prefixes(input);
        let mut found = vec![];
        for_each_candidate(&masks, |pos| found.push(pos));
        // "sk" at position 10
        assert!(found.contains(&10), "expected position 10 in {:?}", found);
    }

    #[test]
    fn test_aws_key() {
        let input = b"key: AKIA1234567890ABCDEF";
        let masks = scan_prefixes(input);
        let mut found = vec![];
        for_each_candidate(&masks, |pos| found.push(pos));
        // "AK" at position 5
        assert!(found.contains(&5), "expected position 5 in {:?}", found);
    }

    #[test]
    fn test_pem_prefix() {
        let input = b"-----BEGIN RSA PRIVATE KEY-----";
        let masks = scan_prefixes(input);
        let mut found = vec![];
        for_each_candidate(&masks, |pos| found.push(pos));
        // "--" at positions 0,1,2,3 and 25,26,27,28
        assert!(found.contains(&0), "expected position 0 in {:?}", found);
    }

    #[test]
    fn test_clean_english_low_candidates() {
        let input = b"The quick brown fox jumps over the lazy dog.";
        let masks = scan_prefixes(input);
        let mut count = 0u32;
        for_each_candidate(&masks, |_| count += 1);
        // Two-byte filter: almost no matches in clean English
        assert!(count < 3, "too many candidates: {count}");
    }
}
