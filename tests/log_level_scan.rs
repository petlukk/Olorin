use olorin::kernels::ffi;

const DELIMS: &[u8] = b" \t\n\r[]\":";

fn is_delim(b: u8) -> bool {
    DELIMS.contains(&b)
}

fn scalar_reference(text: &[u8]) -> [i32; 6] {
    let mut out = [0i32; 6];
    let keywords: [&[u8]; 5] = [b"DEBUG", b"INFO", b"WARN", b"ERROR", b"FATAL"];
    let len = text.len();
    for (p, &b) in text.iter().enumerate() {
        if b == b'\n' {
            out[5] += 1;
        }
        for (idx, kw) in keywords.iter().enumerate() {
            if p + kw.len() > len {
                continue;
            }
            if &text[p..p + kw.len()] != *kw {
                continue;
            }
            let left_ok = p == 0 || is_delim(text[p - 1]);
            let right_ok = p + kw.len() == len || is_delim(text[p + kw.len()]);
            if left_ok && right_ok {
                out[idx] += 1;
            }
        }
    }
    out
}

fn kernel_scan(text: &[u8]) -> [i32; 6] {
    ffi::init().unwrap();
    let mut out = [0i32; 6];
    let mut n_pos = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::log_level_scan(
            text.as_ptr(), text.len() as i32,
            out.as_mut_ptr(),
            std::ptr::null_mut(), 0, &mut n_pos,
            scratch.as_mut_ptr(),
        );
    }
    out
}

fn kernel_scan_with_positions(text: &[u8], max: usize) -> ([i32; 6], Vec<i32>) {
    ffi::init().unwrap();
    let mut out = [0i32; 6];
    let mut positions = vec![0i32; max];
    let mut n_pos = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::log_level_scan(
            text.as_ptr(), text.len() as i32,
            out.as_mut_ptr(),
            positions.as_mut_ptr(), max as i32, &mut n_pos,
            scratch.as_mut_ptr(),
        );
    }
    positions.truncate(n_pos as usize);
    (out, positions)
}

fn scalar_high_severity_positions(text: &[u8]) -> Vec<i32> {
    let mut out = Vec::new();
    let keywords: [&[u8]; 2] = [b"ERROR", b"FATAL"];
    let len = text.len();
    for p in 0..len {
        for kw in &keywords {
            if p + kw.len() > len {
                continue;
            }
            if &text[p..p + kw.len()] != *kw {
                continue;
            }
            let left_ok = p == 0 || is_delim(text[p - 1]);
            let right_ok = p + kw.len() == len || is_delim(text[p + kw.len()]);
            if left_ok && right_ok {
                out.push(p as i32);
                break;
            }
        }
    }
    out
}

fn assert_parity(name: &str, text: &[u8]) {
    let kernel = kernel_scan(text);
    let scalar = scalar_reference(text);
    assert_eq!(
        kernel, scalar,
        "{name}: kernel {kernel:?} != scalar {scalar:?} (len={})",
        text.len()
    );
}

#[test]
fn empty_buffer() {
    assert_parity("empty", b"");
}

#[test]
fn single_byte() {
    assert_parity("D", b"D");
    assert_parity("space", b" ");
}

#[test]
fn isolated_info_no_context() {
    let kernel = kernel_scan(b"INFO");
    assert_eq!(kernel, [0, 1, 0, 0, 0, 0], "INFO alone should count once");
}

#[test]
fn all_five_keywords_isolated() {
    assert_parity("five isolated", b"DEBUG INFO WARN ERROR FATAL");
}

#[test]
fn keyword_at_start() {
    assert_parity("INFO at start", b"INFO message here\n");
}

#[test]
fn keyword_at_end() {
    assert_parity("trailing ERROR", b"the level is ERROR");
}

#[test]
fn no_left_boundary() {
    let kernel = kernel_scan(b"xINFO");
    assert_eq!(kernel, [0, 0, 0, 0, 0, 0], "xINFO must NOT count");
}

#[test]
fn no_right_boundary() {
    let kernel = kernel_scan(b"INFOx");
    assert_eq!(kernel, [0, 0, 0, 0, 0, 0], "INFOx must NOT count");
}

#[test]
fn word_boundary_compound_identifiers() {
    assert_parity("ERROR_HANDLER", b"ERROR_HANDLER fired in INFO_TYPE");
    let kernel = kernel_scan(b"ERROR_HANDLER fired in INFO_TYPE");
    assert_eq!(
        kernel,
        [0, 0, 0, 0, 0, 0],
        "ERROR_HANDLER and INFO_TYPE are not boundary matches"
    );
}

#[test]
fn jsonl_quoted_levels() {
    assert_parity(
        "jsonl",
        b"{\"ts\":1,\"level\":\"INFO\",\"msg\":\"hi\"}\n{\"ts\":2,\"level\":\"ERROR\",\"msg\":\"bad\"}\n",
    );
}

#[test]
fn bracketed_levels() {
    assert_parity("bracketed", b"[INFO] starting\n[WARN] slow\n[ERROR] crash\n");
    let kernel = kernel_scan(b"[INFO] starting\n[WARN] slow\n[ERROR] crash\n");
    assert_eq!(kernel, [0, 1, 1, 1, 0, 3]);
}

#[test]
fn dense_repetition() {
    let text = b" INFO  INFO  INFO  INFO  INFO ";
    assert_parity("dense INFO", text);
    let kernel = kernel_scan(text);
    assert_eq!(kernel[1], 5);
}

#[test]
fn straddle_simd_chunk_boundary() {
    let mut buf = vec![b'x'; 80];
    for i in 0..buf.len() {
        buf[i] = b'x';
    }
    let prefix = b" ".repeat(13);
    let suffix = b" tail message goes here\n";
    let mut text = Vec::new();
    text.extend_from_slice(&prefix);
    text.extend_from_slice(b"ERROR");
    text.extend_from_slice(suffix);
    assert_parity("ERROR crosses byte 16", &text);
    assert_eq!(kernel_scan(&text)[3], 1, "ERROR at p=13 must be counted");
}

#[test]
fn just_below_simd_threshold() {
    let text = b" INFO  WARN  DEBUG xx";
    assert!(text.len() < 37);
    assert_parity("scalar-only path", text);
}

#[test]
fn just_above_simd_threshold() {
    let mut text = Vec::new();
    text.extend_from_slice(b"prefix bytes pad XXXX INFO WARN ERROR FATAL DEBUG\n");
    assert!(text.len() >= 37);
    assert_parity("simd path active", &text);
}

#[test]
fn long_realistic_log() {
    let mut buf = String::new();
    for line in 0..500 {
        let level = ["INFO", "INFO", "INFO", "WARN", "INFO", "DEBUG", "ERROR"][line % 7];
        buf.push_str(&format!("2026-05-11T12:34:{:02} [{level}] line {line}: handler took 12ms\n", line % 60));
    }
    let bytes = buf.into_bytes();
    assert_parity("500-line synthetic", &bytes);
    let counts = kernel_scan(&bytes);
    assert!(counts[1] > 200, "expected INFO majority, got {counts:?}");
    assert!(counts[3] > 0);
    assert!(counts[2] > 0);
}

#[test]
fn fuzz_random_buffers() {
    let seeds: &[u64] = &[1, 42, 1337, 9999, 2026];
    for &seed in seeds {
        let mut state = seed;
        let mut text = Vec::with_capacity(2048);
        let alphabet = b"DEBUGINFOWARNERRORFATALabc_ \n\t[]:\"xyz0123";
        for _ in 0..2048 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((state >> 33) as usize) % alphabet.len();
            text.push(alphabet[idx]);
        }
        assert_parity(&format!("fuzz seed={seed}"), &text);
    }
}

#[test]
fn fuzz_tiny_buffers() {
    let seeds: &[u64] = &[7, 11, 13];
    for &seed in seeds {
        let mut state = seed;
        let alphabet = b"INFOWARNERRORFATAL _";
        for len in 0..40 {
            let mut text = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = ((state >> 33) as usize) % alphabet.len();
                text.push(alphabet[idx]);
            }
            assert_parity(&format!("tiny fuzz len={len} seed={seed}"), &text);
        }
    }
}

#[test]
fn debug_at_start_with_trailing_delim() {
    assert_parity("DEBUG at start", b"DEBUG some context here");
    assert_eq!(kernel_scan(b"DEBUG some context here")[0], 1);
}

#[test]
fn ends_exactly_with_keyword() {
    assert_parity("FATAL at end no trailing newline", b"context FATAL");
    assert_eq!(kernel_scan(b"context FATAL")[4], 1);
}

// ── Position-recording tests ──────────────────────────────────────────────

#[test]
fn positions_empty_for_no_high_severity() {
    let (_, positions) = kernel_scan_with_positions(b" INFO INFO WARN DEBUG WARN INFO ", 16);
    assert!(positions.is_empty(), "INFO/WARN/DEBUG should not produce positions: {positions:?}");
}

#[test]
fn positions_collected_for_error_fatal() {
    let text = b" ERROR at start\n[FATAL] middle\nINFO trailing\n";
    let (_, positions) = kernel_scan_with_positions(text, 16);
    let expected = scalar_high_severity_positions(text);
    assert_eq!(positions, expected,
        "kernel positions {positions:?} != scalar {expected:?}");
    assert_eq!(positions.len(), 2);
}

#[test]
fn positions_saturate_at_max() {
    let mut text = Vec::new();
    for _ in 0..100 {
        text.extend_from_slice(b" ERROR ");
    }
    let (counts, positions) = kernel_scan_with_positions(&text, 7);
    assert_eq!(counts[3], 100, "all 100 ERROR should be counted");
    assert_eq!(positions.len(), 7, "but only 7 positions recorded");
    let scalar = scalar_high_severity_positions(&text);
    assert_eq!(&positions[..], &scalar[..7],
        "first 7 positions must match scalar order");
}

#[test]
fn positions_in_scan_order_across_simd_chunks() {
    let mut buf = b" pad text here ".repeat(5);
    buf.extend_from_slice(b" ERROR a ");
    buf.extend_from_slice(b" pad text padpad pad here ".repeat(5).as_slice());
    buf.extend_from_slice(b" FATAL b ");
    buf.extend_from_slice(b" tail ".as_slice());
    let (_, positions) = kernel_scan_with_positions(&buf, 16);
    let expected = scalar_high_severity_positions(&buf);
    assert_eq!(positions, expected,
        "positions across SIMD chunk boundaries must match scalar");
}

#[test]
fn positions_fuzz_against_scalar() {
    let seeds: &[u64] = &[1, 42, 1337];
    for &seed in seeds {
        let mut state = seed;
        let alphabet = b"DEBUGINFOWARNERRORFATAL_ \n[]:";
        for len in [128usize, 1024, 4096] {
            let mut text = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = ((state >> 33) as usize) % alphabet.len();
                text.push(alphabet[idx]);
            }
            let (_, positions) = kernel_scan_with_positions(&text, 1024);
            let expected = scalar_high_severity_positions(&text);
            assert_eq!(positions, expected,
                "fuzz seed={seed} len={len}: positions diverge");
        }
    }
}

#[test]
fn positions_max_zero_means_no_recording() {
    let text = b" ERROR ERROR ERROR ";
    let mut out = [0i32; 6];
    let mut n_pos = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::log_level_scan(
            text.as_ptr(), text.len() as i32,
            out.as_mut_ptr(),
            std::ptr::null_mut(), 0, &mut n_pos,
            scratch.as_mut_ptr(),
        );
    }
    assert_eq!(out[3], 3, "counts must still work with max_positions=0");
    assert_eq!(n_pos, 0);
}
