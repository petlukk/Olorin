//! Argon2id (RFC 9106) — memory-hard password-based KDF.
//!
//! Sits on top of `storage::blake2b` (for H₀ and the variable-output H′
//! construction in §3.3) and the `argon2_block_compress` kernel
//! (for the per-block G compression in §3.4).
//!
//! Single-binary, no parallelism: lanes (when p > 1) are processed
//! sequentially.  This sacrifices wall-clock against `cargo`-style
//! many-thread Argon2 implementations, but Olorin only invokes the KDF
//! once at vault open, and adding a worker pool just for this would
//! drag the binary dependency surface in the wrong direction.
//!
//! Memory ownership: the memory matrix is a plain `Vec<u64>` — for
//! default params (64 MiB), wrapping it in SecureBuffer (mlock'd) would
//! be heavyweight, and the output key already lands in SecureBuffer
//! via the caller in `key.rs`.  The Vec is dropped on return, releasing
//! the memory.

use crate::error::{Error, Result};
use crate::kernels::ffi;
use crate::storage::blake2b;

const BLOCK_U64: usize = 128;        // 1024-byte block = 128 × u64
const BLOCK_BYTES: usize = 1024;
const SYNC_POINTS: u32 = 4;          // slices per pass (RFC 9106 §3.4)
const ARGON2_TYPE_ID: u32 = 2;       // Argon2id
const ARGON2_VERSION: u32 = 0x13;    // RFC 9106 version

/// Argon2id parameters.  All sizes follow RFC 9106 nomenclature.
#[derive(Clone, Copy)]
pub struct Params {
    /// Memory in KiB (1 KiB = 1024 bytes = 1 Argon2 block).
    pub memory_kib: u32,
    /// Number of passes over the memory matrix.
    pub iterations: u32,
    /// Lanes (parallelism degree).  We process them sequentially regardless.
    pub parallelism: u32,
    /// Output key length in bytes.
    pub tag_length: u32,
}

impl Params {
    /// RFC 9106 §4.1 "second recommended" profile, downsized for an
    /// interactive single-user vault open: 64 MiB, 3 passes, 1 lane.
    /// ~100 ms on a recent Ryzen, ~300 ms on Pi 5 — felt as a slight
    /// pause, not a delay.
    pub const VAULT_DEFAULT: Self = Self {
        memory_kib: 65536,
        iterations: 3,
        parallelism: 1,
        tag_length: 32,
    };
}

fn validate(p: &Params, password_len: usize, salt_len: usize) -> Result<()> {
    if p.parallelism < 1 || p.parallelism > 0x00FF_FFFF {
        return Err(Error::Vault("argon2id: parallelism out of range"));
    }
    if p.iterations < 1 {
        return Err(Error::Vault("argon2id: iterations must be >= 1"));
    }
    if p.tag_length < 4 {
        return Err(Error::Vault("argon2id: tag_length must be >= 4"));
    }
    if salt_len < 8 {
        return Err(Error::Vault("argon2id: salt must be >= 8 bytes"));
    }
    if password_len > 0xFFFF_FFFF {
        return Err(Error::Vault("argon2id: password too large"));
    }
    let min_memory = 8 * p.parallelism;
    if p.memory_kib < min_memory {
        return Err(Error::Vault("argon2id: memory_kib < 8 * parallelism"));
    }
    Ok(())
}

/// Run Argon2id, writing exactly `out.len()` bytes of derived key.
///
/// `secret` and `ad` are optional (empty slice when unused); they are
/// included in the H₀ derivation per RFC 9106 §3.2 so the function is
/// fully spec-compliant — Olorin's normal callers leave both empty,
/// but the RFC 9106 §5.2 KAT exercises both fields.
pub fn argon2id(
    password: &[u8],
    salt: &[u8],
    secret: &[u8],
    ad: &[u8],
    params: Params,
    out: &mut [u8],
) -> Result<()> {
    validate(&params, password.len(), salt.len())?;
    if out.len() != params.tag_length as usize {
        return Err(Error::Vault("argon2id: out length mismatch"));
    }

    let p = params.parallelism as usize;

    // RFC 9106 §3.3.1: lane_length = floor(m_prime / p), and m_prime is
    // m rounded down to a multiple of 4*p so each segment has the same
    // number of blocks.
    let m_prime = (params.memory_kib / (SYNC_POINTS * params.parallelism))
        * (SYNC_POINTS * params.parallelism);
    let lane_length = (m_prime / params.parallelism) as usize;
    let segment_length = lane_length / SYNC_POINTS as usize;

    // H0 = Blake2b(LE32(p) || LE32(T) || LE32(m) || LE32(t) || LE32(v)
    //              || LE32(y) || LE32(|P|) || P || LE32(|S|) || S
    //              || LE32(|K|) || K || LE32(|X|) || X)
    let mut h0 = [0u8; 72];   // 64-byte hash + 8 bytes scratch for the H′ prefix below
    {
        let mut h = blake2b::Hasher::new(64);
        h.update(&params.parallelism.to_le_bytes());
        h.update(&params.tag_length.to_le_bytes());
        h.update(&params.memory_kib.to_le_bytes());
        h.update(&params.iterations.to_le_bytes());
        h.update(&ARGON2_VERSION.to_le_bytes());
        h.update(&ARGON2_TYPE_ID.to_le_bytes());
        h.update(&(password.len() as u32).to_le_bytes());
        h.update(password);
        h.update(&(salt.len() as u32).to_le_bytes());
        h.update(salt);
        h.update(&(secret.len() as u32).to_le_bytes());
        h.update(secret);
        h.update(&(ad.len() as u32).to_le_bytes());
        h.update(ad);
        let mut h0_out = [0u8; 64];
        h.finalize(&mut h0_out);
        h0[..64].copy_from_slice(&h0_out);
    }

    // Memory matrix: p × lane_length × BLOCK_U64.  Single Vec<u64>,
    // indexed via (lane * lane_length + block) * BLOCK_U64 + word.
    let total_blocks = p * lane_length;
    let mut memory: Vec<u64> = vec![0u64; total_blocks * BLOCK_U64];

    // Initial blocks B[i][0] and B[i][1] for each lane.
    let mut block_buf = [0u8; BLOCK_BYTES];
    for lane in 0..p {
        for j in 0..2u32 {
            // H'^1024(H0 || LE32(j) || LE32(lane))
            h0[64..68].copy_from_slice(&j.to_le_bytes());
            h0[68..72].copy_from_slice(&(lane as u32).to_le_bytes());
            h_prime(&h0, &mut block_buf);
            store_block(&mut memory, lane, j as usize, lane_length, &block_buf);
        }
    }

    // Filling: passes × slices × lanes × segment, with the segment fill
    // doing the heavy lifting.
    for pass in 0..params.iterations {
        for slice in 0..SYNC_POINTS {
            for lane in 0..p {
                fill_segment(
                    &mut memory,
                    pass,
                    slice,
                    lane,
                    p,
                    lane_length,
                    segment_length,
                    &params,
                );
            }
        }
    }

    // C = XOR of last block of every lane.
    let mut c = [0u64; BLOCK_U64];
    for lane in 0..p {
        let off = (lane * lane_length + lane_length - 1) * BLOCK_U64;
        for (k, w) in c.iter_mut().enumerate() {
            *w ^= memory[off + k];
        }
    }

    // Tag = H'^T(C)
    let mut c_bytes = [0u8; BLOCK_BYTES];
    for (i, w) in c.iter().enumerate() {
        c_bytes[i * 8..i * 8 + 8].copy_from_slice(&w.to_le_bytes());
    }
    h_prime(&c_bytes, out);
    Ok(())
}

// ── H' variable-output Blake2b (RFC 9106 §3.3) ────────────────────────────────

/// RFC 9106 §3.3 H′: variable-length output via chained Blake2b-512
/// when the target exceeds 64 bytes.  `input` is consumed by the first
/// hash; subsequent hashes chain on Blake2b-512 outputs.
fn h_prime(input: &[u8], out: &mut [u8]) {
    let t = out.len() as u32;
    if (out.len()) <= 64 {
        let mut h = blake2b::Hasher::new(out.len());
        h.update(&t.to_le_bytes());
        h.update(input);
        h.finalize(out);
        return;
    }

    // Long output: V[1] = Blake2b(LE32(T) || A, 64);
    // V[i] = Blake2b(V[i-1], 64) for i in 2..=r;
    // V[r+1] = Blake2b(V[r], T - 32*r).
    let mut v_prev = [0u8; 64];
    {
        let mut h = blake2b::Hasher::new(64);
        h.update(&t.to_le_bytes());
        h.update(input);
        h.finalize(&mut v_prev);
    }
    out[..32].copy_from_slice(&v_prev[..32]);

    let mut written = 32usize;
    while out.len() - written > 64 {
        let mut v_next = [0u8; 64];
        let mut h = blake2b::Hasher::new(64);
        h.update(&v_prev);
        h.finalize(&mut v_next);
        out[written..written + 32].copy_from_slice(&v_next[..32]);
        v_prev = v_next;
        written += 32;
    }

    // Final chunk: T - 32 * (number of 32-byte chunks so far) bytes.
    let final_len = out.len() - written;
    let mut h = blake2b::Hasher::new(final_len);
    h.update(&v_prev);
    h.finalize(&mut out[written..]);
}

// ── Segment filling + reference indexing ──────────────────────────────────────

fn fill_segment(
    memory: &mut [u64],
    pass: u32,
    slice: u32,
    lane: usize,
    parallelism: usize,
    lane_length: usize,
    segment_length: usize,
    params: &Params,
) {
    let first_pass_first_slice = pass == 0 && slice == 0;
    let data_independent_segment = pass == 0 && slice < SYNC_POINTS / 2;

    // For Argon2i / Argon2id-data-independent segments, the PRNG block
    // is a function of (pass, lane, slice, m, t, y, counter); each call
    // produces 128 (J1, J2) pairs.
    let mut pseudo_block = [0u64; BLOCK_U64];
    let mut pseudo_zero = [0u64; BLOCK_U64];
    let mut pseudo_input = [0u64; BLOCK_U64];
    let mut pseudo_scratch = [0u64; 16];
    let mut pseudo_counter: u64 = 1;

    if data_independent_segment {
        pseudo_input[0] = pass as u64;
        pseudo_input[1] = lane as u64;
        pseudo_input[2] = slice as u64;
        pseudo_input[3] = (parallelism as u64) * (lane_length as u64);
        pseudo_input[4] = params.iterations as u64;
        pseudo_input[5] = ARGON2_TYPE_ID as u64;
    }

    let starting_index = if first_pass_first_slice { 2 } else { 0 };
    for i in starting_index..segment_length {
        let current_index_in_lane = slice as usize * segment_length + i;
        let prev_index = if current_index_in_lane == 0 {
            lane_length - 1
        } else {
            current_index_in_lane - 1
        };

        // J1, J2 for reference selection.
        let (j1, j2);
        if data_independent_segment {
            let slot = i % BLOCK_U64;
            if slot == 0 {
                pseudo_input[6] = pseudo_counter;
                pseudo_counter += 1;
                // Argon2id PRNG: G(0, G(0, input)).
                let mut tmp = [0u64; BLOCK_U64];
                unsafe {
                    ffi::argon2_block_compress(
                        pseudo_zero.as_ptr(),
                        pseudo_input.as_ptr(),
                        tmp.as_mut_ptr(),
                        pseudo_scratch.as_mut_ptr(),
                    );
                    ffi::argon2_block_compress(
                        pseudo_zero.as_ptr(),
                        tmp.as_ptr(),
                        pseudo_block.as_mut_ptr(),
                        pseudo_scratch.as_mut_ptr(),
                    );
                }
                // Defensive: keep pseudo_zero zeroed in case a future
                // tweak ever uses it for output (it currently doesn't).
                pseudo_zero = [0u64; BLOCK_U64];
            }
            let w = pseudo_block[slot];
            j1 = (w & 0xFFFF_FFFF) as u32;
            j2 = (w >> 32) as u32;
        } else {
            let prev_off = (lane * lane_length + prev_index) * BLOCK_U64;
            let w = memory[prev_off];
            j1 = (w & 0xFFFF_FFFF) as u32;
            j2 = (w >> 32) as u32;
        }

        // (ref_lane, ref_index) from (J1, J2).
        let ref_lane = if first_pass_first_slice {
            lane
        } else {
            (j2 as usize) % parallelism
        };
        let ref_index = compute_ref_index(
            j1,
            ref_lane == lane,
            pass,
            slice,
            i,
            lane_length,
            segment_length,
        );

        // B[lane][current] = G(B[lane][prev], B[ref_lane][ref_index])
        // For pass > 0: XOR into existing block instead of overwriting.
        let prev_off = (lane * lane_length + prev_index) * BLOCK_U64;
        let ref_off = (ref_lane * lane_length + ref_index) * BLOCK_U64;
        let cur_off = (lane * lane_length + current_index_in_lane) * BLOCK_U64;
        let mut new_block = [0u64; BLOCK_U64];
        let mut compress_scratch = [0u64; 16];

        let prev_ptr = memory[prev_off..prev_off + BLOCK_U64].as_ptr();
        let ref_ptr = memory[ref_off..ref_off + BLOCK_U64].as_ptr();
        unsafe {
            ffi::argon2_block_compress(
                prev_ptr,
                ref_ptr,
                new_block.as_mut_ptr(),
                compress_scratch.as_mut_ptr(),
            );
        }
        if pass == 0 {
            memory[cur_off..cur_off + BLOCK_U64].copy_from_slice(&new_block);
        } else {
            for (k, w) in new_block.iter().enumerate() {
                memory[cur_off + k] ^= *w;
            }
        }
    }
}

fn compute_ref_index(
    j1: u32,
    same_lane: bool,
    pass: u32,
    slice: u32,
    position_in_segment: usize,
    lane_length: usize,
    segment_length: usize,
) -> usize {
    // |W|: number of candidate reference blocks (RFC 9106 §3.4.1.2).
    let reference_area_size = if pass == 0 {
        // First pass: only blocks completed so far in the lane.
        if slice == 0 {
            // Same segment, blocks before current.
            position_in_segment - 1
        } else if same_lane {
            slice as usize * segment_length + position_in_segment - 1
        } else if position_in_segment == 0 {
            slice as usize * segment_length - 1
        } else {
            slice as usize * segment_length
        }
    } else {
        // Subsequent passes: whole lane minus current segment's unprocessed tail.
        if same_lane {
            lane_length - segment_length + position_in_segment - 1
        } else if position_in_segment == 0 {
            lane_length - segment_length - 1
        } else {
            lane_length - segment_length
        }
    };

    // Non-uniform mapping: y = |W| - 1 - (|W| * (J1^2 / 2^32)) / 2^32.
    let j1 = j1 as u64;
    let x = (j1 * j1) >> 32;
    let y = (reference_area_size as u64 * x) >> 32;
    let relative_position = reference_area_size as u64 - 1 - y;

    // Absolute index in the lane.
    let start_position: usize = if pass == 0 {
        0
    } else if slice == SYNC_POINTS - 1 {
        0
    } else {
        (slice as usize + 1) * segment_length
    };
    (start_position + relative_position as usize) % lane_length
}

// ── Memory matrix helpers ─────────────────────────────────────────────────────

fn store_block(memory: &mut [u64], lane: usize, j: usize, lane_length: usize, src: &[u8]) {
    let off = (lane * lane_length + j) * BLOCK_U64;
    for (k, chunk) in src.chunks_exact(8).enumerate() {
        memory[off + k] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
}
