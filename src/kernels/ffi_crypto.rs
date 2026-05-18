//! Crypto-primitive FFI wrappers — poly1305, Blake2b, Argon2.
//!
//! Split out of `ffi.rs` to keep that file under the 500-LOC cap; same
//! pattern as `ffi_data.rs`.  All wrappers are `pub use`d from `ffi.rs`
//! so existing `kernels::ffi::<name>` call sites continue to work.

use super::ffi::k;

/// Poly1305 MAC (RFC 8439 §2.5).
///
/// Computes a 16-byte authentication tag over `msg` using `key`.
///
/// # Safety
/// - `key` must be valid for 32 bytes.
/// - `msg` must be valid for `msg_len` bytes (0 is allowed).
/// - `tag_out` must be valid for 16 bytes (write).
pub unsafe fn poly1305_mac(
    key: *const u8, msg: *const u8, msg_len: i32, tag_out: *mut u8,
) {
    // 22-word scratch buffer: r[0..4], 5*r[0..4], row_k[0..3], h[0..3], tag[0..3].
    let mut scratch = [0i32; 22];
    (k().poly1305_mac_kern)(key, msg, msg_len, tag_out, scratch.as_mut_ptr());
}

/// Constant-time Poly1305 tag verification.
///
/// Returns 1 iff the tag computed over `msg` with `key` exactly equals
/// the 16 bytes at `tag`.  No branches on key/msg/tag bytes; the compare
/// is OR-reduce + branchless `((acc - 1) >> 31) & 1`.
///
/// # Safety
/// - `key` must be valid for 32 bytes.
/// - `msg` must be valid for `msg_len` bytes (0 is allowed).
/// - `tag` must be valid for 16 bytes (read).
pub unsafe fn poly1305_verify(
    key: *const u8, msg: *const u8, msg_len: i32, tag: *const u8,
) -> i32 {
    let mut scratch = [0i32; 22];
    (k().poly1305_verify_kern)(key, msg, msg_len, tag, scratch.as_mut_ptr())
}

/// Blake2b compression function (RFC 7693 §3.2).
///
/// Walks one 128-byte message block through the Blake2b mixing rounds,
/// updating the 8-u64 chaining state in place.  The variable-output
/// hash builder in `storage::blake2b` calls this once per block and
/// finalises by passing `is_final = 1` on the last block.
///
/// # Safety
/// - `state` must be valid for 8 u64s (read+write).
/// - `block` must be valid for 16 u64s (read).
/// - `constants` must be valid for 9 u64s (read): the eight Blake2b
///   IVs followed by the all-ones finalization mask.
/// - `v_scratch` must be valid for 16 u64s (write); contents on entry
///   are ignored.
pub unsafe fn blake2b_compress(
    state: *mut u64,
    block: *const u64,
    constants: *const u64,
    t_low: u64,
    t_high: u64,
    is_final: i32,
    v_scratch: *mut u64,
) {
    (k().blake2b_compress_kern)(state, block, constants, t_low, t_high, is_final, v_scratch);
}

/// Argon2 G compression (RFC 9106 §3.4).
///
/// Combines two 1024-byte blocks `x` and `y` into one output block `z`
/// using row + column applications of the Argon2 P transform.
///
/// # Safety
/// - `x`, `y`, `z` must each be valid for 128 u64s (1024 bytes).
/// - `scratch` must be valid for 16 u64s (column gather/scatter buffer).
/// - All pointers must be 8-byte aligned.
pub unsafe fn argon2_block_compress(
    x: *const u64, y: *const u64, z: *mut u64, scratch: *mut u64,
) {
    (k().argon2_block_kern)(x, y, z, scratch);
}
