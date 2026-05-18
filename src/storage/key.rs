//! Vault key derivation and hashing helpers.
//!
//! Key derivation: Argon2id (RFC 9106) over passphrase + salt, using
//! the storage-layer wrapper in [`crate::storage::argon2id`].  The
//! salt is generated fresh at first vault create and persisted in
//! `<vault_dir>/vault.salt`; without the salt file the vault is
//! undecryptable (by design — losing the salt loses the vault).
//!
//! Histogram + xxHash64 helpers are unchanged from the v2.x layout.

use std::path::Path;

use crate::error::{Error, Result};
use crate::platform::random;
use crate::storage::argon2id::{argon2id, Params};

pub const SALT_BYTES: usize = 16;
const SALT_FILE: &str = "vault.salt";

/// Derive the 32-byte vault key from a passphrase, salt, and Argon2id
/// parameters.  Wraps `storage::argon2id::argon2id` with the
/// no-secret, no-AD profile that the vault uses (per RFC 9106's two
/// recommended profiles, Olorin's vault has neither auxiliary input).
pub fn derive_key(passphrase: &[u8], salt: &[u8; SALT_BYTES], params: Params) -> Result<[u8; 32]> {
    let mut out = [0u8; 32];
    argon2id(passphrase, salt, &[], &[], params, &mut out)?;
    Ok(out)
}

/// Read the per-vault salt from `<dir>/vault.salt`, or generate and
/// persist a fresh 16-byte salt if the file is absent.
///
/// Salt is treated as public-but-stable: it's the input that makes
/// every vault's Argon2id output unique, but it's not a secret in
/// itself.  Stored separately from `vault.bin` so a salt-only leak
/// reveals nothing (and a vault-only copy is unopenable without it).
pub fn load_or_create_salt(dir: &Path) -> Result<[u8; SALT_BYTES]> {
    let path = dir.join(SALT_FILE);
    if path.exists() {
        let bytes = std::fs::read(&path)
            .map_err(|_| Error::Vault("failed to read vault.salt"))?;
        if bytes.len() != SALT_BYTES {
            return Err(Error::Vault("vault.salt has wrong length"));
        }
        let mut salt = [0u8; SALT_BYTES];
        salt.copy_from_slice(&bytes);
        Ok(salt)
    } else {
        let mut salt = [0u8; SALT_BYTES];
        random::fill_bytes(&mut salt)?;
        std::fs::write(&path, salt)
            .map_err(|_| Error::Vault("failed to write vault.salt"))?;
        Ok(salt)
    }
}

pub fn compute_histogram(data: &[u8]) -> [u8; 256] {
    let mut hist = [0u8; 256];
    for &b in data {
        hist[b as usize] = hist[b as usize].saturating_add(1);
    }
    hist
}

pub fn normalize_histogram(hist: &[u8; 256]) -> [f32; 256] {
    let mut norm = [0.0f32; 256];
    let mut sum_sq = 0.0f32;
    for i in 0..256 {
        let v = hist[i] as f32;
        norm[i] = v;
        sum_sq += v * v;
    }
    let mag = sum_sq.sqrt();
    if mag > 0.0 {
        for n in &mut norm { *n /= mag; }
    }
    norm
}

pub fn cosine_similarity(a: &[f32; 256], b: &[f32; 256]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..256 { dot += a[i] * b[i]; }
    dot
}

pub fn xxhash64(data: &[u8], seed: u64) -> u64 {
    const P1: u64 = 0x9E3779B185EBCA87;
    const P2: u64 = 0xC2B2AE3D27D4EB4F;
    const P3: u64 = 0x165667B19E3779F9;
    const P4: u64 = 0x85EBCA77C2B2AE63;
    const P5: u64 = 0x27D4EB2F165667C5;

    let len = data.len();
    let mut h: u64;

    let round = |acc: u64, inp: u64| -> u64 {
        acc.wrapping_add(inp.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
    };
    let merge = |acc: u64, val: u64| -> u64 {
        (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
    };
    let av = |mut x: u64| -> u64 {
        x ^= x >> 33; x = x.wrapping_mul(P2);
        x ^= x >> 29; x = x.wrapping_mul(P3);
        x ^= x >> 32; x
    };
    let r64 = |s: &[u8]| u64::from_le_bytes(s[..8].try_into().unwrap());
    let r32 = |s: &[u8]| u32::from_le_bytes(s[..4].try_into().unwrap()) as u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        let mut i = 0;
        while i + 32 <= len {
            v1 = round(v1, r64(&data[i..]));
            v2 = round(v2, r64(&data[i+8..]));
            v3 = round(v3, r64(&data[i+16..]));
            v4 = round(v4, r64(&data[i+24..]));
            i += 32;
        }
        h = v1.rotate_left(1).wrapping_add(v2.rotate_left(7))
              .wrapping_add(v3.rotate_left(12)).wrapping_add(v4.rotate_left(18));
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed.wrapping_add(P5);
    }

    h = h.wrapping_add(len as u64);
    let rem = &data[len & !31..];
    let mut i = 0;
    while i + 8 <= rem.len() {
        h ^= round(0, r64(&rem[i..]));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }
    if i + 4 <= rem.len() {
        h ^= r32(&rem[i..]).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }
    while i < rem.len() {
        h ^= (rem[i] as u64).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
        i += 1;
    }
    av(h)
}
