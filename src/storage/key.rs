//! Vault key derivation and hashing helpers.
//!
//! Key: XOR-obfuscated seed ^ hardware_id (from `platform::hwid`).
//! Histogram: byte-frequency index for encrypted search.
//! xxHash64: fast non-crypto hash for integrity checks.

/// Compile-time mask — obfuscates the seed in .rodata.
const COMPILE_MASK: [u8; 32] = [
    0xA7, 0x3B, 0xC9, 0x14, 0xE6, 0x58, 0xF2, 0x0D,
    0x8B, 0x61, 0xD4, 0x37, 0x9E, 0xAC, 0x55, 0x73,
    0xC2, 0x1F, 0xB8, 0x46, 0x7A, 0xE3, 0x09, 0xD1,
    0x5C, 0x84, 0xF7, 0x2E, 0x63, 0xA0, 0x4B, 0x19,
];

/// seed ^ COMPILE_MASK — raw seed never appears in .rodata.
const OBFUSCATED_SEED: [u8; 32] = {
    let seed = b"olorin-vault-seed-v0.5-default!!";
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = seed[i] ^ COMPILE_MASK[i];
        i += 1;
    }
    out
};

/// Read hardware identifier for key binding.
fn hardware_id() -> [u8; 32] {
    let raw = crate::platform::hwid::machine_id()
        .unwrap_or_else(|| "olorin-fallback-id".to_string());
    let mut id = [0u8; 32];
    let bytes = raw.trim().as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        id[i % 32] ^= b;
    }
    for i in 1..32 {
        id[i] ^= id[i - 1].wrapping_mul(31);
    }
    id
}

/// Derive the vault key: `OBFUSCATED_SEED ^ COMPILE_MASK ^ hardware_id()` = `seed ^ hw`.
pub fn derive_key() -> [u8; 32] {
    let hw = hardware_id();
    let mut key = [0u8; 32];
    for i in 0..32 {
        key[i] = OBFUSCATED_SEED[i] ^ COMPILE_MASK[i] ^ hw[i];
    }
    key
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
