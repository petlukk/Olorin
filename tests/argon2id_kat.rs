//! Argon2id known-answer test against RFC 9106 §5.2.
//!
//! The RFC's test vector exercises every non-trivial knob:
//!   - parallelism p > 1 (lane indexing path)
//!   - secret K (extra input to H₀)
//!   - associated data X (extra input to H₀)
//!   - iterations t > 1 (the XOR-into-existing-block path on pass 1+)
//!   - all four slices (the sync-point indexing)
//!
//! If this vector matches, every internal path of the algorithm is
//! correct: H₀ construction, H′ chained variable-output, the Argon2id
//! mode switch (Argon2i for the first half of slice 0 of pass 0, then
//! Argon2d everywhere else), reference-block indexing in both
//! data-independent and data-dependent forms, and the per-block G
//! compression that lives in the Eä kernel.

use olorin::kernels::ffi;
use olorin::storage::argon2id::{argon2id, Params};

fn setup() {
    ffi::init().expect("kernel init");
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn argon2id_rfc9106_section_5_2_vector() {
    setup();

    let password = vec![0x01u8; 32];
    let salt = vec![0x02u8; 16];
    let secret = vec![0x03u8; 8];
    let ad = vec![0x04u8; 12];
    let params = Params {
        memory_kib: 32,
        iterations: 3,
        parallelism: 4,
        tag_length: 32,
    };
    let expected = hex_to_bytes(
        "0d640df58d78766c08c037a34a8b53c9d01ef0452d75b65eb52520e96b01e659",
    );

    let mut out = [0u8; 32];
    argon2id(&password, &salt, &secret, &ad, params, &mut out).unwrap();
    assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn vault_default_params_are_deterministic() {
    // The VAULT_DEFAULT params (64 MiB, t=3, p=1) are what derive_key
    // will use in production.  No public KAT for this exact triple, so
    // we just pin same-input → same-output behaviour as a regression
    // guard: a change in the kernel, indexing, or H′ that breaks
    // determinism would surface here even without a known reference.
    setup();
    let password = b"correct horse battery staple";
    let salt = b"olorin-test-salt";

    let mut out_a = [0u8; 32];
    let mut out_b = [0u8; 32];
    argon2id(password, salt, &[], &[], Params::VAULT_DEFAULT, &mut out_a).unwrap();
    argon2id(password, salt, &[], &[], Params::VAULT_DEFAULT, &mut out_b).unwrap();
    assert_eq!(out_a, out_b, "Argon2id must be deterministic");

    // Different passphrase → different key.
    let mut out_diff = [0u8; 32];
    argon2id(b"wrong horse", salt, &[], &[], Params::VAULT_DEFAULT, &mut out_diff).unwrap();
    assert_ne!(out_a, out_diff, "different passphrase must produce different key");
}

#[test]
fn rejects_short_salt() {
    setup();
    let mut out = [0u8; 32];
    let err = argon2id(b"pw", &[0u8; 7], &[], &[], Params::VAULT_DEFAULT, &mut out);
    assert!(err.is_err(), "salt < 8 bytes must be rejected");
}

#[test]
fn rejects_zero_iterations() {
    setup();
    let mut out = [0u8; 32];
    let params = Params {
        iterations: 0,
        ..Params::VAULT_DEFAULT
    };
    let err = argon2id(b"pw", &[0u8; 16], &[], &[], params, &mut out);
    assert!(err.is_err(), "iterations == 0 must be rejected");
}

#[test]
fn output_length_mismatch_is_an_error() {
    setup();
    let mut out_too_short = [0u8; 16];
    let err = argon2id(
        b"pw",
        &[0u8; 16],
        &[],
        &[],
        Params::VAULT_DEFAULT, // tag_length=32 but out is 16
        &mut out_too_short,
    );
    assert!(err.is_err(), "out length != tag_length must be rejected");
}
