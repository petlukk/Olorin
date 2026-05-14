//! Round-trip and version-rejection tests for VaultHeaderV2.
//! The v2 struct exists but is not yet wired into the on-disk format
//! (that lands in Task 9).

use olorin::storage::vault::{HEADER_SIZE_V2, VaultHeaderV2};

#[test]
fn header_v2_round_trip() {
    let h = VaultHeaderV2 {
        magic: *b"OLRN",
        version: 2,
        block_count: 1337,
        index_offset: 0xdead_beef,
        key_id: [0x42; 16],
        nonce_seed_8: [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88],
        header_rewrites: 42,
        header_tag: [0xAB; 16],
        reserved: [0u8; 2],
    };
    let bytes = h.to_bytes();
    assert_eq!(bytes.len(), HEADER_SIZE_V2);
    assert_eq!(HEADER_SIZE_V2, 64);
    let parsed = VaultHeaderV2::from_bytes(&bytes).expect("parse");
    assert_eq!(parsed.magic, h.magic);
    assert_eq!({ parsed.version }, { h.version });
    assert_eq!({ parsed.block_count }, { h.block_count });
    assert_eq!({ parsed.index_offset }, { h.index_offset });
    assert_eq!(parsed.key_id, h.key_id);
    assert_eq!(parsed.nonce_seed_8, h.nonce_seed_8);
    assert_eq!({ parsed.header_rewrites }, { h.header_rewrites });
    assert_eq!(parsed.header_tag, h.header_tag);
}

#[test]
fn header_v2_rejects_wrong_version() {
    let h = VaultHeaderV2 {
        magic: *b"OLRN",
        version: 1, // wrong
        block_count: 0,
        index_offset: 64,
        key_id: [0; 16],
        nonce_seed_8: [0; 8],
        header_rewrites: 0,
        header_tag: [0; 16],
        reserved: [0; 2],
    };
    let bytes = h.to_bytes();
    let result = VaultHeaderV2::from_bytes(&bytes);
    assert!(result.is_err());
}

#[test]
fn header_v2_rejects_bad_magic() {
    let h = VaultHeaderV2 {
        magic: *b"WRNG",
        version: 2,
        block_count: 0,
        index_offset: 64,
        key_id: [0; 16],
        nonce_seed_8: [0; 8],
        header_rewrites: 0,
        header_tag: [0; 16],
        reserved: [0; 2],
    };
    let bytes = h.to_bytes();
    let result = VaultHeaderV2::from_bytes(&bytes);
    assert!(result.is_err());
}
