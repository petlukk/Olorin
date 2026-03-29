use olorin::kernels::ffi;
use olorin::storage::search::FusedSearcher;

#[test]
fn test_fused_search_roundtrip() {
    ffi::init().unwrap();
    let key = [0x42u8; 32];
    let nonce = [0x01u8; 12];
    let plaintext = b"The quick brown fox jumps over the lazy dog";
    let mut ct = plaintext.to_vec();
    olorin::storage::crypto::encrypt(&key, &nonce, 0, &mut ct);

    let mut searcher = FusedSearcher::new();
    let result = searcher.search(&key, &nonce, &ct, &[b"fox"]);
    assert!(result.match_count > 0);
}

#[test]
fn test_fused_search_no_match() {
    ffi::init().unwrap();
    let key = [0x42u8; 32];
    let nonce = [0x01u8; 12];
    let plaintext = b"nothing interesting here";
    let mut ct = plaintext.to_vec();
    olorin::storage::crypto::encrypt(&key, &nonce, 0, &mut ct);

    let mut searcher = FusedSearcher::new();
    let result = searcher.search(&key, &nonce, &ct, &[b"unicorn"]);
    assert_eq!(result.match_count, 0);
}
