//! RFC 8439 §A.5 ChaCha20-Poly1305 AEAD test vector — exact bit-match round-trip.

use olorin::kernels::ffi;
use olorin::storage::aead;

fn hex(s: &str) -> Vec<u8> {
    let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

#[test]
fn rfc8439_appendix_a5() {
    ffi::init().expect("kernel init");

    let key_bytes = hex("1c9240a5eb55d38af333888604f6b5f0473917c1402b80099dca5cbc207075c0");
    let nonce_bytes = hex("000000000102030405060708");
    let aad = hex("f33388860000000000004e91");
    let pt: &[u8] = b"Internet-Drafts are draft documents valid for a maximum of six months \
                     and may be updated, replaced, or obsoleted by other documents at any time. \
                     It is inappropriate to use Internet-Drafts as reference material or to cite \
                     them other than as /\xe2\x80\x9cwork in progress./\xe2\x80\x9d";
    let expected_ct = hex(
        "64a0861575861af460f062c79be643bd5e805cfd345cf389f108670ac76c8cb24c6cfc18755d43eea09e\
         e94e382d26b0bdb7b73c321b0100d4f03b7f355894cf332f830e710b97ce98c8a84abd0b948114ad176\
         e008d33bd60f982b1ff37c8559797a06ef4f0ef61c186324e2b3506383606907b6a7c02b0f9f6157b53\
         c867e4b9166c767b804d46a59b5216cde7a4e99040c5a40433225ee282a1b0a06c523eaf4534d7f83fa\
         1155b0047718cbc546a0d072b04b3564eea1b422273f548271a0bb2316053fa76991955ebd6315943\
         4ecebb4e466dae5a1073a6727627097a1049e617d91d361094fa68f0ff77987130305beaba2eda04df\
         997b714d6c6f2c29a6ad5cb4022b02709b",
    );
    let expected_tag = hex("eead9d67890cbb22392336fea1851f38");

    let key: &[u8; 32] = key_bytes.as_slice().try_into().unwrap();
    let nonce: &[u8; 12] = nonce_bytes.as_slice().try_into().unwrap();

    let mut buf = pt.to_vec();
    let mut tag = [0u8; 16];
    aead::seal(key, nonce, &aad, &mut buf, &mut tag);
    assert_eq!(buf, expected_ct, "ciphertext mismatch");
    assert_eq!(&tag[..], expected_tag.as_slice(), "tag mismatch");

    aead::open(key, nonce, &aad, &mut buf, &tag).expect("open must succeed on RFC vector");
    assert_eq!(buf, pt);
}
