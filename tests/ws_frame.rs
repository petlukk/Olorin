//! WebSocket primitive tests — SHA-1 (FIPS 180-4 vectors), RFC 6455 accept-key
//! derivation, and base64. Moved out of src/ per the zero-`#[cfg(test)]`-in-src rule.

use olorin::interface::ws::{base64_encode, sha1};

#[test]
fn sha1_empty() {
    // FIPS 180-4 test vector
    assert_eq!(
        sha1(b""),
        [
            0xda, 0x39, 0xa3, 0xee, 0x5e, 0x6b, 0x4b, 0x0d, 0x32, 0x55, 0xbf, 0xef, 0x95, 0x60,
            0x18, 0x90, 0xaf, 0xd8, 0x07, 0x09,
        ]
    );
}

#[test]
fn sha1_abc() {
    assert_eq!(
        sha1(b"abc"),
        [
            0xa9, 0x99, 0x3e, 0x36, 0x47, 0x06, 0x81, 0x6a, 0xba, 0x3e, 0x25, 0x71, 0x78, 0x50,
            0xc2, 0x6c, 0x9c, 0xd0, 0xd8, 0x9d,
        ]
    );
}

#[test]
fn ws_accept_key() {
    // RFC 6455 §1.3 example
    let key = "dGhlIHNhbXBsZSBub25jZQ==";
    let mut s = String::from(key);
    s.push_str("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let accept = base64_encode(&sha1(s.as_bytes()));
    assert_eq!(accept, "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
}

#[test]
fn base64_basic() {
    assert_eq!(base64_encode(b""), "");
    assert_eq!(base64_encode(b"f"), "Zg==");
    assert_eq!(base64_encode(b"fo"), "Zm8=");
    assert_eq!(base64_encode(b"foo"), "Zm9v");
    assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
    assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
    assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
}
