//! WebSocket primitive tests — SHA-1 (FIPS 180-4 vectors), RFC 6455 accept-key
//! derivation, base64, and frame read (masking + size bound).
//! Moved out of src/ per the zero-`#[cfg(test)]`-in-src rule.

use olorin::interface::ws::{base64_encode, read_frame, sha1, Opcode};
use std::io::Cursor;

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

#[test]
fn read_masked_text_frame_unmasks() {
    // FIN+Text, masked, len 5, mask [1,2,3,4], payload "hello" XOR mask.
    let mask = [1u8, 2, 3, 4];
    let payload = b"hello";
    let mut frame = vec![0x81u8, 0x80 | 5];
    frame.extend_from_slice(&mask);
    for (i, &b) in payload.iter().enumerate() {
        frame.push(b ^ mask[i & 3]);
    }
    let f = read_frame(&mut Cursor::new(frame)).unwrap().unwrap();
    assert_eq!(f.opcode, Opcode::Text);
    assert_eq!(f.payload, b"hello");
}

#[test]
fn read_frame_rejects_oversized_length() {
    // FIN+Binary, masked, 127-length prefix declaring 1 TiB (>> 16 MiB cap).
    // Must error before allocating, so only the 10-byte header is needed.
    let mut frame = vec![0x82u8, 0x80 | 127];
    frame.extend_from_slice(&(1u64 << 40).to_be_bytes());
    let err = read_frame(&mut Cursor::new(frame)).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}

#[test]
fn read_frame_rejects_unmasked_client_frame() {
    // FIN+Text, NOT masked, len 1 — RFC 6455 requires client frames be masked.
    let err = read_frame(&mut Cursor::new(vec![0x81u8, 0x01, b'x'])).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
}
