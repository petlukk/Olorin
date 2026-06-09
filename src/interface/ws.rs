//! Minimal RFC 6455 WebSocket support — handshake, frame read/write, SHA-1,
//! base64. Used by the term tile so each keystroke isn't a fresh HTTP POST.

use std::io::{self, Read, Write};

/// Extract a header value (case-insensitive name) from a raw HTTP request.
pub fn header<'a>(req: &'a str, name: &str) -> Option<&'a str> {
    for line in req.lines().skip(1) {
        let colon = line.find(':')?;
        if line[..colon].eq_ignore_ascii_case(name) {
            return Some(line[colon + 1..].trim());
        }
    }
    None
}

/// Send the 101 Switching Protocols response. `req` is the raw HTTP upgrade
/// request; the Sec-WebSocket-Key is extracted from it.
pub fn handshake(stream: &mut impl Write, req: &str) -> io::Result<()> {
    let key = header(req, "Sec-WebSocket-Key")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing key"))?;
    let mut concat = String::with_capacity(key.len() + 36);
    concat.push_str(key);
    concat.push_str("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let accept = base64_encode(&sha1(concat.as_bytes()));
    write!(
        stream,
        "HTTP/1.1 101 Switching Protocols\r\n\
         Upgrade: websocket\r\n\
         Connection: Upgrade\r\n\
         Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )?;
    stream.flush()
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Opcode {
    Cont = 0,
    Text = 1,
    Binary = 2,
    Close = 8,
    Ping = 9,
    Pong = 10,
}

#[derive(Debug)]
pub struct Frame {
    pub opcode: Opcode,
    pub payload: Vec<u8>,
}

/// Read a single WebSocket frame from `stream`. Returns Ok(None) on clean
/// EOF (the peer half-closed without sending a Close frame).
pub fn read_frame(stream: &mut impl Read) -> io::Result<Option<Frame>> {
    let mut hdr = [0u8; 2];
    match stream.read_exact(&mut hdr) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let opcode = match hdr[0] & 0x0f {
        0 => Opcode::Cont,
        1 => Opcode::Text,
        2 => Opcode::Binary,
        8 => Opcode::Close,
        9 => Opcode::Ping,
        10 => Opcode::Pong,
        op => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("opcode {op}"))),
    };
    let masked = (hdr[1] & 0x80) != 0;
    let len7 = hdr[1] & 0x7f;
    let payload_len: usize = match len7 {
        126 => {
            let mut b = [0u8; 2];
            stream.read_exact(&mut b)?;
            u16::from_be_bytes(b) as usize
        }
        127 => {
            let mut b = [0u8; 8];
            stream.read_exact(&mut b)?;
            u64::from_be_bytes(b) as usize
        }
        n => n as usize,
    };
    // RFC 6455: client frames MUST be masked. Reject unmasked client frames.
    if !masked {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unmasked client frame"));
    }
    // Cap the declared length before allocating so a malicious/buggy peer
    // can't request a multi-exabyte Vec and abort the process. 16 MiB is far
    // above any real keystroke/paste yet still bounds the allocation.
    const MAX_PAYLOAD: usize = 16 << 20; // 16 MiB
    if payload_len > MAX_PAYLOAD {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "frame too large"));
    }
    let mut mask = [0u8; 4];
    stream.read_exact(&mut mask)?;
    let mut payload = vec![0u8; payload_len];
    stream.read_exact(&mut payload)?;
    for (i, b) in payload.iter_mut().enumerate() {
        *b ^= mask[i & 3];
    }
    Ok(Some(Frame { opcode, payload }))
}

/// Write a server-side (unmasked) text frame.
pub fn write_text(stream: &mut impl Write, text: &str) -> io::Result<()> {
    write_frame(stream, Opcode::Text, text.as_bytes())
}

/// Write a Close frame and flush.
pub fn write_close(stream: &mut impl Write) -> io::Result<()> {
    write_frame(stream, Opcode::Close, &[])?;
    stream.flush()
}

/// Write an empty Ping frame and flush. Browsers auto-reply with Pong and
/// never surface it to JS, so it's an invisible liveness probe — a failed
/// write reveals a half-open socket the reader's blocking read can't detect.
pub fn write_ping(stream: &mut impl Write) -> io::Result<()> {
    write_frame(stream, Opcode::Ping, &[])?;
    stream.flush()
}

fn write_frame(stream: &mut impl Write, opcode: Opcode, payload: &[u8]) -> io::Result<()> {
    let mut hdr = [0u8; 10];
    hdr[0] = 0x80 | opcode as u8; // FIN=1
    let hdr_len = match payload.len() {
        n if n <= 125 => {
            hdr[1] = n as u8;
            2
        }
        n if n <= 0xffff => {
            hdr[1] = 126;
            hdr[2..4].copy_from_slice(&(n as u16).to_be_bytes());
            4
        }
        n => {
            hdr[1] = 127;
            hdr[2..10].copy_from_slice(&(n as u64).to_be_bytes());
            10
        }
    };
    stream.write_all(&hdr[..hdr_len])?;
    stream.write_all(payload)
}

// ─── SHA-1 (FIPS 180-4) ──────────────────────────────────────────────────
// Only used for the WS accept-key digest. ~80 LOC keeps it out of a dep.

pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [
        0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut w = [0u32; 80];
    for chunk in padded.chunks_exact(64) {
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let temp = a.rotate_left(5)
                .wrapping_add(f).wrapping_add(e).wrapping_add(k).wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

// ─── Base64 (standard alphabet, no padding stripping) ────────────────────

pub fn base64_encode(data: &[u8]) -> String {
    const ALPHA: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = if chunk.len() > 1 { chunk[1] } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] } else { 0 };
        out.push(ALPHA[(b0 >> 2) as usize] as char);
        out.push(ALPHA[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHA[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHA[(b2 & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
