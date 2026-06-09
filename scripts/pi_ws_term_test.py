#!/usr/bin/env python3
"""Headless end-to-end test for the web-term WebSocket path on the Pi.

Opens a PTY term, upgrades to WebSocket, verifies the RFC 6455 accept key
(proves ws.rs SHA-1/base64 run correctly on NEON), sends a shell command as a
masked client frame, and confirms the echoed characters come back through the
grid-cell frames. Exits 0 on success.
"""
import socket, base64, hashlib, os, struct, sys, time

HOST, PORT = "127.0.0.1", 8741
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MARKER = "WSOK_NEON"


def http_post(path, body=b""):
    s = socket.create_connection((HOST, PORT), timeout=10)
    req = (f"POST {path} HTTP/1.1\r\nHost: {HOST}\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    s.sendall(req)
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.split(b"\r\n\r\n", 1)[1].decode(errors="replace")


def mask_frame(payload: bytes) -> bytes:
    # Client text frame, FIN=1 opcode=1, masked (RFC 6455 requires client mask).
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    hdr = bytes([0x81])
    n = len(payload)
    if n <= 125:
        hdr += bytes([0x80 | n])
    elif n <= 0xFFFF:
        hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
    return hdr + mask + masked


def read_server_frame(s):
    # Server frames are unmasked. Returns (opcode, payload) or (None, None).
    hdr = recvn(s, 2)
    if not hdr:
        return None, None
    opcode = hdr[0] & 0x0F
    n = hdr[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", recvn(s, 2))[0]
    elif n == 127:
        n = struct.unpack(">Q", recvn(s, 8))[0]
    return opcode, recvn(s, n)


def recvn(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


def main():
    # 1. Open a PTY session.
    resp = http_post("/api/term/open")
    if '"id"' not in resp:
        print(f"FAIL: term open returned: {resp!r}")
        return 1
    term_id = int(resp.split('"id":')[1].split("}")[0])
    print(f"  term opened: id={term_id}")

    # 2. WebSocket upgrade with a real browser-style key; verify accept.
    key = base64.b64encode(os.urandom(16)).decode()
    expected = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    s = socket.create_connection((HOST, PORT), timeout=10)
    s.sendall((f"GET /api/term/{term_id}/ws HTTP/1.1\r\nHost: {HOST}\r\n"
               "Upgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(256)
    head_txt = head.decode(errors="replace")
    if "101 Switching Protocols" not in head_txt:
        print(f"FAIL: no 101 upgrade:\n{head_txt}")
        return 1
    if f"Sec-WebSocket-Accept: {expected}" not in head_txt:
        print(f"FAIL: accept-key mismatch. expected {expected}\n{head_txt}")
        return 1
    print(f"  101 upgrade OK; accept-key verified ({expected})")

    # 3. Send a shell command as a masked client frame.
    s.sendall(mask_frame(f"echo {MARKER}\n".encode()))

    # 4. Collect grid-cell chars from server frames for ~4s; look for the marker.
    s.settimeout(4.0)
    seen = []
    deadline = time.time() + 4.0
    try:
        while time.time() < deadline:
            opcode, payload = read_server_frame(s)
            if opcode is None:
                break
            if opcode == 1:  # text frame = grid JSON
                txt = payload.decode(errors="replace")
                # cheap scan: pull every "ch":"X" single-char cell
                for part in txt.split('"ch":"')[1:]:
                    c = part[0]
                    if c != '\\':
                        seen.append(c)
    except socket.timeout:
        pass
    s.close()

    grid_text = "".join(seen)
    if MARKER in grid_text:
        print(f"  PASS: marker '{MARKER}' echoed back through grid frames")
        return 0
    print(f"FAIL: marker not found. grid chars seen: {grid_text!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
