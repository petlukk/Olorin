#!/usr/bin/env python3
"""Probe: what does the server report for the `echo` frame field at a normal
bash prompt vs during a `read -s` (no-echo) read? Decides whether gating local
echo on termios ECHO is correct, or whether readline disables ECHO at the prompt
(which would wrongly suppress local echo everywhere)."""
import socket, base64, hashlib, os, struct, sys, time, json

HOST, PORT = "127.0.0.1", 8741
GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def http_post(path, body=b""):
    s = socket.create_connection((HOST, PORT), timeout=10)
    s.sendall((f"POST {path} HTTP/1.1\r\nHost:{HOST}\r\nContent-Length:{len(body)}\r\n"
               f"Connection: close\r\n\r\n").encode() + body)
    data = b""
    while True:
        c = s.recv(4096)
        if not c: break
        data += c
    s.close()
    return data.split(b"\r\n\r\n", 1)[1].decode(errors="replace")


def mask_frame(payload):
    mask = os.urandom(4)
    masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
    return bytes([0x81, 0x80 | len(payload)]) + mask + masked


def recvn(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c: return buf
        buf += c
    return buf


def read_text_frame(s):
    hdr = recvn(s, 2)
    if len(hdr) < 2: return None
    n = hdr[1] & 0x7F
    if n == 126: n = struct.unpack(">H", recvn(s, 2))[0]
    elif n == 127: n = struct.unpack(">Q", recvn(s, 8))[0]
    return recvn(s, n).decode(errors="replace") if (hdr[0] & 0x0F) == 1 else ""


def collect(s, secs):
    """Collect parsed frame dicts for `secs` seconds."""
    out = []
    s.settimeout(secs)
    end = time.time() + secs
    try:
        while time.time() < end:
            txt = read_text_frame(s)
            if txt is None: break
            try: out.append(json.loads(txt))
            except Exception: pass
    except socket.timeout:
        pass
    return out


def echo_vals(frames):
    return [f.get("echo") for f in frames if f.get("type") == "frame" and "echo" in f]


def main():
    resp = http_post("/api/term/open")
    tid = int(resp.split('"id":')[1].split("}")[0])
    key = base64.b64encode(os.urandom(16)).decode()
    s = socket.create_connection((HOST, PORT), timeout=10)
    s.sendall((f"GET /api/term/{tid}/ws HTTP/1.1\r\nHost:{HOST}\r\nUpgrade: websocket\r\n"
               f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        head += s.recv(256)
    assert b"101" in head, head

    # Settle the initial prompt, then type a visible char so a frame is
    # produced with the *current* echo state (poll only emits on change).
    collect(s, 1.0)
    s.sendall(mask_frame(b"K"))
    pf = collect(s, 1.5)
    prompt = echo_vals(pf)
    k_shown = any("K" in "".join(c.get("ch", "") for c in f.get("cells", []))
                  for f in pf if f.get("type") == "frame")
    print(f"  echo at bash prompt: {prompt}  (K echoed to grid: {k_shown})")
    s.sendall(mask_frame(b"\x7f"))  # backspace — clean up the K
    collect(s, 0.5)

    # Now a no-echo read. `read -s` disables ECHO before printing the prompt.
    s.sendall(mask_frame(b"read -s -p ZZPROMPTZZ: v\n"))
    readframes = collect(s, 2.0)
    during = echo_vals(readframes)
    saw_prompt_false = any(
        f.get("echo") is False and "Z" in "".join(
            c.get("ch", "") for c in f.get("cells", []))
        for f in readframes if f.get("type") == "frame")
    print(f"  echo during read -s: {during}")
    print(f"  ZZPROMPTZZ frame had echo=false: {saw_prompt_false}")
    s.sendall(mask_frame(b"x\n"))  # release the read
    collect(s, 0.5)
    s.close()

    prompt_true = any(v is True for v in prompt)
    read_false = any(v is False for v in during)
    print(f"\n  VERDICT: prompt_echo_true={prompt_true} read_echo_false={read_false}")
    if prompt_true and read_false:
        print("  => Approach VALID: ECHO on at prompt, off during read -s.")
        return 0
    print("  => Approach INVALID — see values above; local-echo gate needs rethink.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
