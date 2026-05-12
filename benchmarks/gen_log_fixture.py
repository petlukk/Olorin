#!/usr/bin/env python3
"""Generate a deterministic synthetic log file for the eatime
benchmark. Each line starts with an ISO-8601 timestamp; total size
is roughly N MB.

Usage:
    python3 gen_log_fixture.py <size_mb> <out_path>
"""
import sys

def main() -> None:
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <size_mb> <out_path>", file=sys.stderr)
        sys.exit(2)
    size_mb = int(sys.argv[1])
    out_path = sys.argv[2]
    target = size_mb * 1024 * 1024

    written = 0
    line_idx = 0
    with open(out_path, "w") as f:
        while written < target:
            h = line_idx % 24
            m = (line_idx * 7) % 60
            s = (line_idx * 13) % 60
            day = ((line_idx // 1440) % 28) + 1
            line = (
                f"2026-05-{day:02d}T{h:02d}:{m:02d}:{s:02d} "
                f"INFO event=heartbeat shard={line_idx % 100:03d} "
                f"worker={line_idx % 10000:05d} ok=true\n"
            )
            if written + len(line) > target:
                break
            f.write(line)
            written += len(line)
            line_idx += 1
    print(f"wrote {written} bytes, {line_idx} lines -> {out_path}")

if __name__ == "__main__":
    main()
