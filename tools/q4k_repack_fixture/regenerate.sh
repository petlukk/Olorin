#!/usr/bin/env bash
# Regenerate tests/fixtures/q4k_repack/{input.bin,golden.bin}.
#
# Step 1: extract the first 16 rows of blk.0.attn_output.weight raw Q4K bytes
#         from ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf via the
#         #[ignore]'d Rust helper test.
# Step 2: build gen_golden.c and run it on input.bin to produce golden.bin.
#
# Run from anywhere; uses CARGO_MANIFEST_DIR-relative paths.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
FIX_DIR="$REPO/tests/fixtures/q4k_repack"
TOOL_DIR="$REPO/tools/q4k_repack_fixture"

NROWS=16
NCOLS=1536

mkdir -p "$FIX_DIR"

echo "[1/2] Extracting input.bin via cargo test (--ignored)..."
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:/root/dev/eacompute/target/release:$PATH" \
    cargo test --release --test gemma4_batch_verify -- \
    --ignored --nocapture extract_q4k_input_fixture

if [[ ! -s "$FIX_DIR/input.bin" ]]; then
    echo "ERROR: input.bin not produced" >&2
    exit 1
fi

echo "[2/2] Building gen_golden and producing golden.bin..."
cc -O2 -std=c11 -Wall -o "$TOOL_DIR/gen_golden" "$TOOL_DIR/gen_golden.c"
"$TOOL_DIR/gen_golden" "$NROWS" "$NCOLS" "$FIX_DIR/input.bin" "$FIX_DIR/golden.bin"

ls -l "$FIX_DIR"
echo "OK — fixtures regenerated."
