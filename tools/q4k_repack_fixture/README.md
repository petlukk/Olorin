# q4k_repack_fixture

One-shot tool that produces the byte-exact `ggml` reference fixture used by
`tests/gemma4_batch_verify.rs::batch1_repack_q4k_bytes_match_ggml_golden`.

## What it does

1. Runs the `#[ignore]`-marked Rust helper
   `extract_q4k_input_fixture` which dumps the first 16 rows of
   `blk.0.attn_output.weight` from
   `~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf` to
   `tests/fixtures/q4k_repack/input.bin` (18,432 bytes — 16 rows × 8 column-blocks × 144 bytes).
2. Compiles `gen_golden.c` (a verbatim transcription of ggml's
   `make_block_q4_Kx8` + `repack_q4_K_to_q4_K_8_bl` outer loop, copied from
   `llama.cpp` build 8685 `ggml/src/ggml-cpu/repack.cpp`) and runs it on
   `input.bin` to produce `tests/fixtures/q4k_repack/golden.bin` (same size).

The fixture is generated **off the cargo build path** — `cargo build` does
not invoke any C compiler or touch ggml. The generator is checked in for
reproducibility only.

## Regenerating

```bash
tools/q4k_repack_fixture/regenerate.sh
```

Requires:
- `cc` (any C11 compiler)
- The gemma E2B GGUF in `~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf`
- The Eä compiler in PATH (the helper test loads the model via olorin)

## When to regenerate

- ggml ever changes its `block_q4_K` / `block_q4_Kx8` byte layout
  (verify against `ggml/src/ggml-common.h` and `ggml-cpu/repack.h`).
- ggml ever changes the body of `make_block_q4_Kx8` for the
  `blck_size_interleave == 8` path. Bump `gen_golden.c` to match and
  re-run.
- A different gemma quant or layer is needed for coverage.

If you regenerate, also update the byte layout note at
`docs/superpowers/research/2026-04-08-ggml-q4k-8x8-format.md`.

## Why a static fixture instead of linking ggml at test time

`repack_q4_K_to_q4_K_8_bl` is `static` inside `libggml-cpu.so` (no exported
symbol), so a runtime dlopen is impossible without modifying ggml. Compiling
ggml's full source tree into our test build would pull in C++ + the entire
ggml header chain, contradicting olorin's "true zero deps" rule. The static
fixture preserves byte-exact ggml semantics with zero build-system impact.
