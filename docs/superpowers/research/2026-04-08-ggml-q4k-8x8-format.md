# ggml Q4K_8x8 repack format

Source: llama.cpp build 8685, `ggml/src/ggml-cpu/repack.h` and `ggml/src/ggml-cpu/repack.cpp`.

This note is the byte-level reference for the Eä `q4k_repack_8x8` kernel (Task 5).
The Eä kernel must produce output that is **byte-for-byte identical** to what
`repack_q4_K_to_q4_K_8_bl(t, 8, ...)` writes.

## 1. Canonical `block_q4_K` (source, per-row)

From `ggml/src/ggml-common.h`, struct `block_q4_K` (at the time of writing, lines 317-328):

```c
#define QK_K          256
#define K_SCALE_SIZE  12

typedef struct {
    union {
        struct { ggml_half d; ggml_half dmin; };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];   // 12 bytes, 6-bit packed scales + mins
    uint8_t qs[QK_K/2];             // 128 bytes, 4-bit quants (low nibble,high nibble)
} block_q4_K;

static_assert(sizeof(block_q4_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2,
              "wrong q4_K block size/padding");
```

`ggml_half` is IEEE-754 f16 (2 bytes).

Byte layout of one `block_q4_K` (offset → field):

| offset | size | field          |
|--------|------|----------------|
| 0      | 2    | `d`    (f16)   |
| 2      | 2    | `dmin` (f16)   |
| 4      | 12   | `scales[12]`   |
| 16     | 128  | `qs[128]`      |

**Total: 144 bytes per Q4K block.** 256 weights per block → 4.5 bits/weight.

`d` and `dmin` are stored as **f16** (not f32) both in the source and the repacked layout.

## 2. `block_q4_Kx8` (destination, 8 rows interleaved)

From `ggml/src/ggml-cpu/repack.h`, struct `block_q4_Kx8` (at the time of writing, lines 43-50):

```c
struct block_q4_Kx8 {
    ggml_half d[8];      // 16 bytes
    ggml_half dmin[8];   // 16 bytes
    uint8_t   scales[96];// 96 bytes  (12 bytes * 8 rows, repacked)
    uint8_t   qs[1024];  // 1024 bytes (128 bytes * 8 rows, interleaved)
};

static_assert(sizeof(block_q4_Kx8) ==
              sizeof(ggml_half)*16 + K_SCALE_SIZE*8 + QK_K*4,
              "wrong q4_K block size/padding");
```

Byte layout of one `block_q4_Kx8`:

| offset | size | field          |
|--------|------|----------------|
|    0   |  16  | `d[0..8]`    (8 × f16) |
|   16   |  16  | `dmin[0..8]` (8 × f16) |
|   32   |  96  | `scales[96]` |
|  128   | 1024 | `qs[1024]`   |

**Total: 1152 bytes per repacked block.** This is the number that drives buffer
sizing for the Eä kernel.

Sanity check: `16 + 16 + 96 + 1024 = 1152 = 8 * 144`, i.e. no padding — pure
rearrangement of 8 source blocks.

## 3. Outer repack loop

From `ggml/src/ggml-cpu/repack.cpp`, function `repack_q4_K_to_q4_K_8_bl` (at the time of writing, lines 3231-3260):

```c
constexpr int nrows_interleaved = 8;
block_q4_Kx8  * dst = (block_q4_Kx8*)t->data;
const block_q4_K * src = (const block_q4_K*) data;
block_q4_K dst_tmp[8];
int nrow    = ggml_nrows(t);
int nblocks = t->ne[0] / QK_K;          // column-blocks per row

for (int b = 0; b < nrow; b += 8) {     // 8 rows per stripe
    for (int64_t x = 0; x < nblocks; x++) {
        for (int i = 0; i < 8; i++) {
            dst_tmp[i] = src[x + i * nblocks];   // gather one column-block from 8 rows
        }
        *dst++ = make_block_q4_Kx8(dst_tmp, 8);   // interleave_block = 8
    }
    src += 8 * nblocks;
}
```

Key facts:

- Rows are processed in stripes of 8. The tensor's row count must be divisible by 8.
- Within a stripe, the **column-block** `x` is the outer loop, so the 8 rows of
  column-block 0 are packed first, then all 8 rows of column-block 1, etc.
- Source index `src[x + i*nblocks]` = row `b+i`, column-block `x`.
- Destination is written sequentially as `*dst++`.
- `interleave_block` is always 8 for the `_8_bl` variant.

## 4. `make_block_q4_Kx8` (inner repack)

From `ggml/src/ggml-cpu/repack.cpp`, function `make_block_q4_Kx8` (at the time of writing, lines 2836-2911).

### 4a. `d` and `dmin`

Straight copy, in row order:

```c
for (i = 0; i < 8; i++) out.d[i]    = in[i].d;      // 16 bytes at offset 0
for (i = 0; i < 8; i++) out.dmin[i] = in[i].dmin;   // 16 bytes at offset 16
```

Both remain **f16**.

### 4b. `qs` interleaving (8-byte granularity)

```c
const int end = QK_K * 4 / blck_size_interleave;    // 256*4/8 = 128 iterations

for (int i = 0; i < end; ++i) {
    int src_id     = i % 8;                          // which of the 8 source rows
    int src_offset = (i / 8) * 8;                    // byte offset within that row's qs[128]
    int dst_offset = i * 8;                          // byte offset within out.qs[1024]

    uint64_t elems;
    memcpy(&elems, &in[src_id].qs[src_offset], 8);
    memcpy(&out.qs[dst_offset], &elems, 8);
}
```

So `out.qs` is a sequence of 128 eight-byte chunks. Chunk `i` (0..127) is:

- from source row `r = i % 8`
- from byte offset `(i / 8) * 8` within that row's `qs[128]`

Equivalently: write 16 groups (`g = i/8 = 0..15`) of 8 rows, where each group
copies bytes `[g*8 .. g*8+8)` from each of the 8 source rows in row-major order:

```
dst[   0.. 7] = row0.qs[  0.. 7]
dst[   8..15] = row1.qs[  0.. 7]
...
dst[  56..63] = row7.qs[  0.. 7]
dst[  64..71] = row0.qs[  8..15]
dst[  72..79] = row1.qs[  8..15]
...
dst[1016..1023] = row7.qs[120..127]
```

This is the pattern the Eä kernel must reproduce. No bit-twiddling on qs — it's
pure 8-byte gather/scatter.

All `memcpy`/`u64` copies in `make_block_q4_Kx8` are byte-identity moves; the Eä kernel must do an 8-byte byte copy, not a semantic u64 load/store. (Olorin only targets little-endian; this is documentation hygiene.)

### 4c. `scales` repacking (out.scales[96])

**Q4K scales convention (canonical source layout).** Before reading the repack,
it helps to know what `in[j].scales[i]` actually means in the source
`block_q4_K`. A Q4K block contains 8 sub-blocks of 32 weights each, so there
are 8 per-sub-block *scale* values and 8 per-sub-block *min* values, each 6
bits wide, packed into the 12-byte `scales[12]` field as follows:

- `scales[0..3]` hold the low 6 bits of the per-sub-block *scale* for
  sub-blocks 0..3 (one value per byte, bits 0..5).
- `scales[4..7]` hold the low 6 bits of the per-sub-block *min* for sub-blocks
  0..3 (bits 0..5).
- `scales[8..11]` hold the combined info for sub-blocks 4..7: the **low 4
  bits** of the scale for sub-block `4+k` in the low nibble of byte `8+k`, and
  the low 4 bits of the min for sub-block `4+k` in the high nibble of byte
  `8+k`.
- The **high 2 bits** of scale/min for sub-blocks 4..7 are stolen from bits
  6..7 of `scales[0..3]` (scales) and `scales[4..7]` (mins) respectively.

So reconstructing the 6-bit scale for sub-block `i` (`i` in 0..3) on row `j`
is `in[j].scales[i] & 63`, while for sub-block `i+4` it is
`((in[j].scales[i] & 0xC0) >> 2) | (in[j].scales[i+8] & 0x0F)`. The repack
below uses exactly this decomposition. (Verified against `ggml-common.h` near
`block_q4_K`; if that source ever changes, re-check before trusting this
summary.)

The repack rearranges these so that sub-block-`i` from row-`r` ends up at a
known offset inside the interleaved `scales[96]` block of `block_q4_Kx8`.

Source `scales[12]` is the standard Q4K 6-bit packed layout encoding 8 sub-block
scales and 8 sub-block mins. The repack unpacks into locals `s[0..7]` and
`m[0..7]` (each 6-bit values in a uint8) and then re-packs across 8 rows.

**First half of `out.scales` — sub-blocks 0..3 (lines 2868-2887):**

For each sub-block `i` in `0..3`, extract 8 scales and 8 mins from the 8 source rows:

```c
s[j] = in[j].scales[i]     & 63;    // low 6 bits, scales sub-block i, row j
m[j] = in[j].scales[i + 4] & 63;    // low 6 bits, mins   sub-block i, row j
```

Then write 12 output bytes at `out.scales[i*12 .. i*12+11]`:

```
out.scales[i*12 + 0] = (s[0] & 63) | ((s[4] & 48) << 2);
out.scales[i*12 + 1] = (s[1] & 63) | ((s[5] & 48) << 2);
out.scales[i*12 + 2] = (s[2] & 63) | ((s[6] & 48) << 2);
out.scales[i*12 + 3] = (s[3] & 63) | ((s[7] & 48) << 2);
out.scales[i*12 + 4] = (m[0] & 63) | ((m[4] & 48) << 2);
out.scales[i*12 + 5] = (m[1] & 63) | ((m[5] & 48) << 2);
out.scales[i*12 + 6] = (m[2] & 63) | ((m[6] & 48) << 2);
out.scales[i*12 + 7] = (m[3] & 63) | ((m[7] & 48) << 2);
out.scales[i*12 + 8] = (s[4] & 15) | ((m[4] & 15) << 4);
out.scales[i*12 + 9] = (s[5] & 15) | ((m[5] & 15) << 4);
out.scales[i*12 +10] = (s[6] & 15) | ((m[6] & 15) << 4);
out.scales[i*12 +11] = (s[7] & 15) | ((m[7] & 15) << 4);
```

(The source uses `+`; with the masks these are equivalent to bitwise OR — there
are never overlapping bits in any of the twelve expressions above. The Eä kernel
should use OR to be explicit about intent.)

Fills bytes 0..47 of `out.scales`.

**Second half of `out.scales` — sub-blocks 4..7 (lines 2889-2908):**

For each sub-block `i` in `0..3`, extract the *upper* scales/mins (sub-blocks
4..7 in the source), which live in the high 2 bits of `scales[0..7]` combined
with the nibbles of `scales[8..11]`:

```c
s[j] = ((in[j].scales[i]     & 192) >> 2) | (in[j].scales[i+8] & 15);
m[j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
```

Then write 12 output bytes at `out.scales[i*12 + 48 .. i*12 + 59]` with the
same twelve-line pattern as above (just offset by 48):

```
out.scales[i*12 + 48] = (s[0] & 63) | ((s[4] & 48) << 2);
out.scales[i*12 + 49] = (s[1] & 63) | ((s[5] & 48) << 2);
out.scales[i*12 + 50] = (s[2] & 63) | ((s[6] & 48) << 2);
out.scales[i*12 + 51] = (s[3] & 63) | ((s[7] & 48) << 2);
out.scales[i*12 + 52] = (m[0] & 63) | ((m[4] & 48) << 2);
out.scales[i*12 + 53] = (m[1] & 63) | ((m[5] & 48) << 2);
out.scales[i*12 + 54] = (m[2] & 63) | ((m[6] & 48) << 2);
out.scales[i*12 + 55] = (m[3] & 63) | ((m[7] & 48) << 2);
out.scales[i*12 + 56] = (s[4] & 15) | ((m[4] & 15) << 4);
out.scales[i*12 + 57] = (s[5] & 15) | ((m[5] & 15) << 4);
out.scales[i*12 + 58] = (s[6] & 15) | ((m[6] & 15) << 4);
out.scales[i*12 + 59] = (s[7] & 15) | ((m[7] & 15) << 4);
```

Note: in this second half, `s[0..7]` and `m[0..7]` are the *upper*
sub-block values (sub-blocks 4..7 of the source, i.e. `s4..s7`/`m4..m7`
in the preamble's terminology), reconstructed from the high-2-bits +
low-nibble encoding described above.

Fills bytes 48..95 of `out.scales`.

Summary of the scales layout:

- Bytes `[ 0..48)` of `out.scales` encode sub-blocks **0..3** of all 8 rows.
- Bytes `[48..96)` of `out.scales` encode sub-blocks **4..7** of all 8 rows.
- Within each half, the data for sub-block `i` (0..3) lives in a 12-byte group
  at offset `i*12` (or `48 + i*12`), laid out as:
    - bytes 0..3: 6-bit scales for rows 0..3 (low 6 bits of scales, high 2 bits
      stolen from row's sub-block+4 scales)
    - bytes 4..7: 6-bit mins for rows 0..3 (same trick)
    - bytes 8..11: low nibbles of scales/mins for rows 4..7 (scale in low
      nibble, min in high nibble)

The trick keeps each 12-byte group self-contained for all 8 rows of one
sub-block, so an AVX2 gemm can load scales with a single 12-byte read per
sub-block.

## 5. Repacked block size (for Eä buffer sizing)

```
sizeof(block_q4_Kx8) = 16 (d) + 16 (dmin) + 96 (scales) + 1024 (qs) = 1152 bytes
                     = 8 * sizeof(block_q4_K) = 8 * 144
```

**1152 bytes per repacked 8×1 super-block** (8 rows × 1 column-block of 256 weights each).

For a weight matrix with `nrow` rows and `ncol` columns where `nrow % 8 == 0`
and `ncol % 256 == 0`:

```
n_repacked_blocks = (nrow / 8) * (ncol / 256)
total_bytes       = n_repacked_blocks * 1152
```

This equals `nrow * nblocks * sizeof(block_q4_K)` — the `8 * 144 = 1152`
identity in section 2 already proves the repack does not add or remove bytes,
it only permutes them.

## 6. Loop order summary for the Eä kernel

```
for stripe in 0 .. nrow/8:
    for x in 0 .. nblocks:                       // column-block
        gather 8 source blocks: src[stripe*8*nblocks + i*nblocks + x], i=0..7
        emit one block_q4_Kx8 (1152 bytes):
            - 16 B: f16 d    [row0..row7]
            - 16 B: f16 dmin [row0..row7]
            - 96 B: repacked scales (sub-blocks 0..3 then 4..7, as above)
            - 1024 B: qs in 128 × 8-byte chunks, row-major within each
                      16-group of src byte offset
```

## 7. Notes for bit-exact reproduction

- `d`, `dmin` stay f16 — do not round-trip through f32.
- The scales byte expressions use **addition** in the source. In this specific
  layout no two added operands share a bit, so `|` and `+` produce identical
  results; the Eä kernel should use `|` for clarity.
- The qs interleave granularity is a fixed 8 bytes — do not generalise to 4
  bytes (that's `q4_K_4_bl`, a different repack).
- Row count must be a multiple of 8 and column count a multiple of 256; the
  caller is responsible for padding. `repack_q4_K_to_q4_K_8_bl` returns `-1`
  otherwise.
