//! Type aliases for inference kernel FFI function pointers.

pub type QuantF32Q8kFn  = unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i16, i32);
pub type Q4kDotQ8kFn    = unsafe extern "C" fn(
    *const u8, *const i8, *const i16,
    i32, *const f32, *const f32) -> f32;
pub type Q4kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i16,
    *mut f32, i32, *const f32, *const f32);
pub type Q4kDot4RowDualFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i16,
    *mut f32, *mut f32, i32, *const f32, *const f32);
pub type Q5kDotQ8kFn = unsafe extern "C" fn(
    *const u8, *const i8, *const i16,
    i32, *const f32, *const f32) -> f32;
pub type Q5kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i16,
    *mut f32, i32, *const f32, *const f32);
pub type Q6kDotQ8kFn = unsafe extern "C" fn(
    *const u8, *const i8, *const i16,
    i32, *const f32) -> f32;
pub type Q6kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i16, *mut f32, i32,
    *const f32, *const f32, *const f32, *const f32);
pub type F32ToF16Fn = unsafe extern "C" fn(*const f32, *mut u16, i32);
pub type F16ToF32Fn = unsafe extern "C" fn(*const u16, *mut f32, i32);
pub type SoftmaxF32Fn = unsafe extern "C" fn(*mut f32, i32, f32);
pub type Gemma4RmsnormFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, f32);
pub type GeluMulFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
pub type Gemma4RopeFn = unsafe extern "C" fn(*mut f32, *const f32, *const f32, i32, i32);
pub type Bf16DotF32Fn = unsafe extern "C" fn(*const u16, *const f32, *mut i32, i32) -> f32;
pub type Bf16Dot4RowFn = unsafe extern "C" fn(
    *const u16, *const u16, *const u16, *const u16,
    *const f32, *mut f32, *mut i32, i32);
pub type VecAddF32Fn   = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
pub type VecScaleF32Fn = unsafe extern "C" fn(*const f32, *mut f32, f32, i32);
pub type VecFmaF32Fn   = unsafe extern "C" fn(*const f32, *const f32, *mut f32, f32, i32);
pub type VecAccF32Fn   = unsafe extern "C" fn(*mut f32, *const f32, f32, i32);
pub type F32DotFn      = unsafe extern "C" fn(*const f32, *const f32, i32) -> f32;
pub type F32DotAccFn   = unsafe extern "C" fn(*mut f32, *const f32, f32, i32);
pub type BareRmsnormF32Fn = unsafe extern "C" fn(*mut f32, i32, f32);
pub type SoftcapF32Fn = unsafe extern "C" fn(*mut f32, i32, f32);
pub type Q4kRepack8x8Fn = unsafe extern "C" fn(
    *const u8,   // src (standard Q4K blocks, row-major)
    *mut u8,     // dst (block_q4_Kx8 interleaved tiles)
    i32,         // n_rows (multiple of 8)
    i32,         // n_cols (multiple of 256)
);
pub type Q4k8x8MatvecFn = unsafe extern "C" fn(
    *const u8,   // packed (block_q4_Kx8 tiles)
    *const i8,   // q8_qs
    *const f32,  // q8_d
    *const i16,  // q8_bsums
    *const f32,  // pow2 (scale LUT, same as q4k_dot_q8k)
    *mut u8,     // scratch (utmp[32] = 128 bytes, caller-owned)
    *mut f32,    // out (n_rows scores)
    i32,         // n_rows
    i32,         // n_cols
);
pub type Q4k8x8MatvecDualFn = unsafe extern "C" fn(
    *const u8,   // packed_a (block_q4_Kx8 tiles, first weight matrix)
    *const u8,   // packed_b (block_q4_Kx8 tiles, second weight matrix)
    *const i8,   // q8_qs    (shared Q8K input column)
    *const f32,  // q8_d     (shared)
    *const i16,  // q8_bsums (shared)
    *const f32,  // pow2     (shared scale LUT)
    *mut u8,     // scratch  (128 bytes, shared — bsums hadd only depends on Q8K)
    *mut f32,    // out_a    (n_rows scores for first matrix)
    *mut f32,    // out_b    (n_rows scores for second matrix)
    i32,         // n_rows
    i32,         // n_cols
);
pub type Q8kRepack4Fn = unsafe extern "C" fn(
    *const i8,   // row0_qs
    *const i8,   // row1_qs
    *const i8,   // row2_qs
    *const i8,   // row3_qs
    *const f32,  // row_d (4 × nb floats)
    *const i16,  // row0_bsums
    *const i16,  // row1_bsums
    *const i16,  // row2_bsums
    *const i16,  // row3_bsums
    *mut u8,     // dst (block_q8_Kx4 output)
    i32,         // nb (number of super-blocks)
);
pub type Q4k8x8GemmFn = unsafe extern "C" fn(
    *const u8,   // packed (block_q4_Kx8 tiles)
    *const u8,   // q8_a (block_q8_Kx4 tiles)
    *mut u8,     // scratch (64+ bytes)
    *mut f32,    // out
    i32,         // bs (row stride in floats)
    i32,         // n (inner dimension)
    i32,         // nr (A rows, must be % 4 == 0)
    i32,         // nc (B cols, must be % 8 == 0)
);
