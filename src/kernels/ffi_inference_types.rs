//! Type aliases for inference kernel FFI function pointers.

pub type I2DotI8Fn = unsafe extern "C" fn(*const u8, *const i8, i32) -> i32;
pub type I2DotI8_4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8, *const i8, *mut i32, i32);
pub type I2DotI8_4RowDualFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *mut i32, *mut i32, i32);
pub type QuantF32I8Fn   = unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i32, i32);
pub type RmsnormFn      = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, f32);
pub type FusedAttnF32Fn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i32, i32, f32);
pub type I8Dot1RowFn    = unsafe extern "C" fn(*const i8, *const u8, i32) -> i32;
pub type I8Dot4RowFn    = unsafe extern "C" fn(*const i8, *const u8, *const u8, *const u8, *const u8, *mut i32, i32);
pub type SquaredReluFn  = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
pub type VecAddFn       = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
pub type QuantF32Q8kFn  = unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i32, i32);
pub type Q4kDotQ8kFn    = unsafe extern "C" fn(
    *const u8, *const i8, *const i32,
    i32, *const f32, *const f32) -> f32;
pub type Q4kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i32,
    *mut f32, i32,
    *const f32, *const f32, *const f32, *const f32,
    *const f32, *const f32, *const f32, *const f32);
pub type Q4kDot4RowDualFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i32,
    *mut f32, *mut f32, i32,
    *const f32, *const f32, *const f32, *const f32,
    *const f32, *const f32, *const f32, *const f32,
    *const f32, *const f32, *const f32, *const f32,
    *const f32, *const f32, *const f32, *const f32);
pub type Q6kDotQ8kFn = unsafe extern "C" fn(
    *const u8, *const i8, *const i32,
    i32, *const f32) -> f32;
pub type Q6kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i32, *mut f32, i32,
    *const f32, *const f32, *const f32, *const f32);
pub type ApplyRopeFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32);
#[allow(clippy::type_complexity)]
pub type Q4kGemm4x4Fn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i8, *const i8, *const i8,
    *const i32, *const i32, *const i32, *const i32,
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const f32, *const f32, *const f32, *const f32,
    *const f32, *const f32, *const f32, *const f32,
    *const f32, *const f32, *const f32, *const f32,
    *mut f32, i32,
);

pub type QuantizeSIMDFn   = unsafe extern "C" fn(*const f32, *mut i32, *mut f32, *mut f32, i32);
pub type DequantizeSIMDFn = unsafe extern "C" fn(*const u8, *const f32, *const f32, *mut f32, i32);
pub type KScoreMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32);
pub type KScoreGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32, i32);
pub type VSumMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32);
pub type VSumGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32, i32);
pub type FusedAttentionFn = unsafe extern "C" fn(
    *const f32,
    *const u8, *const f32, *const f32,
    *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32);
pub type FusedCausalAttnFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *const u8, *const f32, *const f32,
    *mut f32, *mut f32, i32, i32, i32, i32, i32);
pub type ValidateFn = unsafe extern "C" fn(*const f32, *const f32, *const i32, *const i32, i32) -> i32;
