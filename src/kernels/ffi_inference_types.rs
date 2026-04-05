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
