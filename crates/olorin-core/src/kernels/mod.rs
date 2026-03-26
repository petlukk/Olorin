pub mod ffi;

pub mod arg_tokenizer;
pub mod byte_classifier;
pub mod command_router;
pub mod fused_safety;
pub mod json_scanner;
pub mod leak_scanner;
pub mod sanitizer_kernel;
pub mod search;

/// Initialize the embedded SIMD kernels. Must be called once at startup.
pub fn init() -> Result<(), String> {
    ffi::init()
}
