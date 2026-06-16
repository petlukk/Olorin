pub mod ffi;
pub mod ffi_crypto;
pub mod ffi_data;
pub mod ffi_inference;
pub mod ffi_inference_types;
pub mod ffi_types;
pub mod loader;

/// Platform-native dynamic-library filename for a kernel stem.
/// `command_router` becomes `libcommand_router.so` on Linux,
/// `libcommand_router.dylib` on macOS, `command_router.dll` on Windows.
pub fn dynlib_filename(name: &str) -> String {
    #[cfg(target_os = "windows")]
    { format!("{name}.dll") }
    #[cfg(target_os = "macos")]
    { format!("lib{name}.dylib") }
    #[cfg(all(unix, not(target_os = "macos")))]
    { format!("lib{name}.so") }
}
