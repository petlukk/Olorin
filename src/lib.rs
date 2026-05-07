#[macro_export]
macro_rules! olorin_debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        eprintln!("[DBG] {}", format_args!($($arg)*))
    }
}

pub mod error;
pub mod core;
pub mod inference;
pub mod storage;
pub mod interface;
pub mod tools;
pub mod kernels;
pub mod platform;
pub mod recall;
pub mod runes;

pub use platform::home::home_dir;
