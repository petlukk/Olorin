//! Platform-specific syscall wrappers used by Olorin's storage and
//! inference layers. Each submodule exposes a thin cross-platform
//! free-function API and gates the actual syscall behind cfg.

pub mod futex;
pub mod home;
pub mod hwid;
pub mod lock;
pub mod mmap;
pub mod sysinfo;
