//! Raw FFI bindings to eakv — Q4 KV cache compression library.

use std::ffi::c_int;
use std::os::raw::c_uchar;

// Opaque type
#[allow(non_camel_case_types)]
pub enum eakv_cache_t {}

pub const EAKV_OK: c_int = 0;
pub const EAKV_ERR_INVALID: c_int = -4;

extern "C" {
    pub fn eakv_cache_create(
        n_layers: c_int, n_kv_heads: c_int,
        head_dim: c_int, max_seq_len: c_int,
    ) -> *mut eakv_cache_t;

    pub fn eakv_cache_free(cache: *mut eakv_cache_t);

    pub fn eakv_cache_seq_len(cache: *const eakv_cache_t) -> c_int;

    pub fn eakv_checkpoint(cache: *mut eakv_cache_t) -> c_int;

    pub fn eakv_restore(cache: *mut eakv_cache_t, seq_len: c_int) -> c_int;

    pub fn eakv_from_llama_state_append(
        cache: *mut eakv_cache_t,
        state_buf: *const c_uchar, state_size: usize,
        n_layers: c_int, n_kv_heads: c_int, head_dim: c_int,
        start_pos: c_int,
    ) -> c_int;
}

/// Safe wrapper around an eakv cache.
pub struct EakvCache {
    ptr: *mut eakv_cache_t,
    n_layers: i32,
    n_kv_heads: i32,
    head_dim: i32,
}

unsafe impl Send for EakvCache {}

impl EakvCache {
    pub fn new(n_layers: i32, n_kv_heads: i32, head_dim: i32, max_seq_len: i32) -> Option<Self> {
        let ptr = unsafe { eakv_cache_create(n_layers, n_kv_heads, head_dim, max_seq_len) };
        if ptr.is_null() { None } else {
            Some(Self { ptr, n_layers, n_kv_heads, head_dim })
        }
    }

    pub fn checkpoint(&mut self) -> i32 {
        unsafe { eakv_checkpoint(self.ptr) }
    }

    pub fn restore(&mut self, seq_len: i32) -> Result<(), String> {
        let rc = unsafe { eakv_restore(self.ptr, seq_len) };
        if rc == EAKV_OK { Ok(()) } else {
            Err(format!("eakv_restore failed: {rc}"))
        }
    }

    pub fn seq_len(&self) -> i32 {
        unsafe { eakv_cache_seq_len(self.ptr) }
    }

    /// Import KV state from llama.cpp (Approach A).
    /// Appends new tokens starting from start_pos.
    pub fn import_llama_state(&mut self, state: &[u8], start_pos: i32) -> Result<(), String> {
        let rc = unsafe {
            eakv_from_llama_state_append(
                self.ptr, state.as_ptr(), state.len(),
                self.n_layers, self.n_kv_heads, self.head_dim,
                start_pos,
            )
        };
        if rc == EAKV_OK { Ok(()) } else {
            Err(format!("eakv_from_llama_state_append failed: {rc}"))
        }
    }
}

impl Drop for EakvCache {
    fn drop(&mut self) {
        unsafe { eakv_cache_free(self.ptr) }
    }
}
