//! Shared math utilities for inference forward passes.

pub(crate) fn softmax_rows(data: &mut [f32], n_rows: usize, seq_len: usize) {
    for r in 0..n_rows {
        let row = &mut data[r * seq_len..(r + 1) * seq_len];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}

pub(crate) fn wipe_f32(buf: &mut [f32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn wipe_i8(buf: &mut [i8]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn wipe_i32(buf: &mut [i32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}
