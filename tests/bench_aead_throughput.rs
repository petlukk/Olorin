//! ChaCha20-Poly1305 AEAD seal throughput across plaintext sizes.
//!
//! Run with:
//!   cargo test --release --test bench_aead_throughput -- --nocapture --test-threads=1
//!
//! Reports ns/byte and GB/s per plaintext size.  Captures the per-block
//! fixed cost (header + OTK derivation + final reduction) at small sizes
//! and the stream rate at large sizes.

use olorin::kernels::ffi;
use olorin::storage::aead;
use std::time::Instant;

fn bench_seal(pt_len: usize, iters: usize) -> (f64, f64) {
    let key = [0x42u8; 32];
    let nonce = [0x07u8; 12];
    let aad = b"olorin-vault";
    let pt = vec![0xCDu8; pt_len];

    let mut buf = pt.clone();
    let mut tag = [0u8; 16];
    // Warm-up so the first iteration doesn't include first-touch costs.
    for _ in 0..10 {
        buf.copy_from_slice(&pt);
        aead::seal(&key, &nonce, aad, &mut buf, &mut tag);
    }

    let total_bytes = (pt_len * iters) as u64;
    let start = Instant::now();
    for _ in 0..iters {
        buf.copy_from_slice(&pt);
        aead::seal(&key, &nonce, aad, &mut buf, &mut tag);
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;

    let ns_per_byte = elapsed_ns / total_bytes as f64;
    let gb_per_sec = (total_bytes as f64) / (elapsed_ns / 1e9) / 1e9;
    (ns_per_byte, gb_per_sec)
}

#[test]
fn report_aead_throughput() {
    ffi::init().expect("kernel init");
    println!();
    println!("=== ChaCha20-Poly1305 AEAD seal throughput ===");
    println!("  {:>6}  {:>7}  {:>10}  {:>9}", "bytes", "iters", "ns/byte", "GB/s");
    for &len in &[64usize, 256, 1024, 4096, 16384] {
        let iters = (1_000_000 / len.max(1)).max(100);
        let (ns_per_byte, gb_per_sec) = bench_seal(len, iters);
        println!("  {len:>6}  {iters:>7}  {ns_per_byte:>10.3}  {gb_per_sec:>9.3}");
    }
    println!();
}
