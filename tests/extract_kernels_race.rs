//! Race test for `ffi::extract_kernels` under concurrent callers.
//!
//! Reproduces the bug that surfaced during ealog test development:
//! `cargo test` runs many test binaries in parallel, all sharing
//! `~/.olorin/lib/{ver}-{hash}/`. The previous implementation used
//! `std::fs::write` (open with TRUNC) directly on the final path, so
//! one process could open-truncate a `.so` while another was still
//! mid-write, leaving a 0-byte file for the next libloading consumer
//! to choke on with "file too short".
//!
//! The fix writes each kernel to a per-pid `.tmp` path then
//! `rename()`s it onto the final path. POSIX rename + Windows
//! `MoveFileExW(REPLACE_EXISTING)` are atomic for same-volume moves,
//! so any concurrent reader sees either the old version or a
//! complete new version.
//!
//! This test uses threads — within one process — to hammer the same
//! extraction path concurrently. `extract_kernels` is exposed (not
//! `OnceLock`-guarded like `init`) precisely so this race is reachable
//! from a test. Threads also fairly approximate cross-process races
//! against the same directory because the failure mode is identical:
//! racing TRUNC+write on the final path.

use olorin::kernels::ffi;

#[test]
fn extract_kernels_atomic_under_concurrency() {
    let dir = ffi::kernel_dir().expect("kernel_dir");
    // Clean cache so every thread sees marker missing on entry.
    let _ = std::fs::remove_dir_all(&dir);

    let n_threads = 8;
    let mut handles = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        handles.push(std::thread::spawn(|| {
            ffi::extract_kernels().expect("extract_kernels under race")
        }));
    }
    for h in handles {
        let returned = h.join().expect("thread panic");
        assert_eq!(returned, dir, "extract_kernels returned wrong dir");
    }

    // Marker file must exist and be non-empty.
    let marker = dir.join(".extracted");
    let marker_meta = std::fs::metadata(&marker).expect(".extracted marker absent");
    assert!(marker_meta.len() > 0, "marker file is empty");

    // Every embedded kernel file must exist at the final path and be
    // exactly the size of its embedded source (proves no truncation).
    for (_id, filename, expected) in ffi::embedded::FILES {
        let path = dir.join(filename);
        let meta = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("metadata for {filename}: {e}"));
        assert_eq!(
            meta.len(),
            expected.len() as u64,
            "{filename}: kernel file is {} bytes, expected {} (truncation regression)",
            meta.len(),
            expected.len(),
        );
    }

    // No stray `.tmp` files left behind by failed renames.
    for entry in std::fs::read_dir(&dir).expect("read_dir") {
        let entry = entry.expect("entry");
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        assert!(
            !name_str.ends_with(".tmp") && !name_str.contains(".tmp."),
            "leftover tmp file: {name_str}"
        );
    }
}

