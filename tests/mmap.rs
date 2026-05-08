//! Cross-platform contract test for `platform::mmap::map_file_readonly`.
//! Exercises both the Unix `mmap` backend and the Windows
//! `CreateFileMappingW` backend through the same assertions.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use olorin::platform::mmap::map_file_readonly;

struct TempFile(PathBuf);

impl TempFile {
    fn new(name: &str, payload: &[u8]) -> Self {
        let mut path = std::env::temp_dir();
        let pid = std::process::id();
        path.push(format!("olorin-mmap-test-{pid}-{name}"));
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(payload).unwrap();
        f.sync_all().unwrap();
        TempFile(path)
    }
}

impl Drop for TempFile {
    fn drop(&mut self) { let _ = fs::remove_file(&self.0); }
}

#[test]
fn maps_payload_visible_through_slice() {
    let payload = b"hello mmap world";
    let f = TempFile::new("payload", payload);

    let view = map_file_readonly(&f.0).unwrap();
    assert_eq!(view.len(), payload.len());
    assert!(!view.is_empty());
    assert_eq!(view.as_slice(), payload);
}

#[test]
fn empty_file_is_invalid_data() {
    let f = TempFile::new("empty", b"");
    match map_file_readonly(&f.0) {
        Ok(_)  => panic!("expected error for empty file"),
        Err(e) => assert_eq!(e.kind(), std::io::ErrorKind::InvalidData),
    }
}

#[test]
fn missing_file_errors() {
    let mut path = std::env::temp_dir();
    path.push(format!("olorin-mmap-no-such-file-{}", std::process::id()));
    assert!(map_file_readonly(&path).is_err());
}

#[test]
fn view_outlives_its_open() {
    // Opening, reading, dropping — checks that Drop's munmap /
    // UnmapViewOfFile doesn't double-free or leak.
    let payload = vec![0xAB_u8; 8192];
    let f = TempFile::new("dropcycle", &payload);
    for _ in 0..16 {
        let view = map_file_readonly(&f.0).unwrap();
        assert_eq!(view.as_slice()[0], 0xAB);
        assert_eq!(view.as_slice()[8191], 0xAB);
    }
}
