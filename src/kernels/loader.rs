//! Kernel extraction: writes the embedded `.so` / `.dll` files for the
//! current Olorin version to `~/.olorin/lib/{version}-{hash}/` and
//! publishes a `.extracted` marker once they're all in place. Separated
//! from `ffi.rs` to keep that file under the 500-LOC cap.

use std::path::PathBuf;

use super::ffi::embedded;

/// Return the versioned kernel directory path.
pub fn kernel_dir() -> Result<PathBuf, String> {
    let home = crate::home_dir()
        .ok_or_else(|| "home directory not found".to_string())?;
    Ok(home
        .join(".olorin")
        .join("lib")
        .join(embedded::VERSION))
}

/// Extract embedded kernel libraries to the per-version directory.
/// Writes each file to a per-process per-thread `.tmp` path then
/// `rename()`s onto the final path — atomic vs concurrent extraction
/// by other processes/threads, so a parallel `libloading` consumer
/// never sees a half-written `.so`. Public so tests can drive
/// extraction without going through `OnceLock`-guarded `init()`.
pub fn extract_kernels() -> Result<PathBuf, String> {
    let dir = kernel_dir()?;
    let marker = dir.join(".extracted");
    if marker.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let pid = std::process::id();
    let tid = format!("{:?}", std::thread::current().id());
    let tid_clean = tid.replace(|c: char| !c.is_ascii_alphanumeric(), "");
    for (_id, filename, bytes) in embedded::FILES {
        let final_path = dir.join(filename);
        if let Ok(meta) = std::fs::metadata(&final_path) {
            if meta.len() == bytes.len() as u64 {
                continue;
            }
        }
        let tmp_path = dir.join(format!("{filename}.{pid}.{tid_clean}.tmp"));
        std::fs::write(&tmp_path, bytes)
            .map_err(|e| format!("failed to write {}: {e}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp_path);
            format!("failed to publish {}: {e}", final_path.display())
        })?;
    }

    let marker_tmp = dir.join(format!(".extracted.{pid}.{tid_clean}.tmp"));
    std::fs::write(&marker_tmp, embedded::VERSION)
        .map_err(|e| format!("failed to write marker tmp: {e}"))?;
    std::fs::rename(&marker_tmp, &marker).map_err(|e| {
        let _ = std::fs::remove_file(&marker_tmp);
        format!("failed to publish marker: {e}")
    })?;

    eprintln!("olorin: extracted kernels to {}", dir.display());
    Ok(dir)
}
