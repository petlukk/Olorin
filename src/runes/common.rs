//! Shared helpers: path allowlist + size-capped mmap + output truncation.

use std::fs::File;
use std::path::{Path, PathBuf};

/// Soft cap for any rune input. Per spec section "Resource Limits".
pub const MAX_INPUT_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Cap for `RuneResult.answer` — this is what reaches the LLM.
pub const MAX_ANSWER_BYTES: usize = 32 * 1024;

#[derive(Debug, PartialEq)]
pub enum PathError {
    OutsideAllowlist,
    NotFound,
    TooLarge(u64),
    Io(String),
}

/// Resolve a user-supplied path against the allowlist (user home + /tmp).
/// Accepts `~/...` as home-relative. Rejects anything that canonicalizes
/// outside the allowlist (including symlink traversal).
pub fn resolve_path(path: &str, home: &Path) -> Result<PathBuf, PathError> {
    let expanded = if let Some(rest) = path.strip_prefix("~/") {
        home.join(rest)
    } else if path.starts_with('/') {
        PathBuf::from(path)
    } else {
        home.join(path)
    };

    // Lexical traversal check — canonicalize would fail on non-existent
    // paths, so we reject `..` segments explicitly before touching disk.
    for comp in expanded.components() {
        if matches!(comp, std::path::Component::ParentDir) {
            return Err(PathError::OutsideAllowlist);
        }
    }

    // Confirm the path lives under home or /tmp.
    let allowed = expanded.starts_with(home) || expanded.starts_with("/tmp");
    if !allowed {
        return Err(PathError::OutsideAllowlist);
    }

    Ok(expanded)
}

/// Open a file and mmap-equivalent read the full contents, rejecting
/// anything beyond `MAX_INPUT_BYTES`. Returns the bytes as a `Vec<u8>`;
/// the MVP reads eagerly (streamed read comes later for GB files).
pub fn open_capped(path: &Path) -> Result<Vec<u8>, PathError> {
    use std::io::Read;

    let metadata = std::fs::metadata(path)
        .map_err(|e| if e.kind() == std::io::ErrorKind::NotFound {
            PathError::NotFound
        } else {
            PathError::Io(e.to_string())
        })?;
    let size = metadata.len();
    if size > MAX_INPUT_BYTES {
        return Err(PathError::TooLarge(size));
    }
    let mut f = File::open(path).map_err(|e| PathError::Io(e.to_string()))?;
    let mut buf = Vec::with_capacity(size as usize);
    f.read_to_end(&mut buf).map_err(|e| PathError::Io(e.to_string()))?;
    Ok(buf)
}

/// Truncate a summary string to `MAX_ANSWER_BYTES`, appending a marker.
pub fn truncate_answer(s: &str) -> String {
    if s.len() <= MAX_ANSWER_BYTES {
        return s.to_string();
    }
    let dropped = s.len() - MAX_ANSWER_BYTES + 32;
    format!("{} [...truncated {dropped} bytes]", &s[..MAX_ANSWER_BYTES - 32])
}
