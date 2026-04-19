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
/// Accepts `~/...` as home-relative. Performs lexical-only validation —
/// rejects `..` components and paths that don't start with `home` or `/tmp`
/// after expansion. Does NOT follow symlinks; see `open_capped` for the
/// canonical-path check that covers symlink traversal.
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
///
/// After confirming the file exists, canonicalizes the path and re-checks
/// that the canonical form lives under `home` or `/tmp`. This catches
/// symlinks in `/tmp` (or elsewhere in the allowlist) that point outside it.
pub fn open_capped(path: &Path, home: &Path) -> Result<Vec<u8>, PathError> {
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

    // Canonicalize now that we know the file exists, and re-check the
    // allowlist on the real path to catch symlink traversal.
    let canonical = std::fs::canonicalize(path)
        .map_err(|e| PathError::Io(e.to_string()))?;
    let allowed = canonical.starts_with(home) || canonical.starts_with("/tmp");
    if !allowed {
        return Err(PathError::OutsideAllowlist);
    }

    let mut f = File::open(path).map_err(|e| PathError::Io(e.to_string()))?;
    let mut buf = Vec::with_capacity(size as usize);
    f.read_to_end(&mut buf).map_err(|e| PathError::Io(e.to_string()))?;
    Ok(buf)
}

/// Truncate a summary string to `MAX_ANSWER_BYTES`, appending a marker.
/// Walks the cut point back to the nearest valid UTF-8 char boundary so
/// multi-byte characters (e.g. Swedish letters, emoji) never cause a panic.
pub fn truncate_answer(s: &str) -> String {
    if s.len() <= MAX_ANSWER_BYTES {
        return s.to_string();
    }
    let cut = MAX_ANSWER_BYTES - 32;
    let cut = (0..=cut).rev().find(|&i| s.is_char_boundary(i)).unwrap_or(0);
    let dropped = s.len() - cut;
    format!("{} [...truncated {dropped} bytes]", &s[..cut])
}
