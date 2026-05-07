//! Cross-platform home-directory lookup.
//!
//! Unix reads `$HOME`. Windows reads `%USERPROFILE%`. Returns `None`
//! if the relevant environment variable is unset — every caller in
//! the codebase handles missing-home gracefully (vault disables
//! persistence, runes fall back to `/tmp`, model loader returns
//! `None`), so we hand back the option and let them decide.

use std::path::PathBuf;

pub fn home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    let var = std::env::var("HOME");
    #[cfg(windows)]
    let var = std::env::var("USERPROFILE");
    var.ok().map(PathBuf::from)
}
