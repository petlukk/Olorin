//! Path guard for LLM-invoked tools — allowlist + sensitive denylist.
//!
//! Tools called by the model receive paths from prompt content. If a
//! prompt injection (or just a naively cooperating model) passes an
//! attacker-chosen path to `read_file` / `write_file` / `grep` / `ls`,
//! this module is the last line of defense before any FS operation.
//!
//! Two checks, lexical only — they fire before any disk access:
//! 1. Allowlist: canonical path must live under `$HOME` or `/tmp`.
//!    Mirrors `runes/common.rs::resolve_path`.
//! 2. Denylist: even within the allowlist, certain subtrees are
//!    off-limits (vault, SSH keys, AWS credentials, shell rc/history).
//!
//! [`resolve_safe_path`] is lexical only. [`resolve_safe_path_checked`]
//! additionally follows symlinks (`canonicalize`) and re-runs the same
//! policy on the *real* target, so a symlink under `$HOME` pointing into
//! `~/.ssh` or `~/.olorin` can't smuggle a sensitive file past the
//! lexical denylist. Tools that touch the filesystem use the checked
//! variant; the lexical one stays public for callers that only need the
//! string-level guard.

use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq)]
pub enum PathError {
    /// Path is outside `$HOME` and `/tmp`.
    OutsideAllowlist,
    /// Path is inside a sensitive subtree (e.g. `~/.olorin/`, `~/.ssh/`).
    Sensitive(&'static str),
    /// Path contains `..` — could escape the allowed roots.
    ParentTraversal,
    /// The path is (or routes through) a symlink that resolves outside the
    /// policy, or a dangling symlink whose target can't be checked.
    UnsafeSymlink,
    /// No `$HOME` / `%USERPROFILE%` — can't resolve `~/...`.
    NoHome,
}

impl PathError {
    /// Human-readable refusal message suitable for the LLM follow-up
    /// and the user. Tools route this through `ToolResult { success: false, output }`.
    pub fn refusal_message(&self) -> String {
        match self {
            PathError::OutsideAllowlist =>
                "refused: path outside the allowed roots ($HOME or /tmp)".into(),
            PathError::Sensitive(label) =>
                format!("refused: path under sensitive subtree ({label}); \
                         agent tools never read or write there"),
            PathError::ParentTraversal =>
                "refused: path contains `..` — could escape allowed roots".into(),
            PathError::UnsafeSymlink =>
                "refused: path is a symlink leading outside the allowed roots".into(),
            PathError::NoHome =>
                "refused: no $HOME / %USERPROFILE% in environment".into(),
        }
    }
}

/// Whether the tool intends to read or write. Reserved for future use
/// when read-only paths (e.g. ~/.gitconfig) need different treatment
/// from write-only (e.g. ~/.bashrc). Today both modes share the
/// denylist — kept in the API for forward compatibility.
#[derive(Debug, Clone, Copy)]
pub enum AccessMode { Read, Write }

/// $HOME-relative sensitive subtrees. Component-wise prefix matching
/// against the expanded path; `~/.ssh` denies `~/.ssh/id_rsa`,
/// `~/.bashrc` denies only the exact file (not `~/.bashrc_backup`,
/// which is a different component).
const SENSITIVE_RELATIVE: &[(&str, &str)] = &[
    (".olorin",           "Olorin vault directory"),
    (".ssh",              "SSH keys"),
    (".aws",              "AWS credentials"),
    (".gnupg",            "GPG keys"),
    (".config/anthropic", "Anthropic API config"),
    (".config/openai",    "OpenAI API config"),
    (".bashrc",           "shell rc file"),
    (".zshrc",            "shell rc file"),
    (".profile",          "shell profile"),
    (".bash_history",     "shell history"),
    (".zsh_history",      "shell history"),
];

/// Absolute sensitive paths. Many already fail the allowlist, but
/// keep them explicit so the refusal message names the reason.
const SENSITIVE_ABSOLUTE: &[(&str, &str)] = &[
    ("/etc/shadow",     "system password file"),
    ("/etc/sudoers",    "sudo config"),
    ("/etc/sudoers.d",  "sudo config"),
];

/// Resolve a user-supplied path string, returning the expanded
/// `PathBuf` if every guard passes. Performs no I/O — purely lexical.
pub fn resolve_safe_path(path: &str, _mode: AccessMode) -> Result<PathBuf, PathError> {
    let home = crate::home_dir().ok_or(PathError::NoHome)?;
    let expanded = expand(path, &home);
    enforce_policy(&expanded, &home)?;
    Ok(expanded)
}

/// Lexical guard + symlink-aware re-check. Runs [`resolve_safe_path`]
/// first, then `canonicalize`s the result (or, for a not-yet-existing
/// write target, its parent directory) and re-applies the allowlist +
/// denylist to the *real* path. This closes the symlink-escape gap:
/// `~/link -> ~/.ssh/id_rsa` passes the lexical denylist but its
/// canonical form lands in `~/.ssh` and is refused here.
///
/// Returns the canonical path on success so callers operate on the same
/// resolved target the guard approved.
pub fn resolve_safe_path_checked(path: &str, mode: AccessMode) -> Result<PathBuf, PathError> {
    let home = crate::home_dir().ok_or(PathError::NoHome)?;
    let expanded = resolve_safe_path(path, mode)?;

    // Compare against canonical roots so prefix matching is consistent
    // even when $HOME itself contains a symlink (or Windows adds a
    // verbatim `\\?\` prefix on canonicalize).
    let home_real = home.canonicalize().unwrap_or(home);

    // Fully-resolvable path: canonicalize follows every symlink in the chain;
    // re-check the real target.
    if let Ok(real) = expanded.canonicalize() {
        enforce_policy(&real, &home_real)?;
        return Ok(real);
    }

    // Not fully resolvable. If the leaf itself is a symlink, it's dangling —
    // refuse, so a write can't follow it into an unknown (possibly sensitive)
    // location. (A symlinked *parent* is still resolved below via the parent
    // canonicalize, which re-checks the real directory.)
    if let Ok(meta) = expanded.symlink_metadata() {
        if meta.file_type().is_symlink() {
            return Err(PathError::UnsafeSymlink);
        }
    }

    // Not-yet-existing target (e.g. a new file to write). Canonicalize the
    // parent directory — which must be real — and re-check parent + leaf, so a
    // symlinked parent pointing into a sensitive subtree is caught.
    match (expanded.parent(), expanded.file_name()) {
        (Some(parent), Some(file)) => match parent.canonicalize() {
            Ok(c) => {
                let candidate = c.join(file);
                enforce_policy(&candidate, &home_real)?;
                Ok(candidate)
            }
            // Parent missing too: nothing to follow, lexical guard already
            // passed, and the FS op will fail on its own.
            Err(_) => Ok(expanded),
        },
        _ => Ok(expanded),
    }
}

/// Shared allowlist + denylist policy. `home` is the root to measure
/// `$HOME`-relative sensitive subtrees against — the caller passes the
/// lexical home for [`resolve_safe_path`] and the canonical home for
/// [`resolve_safe_path_checked`].
fn enforce_policy(expanded: &Path, home: &Path) -> Result<(), PathError> {
    for comp in expanded.components() {
        if matches!(comp, std::path::Component::ParentDir) {
            return Err(PathError::ParentTraversal);
        }
    }

    let under_home = expanded.starts_with(home);
    // `/private/tmp` is the canonical form of `/tmp` on macOS; accept both
    // so the checked variant doesn't reject a legitimate /tmp target.
    let under_tmp = expanded.starts_with("/tmp") || expanded.starts_with("/private/tmp");
    if !under_home && !under_tmp {
        return Err(PathError::OutsideAllowlist);
    }

    if under_home {
        for (rel, label) in SENSITIVE_RELATIVE {
            if expanded.starts_with(home.join(rel)) {
                return Err(PathError::Sensitive(label));
            }
        }
    }
    for (abs, label) in SENSITIVE_ABSOLUTE {
        if expanded.starts_with(abs) {
            return Err(PathError::Sensitive(label));
        }
    }

    Ok(())
}

fn expand(path: &str, home: &Path) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        home.join(rest)
    } else if path == "~" {
        home.to_path_buf()
    } else if path.starts_with('/') {
        PathBuf::from(path)
    } else {
        home.join(path)
    }
}
