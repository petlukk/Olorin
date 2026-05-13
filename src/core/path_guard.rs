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
//! Symlink-after-resolve is out of scope here — that's a per-tool
//! `canonicalize` check after the file exists. Lexical alone closes
//! the prompt-injection attack class.

use std::path::{Path, PathBuf};

#[derive(Debug, PartialEq)]
pub enum PathError {
    /// Path is outside `$HOME` and `/tmp`.
    OutsideAllowlist,
    /// Path is inside a sensitive subtree (e.g. `~/.olorin/`, `~/.ssh/`).
    Sensitive(&'static str),
    /// Path contains `..` — could escape the allowed roots.
    ParentTraversal,
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

    for comp in expanded.components() {
        if matches!(comp, std::path::Component::ParentDir) {
            return Err(PathError::ParentTraversal);
        }
    }

    let under_home = expanded.starts_with(&home);
    let under_tmp  = expanded.starts_with("/tmp");
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

    Ok(expanded)
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
