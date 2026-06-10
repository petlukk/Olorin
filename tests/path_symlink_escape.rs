//! HIGH-3 regression: the FS tools must not be tricked into reading or
//! writing a sensitive file through a symlink.
//!
//! `path_guard`'s lexical check approves `~/link` because the *string* lives
//! under $HOME and isn't in the denylist. But if `~/link` is a symlink to
//! `~/.ssh/id_rsa`, the real target is sensitive. The checked resolver
//! canonicalizes and re-applies the policy to the real path; these tests
//! prove the gap is closed end-to-end through `run_tool`.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

use olorin::core::path_guard::{resolve_safe_path, resolve_safe_path_checked, AccessMode, PathError};
use olorin::tools::run_tool;

const SECRET: &str = "SECRET-KEY-MATERIAL-do-not-leak";

fn unique_home() -> PathBuf {
    std::env::temp_dir().join(format!("olorin_symlink_test_{}", std::process::id()))
}

/// All assertions live in one test so the process-global `HOME` mutation
/// can't race a sibling test.
#[test]
fn symlink_into_sensitive_subtree_is_refused_everywhere() {
    let home = unique_home();
    let _ = fs::remove_dir_all(&home);
    fs::create_dir_all(home.join(".ssh")).unwrap();
    fs::write(home.join(".ssh/id_rsa"), SECRET).unwrap();
    fs::write(home.join("notes.txt"), "hello world").unwrap();
    // ~/link -> ~/.ssh/id_rsa  (existing-target symlink)
    symlink(home.join(".ssh/id_rsa"), home.join("link")).unwrap();
    // ~/wlink -> ~/.ssh/implanted  (dangling: target doesn't exist yet)
    symlink(home.join(".ssh/implanted"), home.join("wlink")).unwrap();

    std::env::set_var("HOME", &home);

    // 1. The lexical guard alone is fooled — this is the gap we're closing.
    assert!(
        resolve_safe_path("~/link", AccessMode::Read).is_ok(),
        "lexical guard is expected to pass the symlink (demonstrates the gap)"
    );

    // 2. The checked guard resolves the link and refuses the real target.
    match resolve_safe_path_checked("~/link", AccessMode::Read) {
        Err(PathError::Sensitive(_)) => {}
        other => panic!("expected Sensitive refusal, got {other:?}"),
    }

    // 3. read_file tool must refuse and must NOT return the secret bytes.
    let r = run_tool("read", "~/link").unwrap();
    assert!(!r.success, "read through symlink must fail");
    assert!(!r.output.contains(SECRET), "secret must never leak: {}", r.output);

    // 4. write through a dangling symlink into .ssh must be refused, and must
    //    NOT create the target file inside the sensitive subtree.
    let w = run_tool("write", "~/wlink pwned").unwrap();
    assert!(!w.success, "write through dangling symlink must fail");
    assert!(
        !home.join(".ssh/implanted").exists(),
        "write must not have followed the symlink into ~/.ssh"
    );

    // 5. Benign reads/writes still work — the guard isn't over-blocking.
    let ok = run_tool("read", "~/notes.txt").unwrap();
    assert!(ok.success && ok.output.contains("hello world"));
    let okw = run_tool("write", "~/fresh.txt data").unwrap();
    assert!(okw.success, "writing a normal new file must still succeed");
    assert_eq!(fs::read_to_string(home.join("fresh.txt")).unwrap(), "data");

    let _ = fs::remove_dir_all(&home);
}
