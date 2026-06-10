//! HIGH-2 regression: the shell classifier must not be evaded by newlines,
//! command substitution, or quote-split sensitive paths.
//!
//! `shell.rs` runs `sh -c <raw>`, so a newline terminates a command and
//! `$(...)`/backticks run nested commands. Before the fix, the classifier
//! only split on `; && || |`, so destructive/exfil commands hid behind an
//! allowed outer command. The sensitive-path block was a literal substring
//! match, defeated by `cat ~/.s""sh/id_rsa`.

use olorin::core::shell_guard::{ShellGuard, ShellPolicy};

#[test]
fn newline_hidden_destructive_is_blocked() {
    let g = ShellGuard::new(ShellPolicy::Safe);
    assert!(
        g.check("echo hi\nrm -rf ~/Documents").is_err(),
        "newline-separated rm -rf must be classified"
    );
    assert!(g.check("ls\r\nshutdown now").is_err(), "CRLF separator too");
}

#[test]
fn command_substitution_destructive_is_blocked() {
    let g = ShellGuard::new(ShellPolicy::Safe);
    assert!(g.check("echo $(rm -rf ~/x)").is_err(), "$() body must be classified");
    assert!(g.check("echo `rm -rf ~/x`").is_err(), "backtick body must be classified");
    assert!(
        g.check("echo $(echo $(rm -rf ~/x))").is_err(),
        "nested substitution must be classified"
    );
}

#[test]
fn quote_split_sensitive_path_is_blocked() {
    // The path-textual block fires in every policy mode, including Open.
    let g = ShellGuard::new(ShellPolicy::Open);
    assert!(g.check("cat ~/.s\"\"sh/id_rsa").is_err(), "double-quote split");
    assert!(g.check("cat ~/.s''sh/id_rsa").is_err(), "single-quote split");
    assert!(g.check("cat ~/.olo\"\"rin/vault/default/vault.bin").is_err());
}

#[test]
fn plain_sensitive_path_still_blocked() {
    let g = ShellGuard::new(ShellPolicy::Open);
    assert!(g.check("cat ~/.ssh/id_rsa").is_err());
    assert!(g.check("cp ~/.olorin/vault/default/vault.bin /tmp/x").is_err());
}

#[test]
fn benign_substitution_and_commands_still_allowed() {
    let g = ShellGuard::new(ShellPolicy::Safe);
    // Substitution body is itself a read-only command — must not over-block.
    assert!(g.check("echo $(date)").is_ok(), "echo $(date) is harmless");
    assert!(g.check("echo today is $(whoami)").is_ok());
    assert!(g.check("ls -la").is_ok());
    assert!(g.check("grep -rn foo .").is_ok());
}
