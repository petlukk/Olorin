//! End-to-end test for task #5 of arc 3: server entry points refuse
//! to start without an opened vault.
//!
//! The REPL is allowed to start with persistence disabled (you can
//! still ask one-off questions), but `--serve` and `--whatsapp` are
//! background processes whose only point is to serve a persistent
//! session.  Starting them without a vault silently throws away every
//! conversation on restart and means the vault-stored API key is
//! never loaded — a footgun the operator never gets to see.
//!
//! Spawns the real `olorin` binary via `CARGO_BIN_EXE_olorin` with
//! stdin piped (so `stdin_is_tty()` is false), HOME pointed at a
//! fresh temp dir, and `OLORIN_PASSPHRASE` deliberately unset.  Under
//! those conditions the only outcome consistent with task #5 is a
//! non-zero exit and a clear stderr message.

use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

fn unique_home(label: &str) -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_server_vault_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp HOME");
    dir
}

#[test]
fn serve_refuses_to_start_without_passphrase() {
    let home = unique_home("serve_no_pass");

    let out = Command::new(OLORIN)
        .args(["--serve", "--strict", "--port", "0"])
        .env_remove("OLORIN_PASSPHRASE")
        .env("HOME", &home)
        // Pipe stdin so isatty() returns false — the tty-prompt path
        // can't kick in and disguise the missing-passphrase case.
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn olorin --serve");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "expected non-zero exit, got {:?}\nstderr: {stderr}",
        out.status
    );
    assert!(
        stderr.contains("vault unavailable") && stderr.contains("--serve"),
        "stderr should explain the missing-vault failure; got:\n{stderr}"
    );
}

#[test]
fn whatsapp_refuses_to_start_without_passphrase() {
    let home = unique_home("whatsapp_no_pass");

    let out = Command::new(OLORIN)
        .args(["--whatsapp"])
        .env_remove("OLORIN_PASSPHRASE")
        // OLORIN_BRIDGE is irrelevant — we must fail before the
        // bridge spawn, otherwise the test is testing the wrong path.
        .env_remove("OLORIN_BRIDGE")
        .env("HOME", &home)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn olorin --whatsapp");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "expected non-zero exit, got {:?}\nstderr: {stderr}",
        out.status
    );
    assert!(
        stderr.contains("vault unavailable") && stderr.contains("--whatsapp"),
        "stderr should explain the missing-vault failure; got:\n{stderr}"
    );
    // Bridge process must not have been spawned — vault check runs
    // first.  If `wa-bridge` is on PATH or in the build tree this
    // would otherwise leave a child running.
    assert!(
        !stderr.contains("WhatsApp bridge started"),
        "bridge must not be spawned when vault auth fails; got:\n{stderr}"
    );
}
