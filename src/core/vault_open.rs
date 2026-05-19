//! Startup-time encrypted-vault bootstrap.
//!
//! Resolves the user's passphrase from one of two sources, runs the
//! Argon2id KDF, and hands back an open [`Vault`] — or `None` (with
//! an explanatory `eprintln`) when the vault can't be opened.  The
//! REPL entry point treats `None` as "persistence disabled, but
//! still let the user chat"; `--serve` / `--whatsapp` treat it as a
//! hard failure (see `DispatchContext::has_vault`).
//!
//! Source priority:
//!   1. `OLORIN_PASSPHRASE` env var — CI, scripts, non-interactive
//!   2. Interactive prompt on `/dev/tty` (Unix) or `CONIN$` (Windows)
//!   3. Neither available → return `None`

use crate::error::{Error, Result};
use crate::storage::secure::SecureBuffer;
use crate::storage::vault::Vault;

/// Prompt for the vault passphrase on the tty.  Asks once for an
/// existing vault, twice (with confirmation) for a fresh vault so
/// a typo doesn't lock conversation history behind an unknown
/// passphrase.
fn prompt_for_passphrase(is_new: bool) -> Result<SecureBuffer> {
    let first_prompt = if is_new {
        "[vault] new vault — set passphrase: "
    } else {
        "[vault] passphrase: "
    };
    let one = crate::platform::term::read_secret(first_prompt)?;
    if !is_new {
        return Ok(one);
    }
    let two = crate::platform::term::read_secret("[vault] confirm passphrase: ")?;
    if one.as_slice() != two.as_slice() {
        return Err(Error::Vault("passphrases did not match — try again"));
    }
    Ok(one)
}

/// Open the user's default vault, prompting for the passphrase on
/// the tty when the env var isn't set.  Returns `None` (with an
/// eprintln explaining why) when no passphrase source is available
/// or the open itself fails.
pub fn open_vault() -> Option<Vault> {
    let home = crate::home_dir().or_else(|| {
        eprintln!("[vault] home unset, persistence disabled");
        None
    })?;
    let vault_dir = home.join(".olorin").join("vault").join("default");
    std::fs::create_dir_all(&vault_dir)
        .map_err(|e| eprintln!("[vault] mkdir {} failed, persistence disabled: {e}", vault_dir.display()))
        .ok()?;
    let is_new_vault = !vault_dir.join("vault.bin").exists();

    let passphrase = if let Ok(p) = std::env::var("OLORIN_PASSPHRASE") {
        if p.is_empty() {
            eprintln!("[vault] OLORIN_PASSPHRASE is empty, persistence disabled");
            return None;
        }
        // Copy the env-var bytes into a SecureBuffer so the rest of
        // the function takes one code path.  The env-var string
        // itself stays in the process environment — there is no way
        // around that — but the bytes we hand to Argon2id are mlock'd.
        let mut buf = SecureBuffer::new(p.len());
        buf.write(p.as_bytes());
        buf
    } else if crate::platform::term::stdin_is_tty() {
        match prompt_for_passphrase(is_new_vault) {
            Ok(buf) => buf,
            Err(e) => {
                eprintln!("[vault] passphrase prompt failed: {e:?}, persistence disabled");
                return None;
            }
        }
    } else {
        eprintln!("[vault] no OLORIN_PASSPHRASE and stdin is not a tty — persistence disabled");
        return None;
    };

    Vault::open(&vault_dir, passphrase.as_slice())
        .map_err(|e| eprintln!("[vault] open {} failed, persistence disabled: {e:?}", vault_dir.display()))
        .ok()
}
