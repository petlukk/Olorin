# Security policy

Olorin is a local-first AI agent whose headline guarantees are about
**vault confidentiality**, **agent-tool path isolation**, and **prompt
injection resistance**. Vulnerabilities in those areas are taken
seriously even though the project is solo-maintained.

## Reporting a vulnerability

**Please do not file a public GitHub issue for security bugs.**

Use GitHub's private disclosure channel:
[**Report a vulnerability**](https://github.com/petlukk/Olorin/security/advisories/new).

Include enough detail to reproduce:

- Olorin version (`./olorin --version` or git SHA)
- Platform (Linux x86_64 / aarch64 / Windows) and how the binary was built
- Steps to reproduce, or a proof-of-concept
- What you observed vs. what the threat model claims

A best-effort acknowledgement within **7 days**, and a fix-or-mitigation
plan within **30 days** for confirmed reports. This is a solo project —
response times are best-effort, not guaranteed.

## Supported versions

Only the **latest released minor** receives security fixes. Earlier
releases are not patched; upgrade is the supported remediation.

| Version | Security fixes |
|---------|----------------|
| 2.0.x   | Yes            |
| < 2.0   | No             |

## In scope

Vulnerabilities in any of the following are in scope:

- **Vault confidentiality and integrity** — ChaCha20-Poly1305 AEAD,
  Argon2id key derivation (RFC 9106), Blake2b primitive, salt handling,
  tag verification, the `FusedSearcher` path.
- **SecureBuffer guarantees** — `mlock` behaviour, SIMD zeroize on Drop,
  any path where plaintext escapes the searcher's SIMD registers into
  general-purpose memory.
- **Agent tool path isolation** — `core/path_guard.rs` and
  `core/shell_guard.rs`; bypasses of the sensitive-subtree denylist
  (`~/.olorin`, `~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.bash_history`,
  `/etc/shadow`, etc.) through any built-in tool.
- **Prompt-injection bypass** — `core/safety.rs` score-based inbound
  matcher, including Unicode normalization, language coverage (EN/SV),
  and the `<rune_output untrusted="true">` wrapping that protects the
  LLM from rune-output instructions.
- **Constant-time guarantees** — `poly1305_verify` and any other
  authentication-tag comparison.
- **Memory safety** — any panic, integer overflow, out-of-bounds read or
  write, or use-after-free reachable from untrusted input (REPL, web UI,
  WhatsApp, vault file, GGUF model file, rune input file).

## Out of scope

The following are **acknowledged limitations**, not bugs:

- **Weak or leaked passphrases.** Argon2id raises the per-guess cost
  significantly; it does not protect against a dictionary-strength
  passphrase paired with a stolen vault file.
- **Sophisticated prompt injection.** Adversarial paraphrasing and
  out-of-distribution languages can slip past keyword + score matching.
  A full-ML classifier is a separate project, not on the current
  roadmap.
- **Host-level compromise.** Olorin is a local binary the user chooses
  to run; an attacker with shell access on the host can read process
  memory, install a malicious model, or replace the binary. Olorin does
  not attempt to defend against this.
- **Side-channel attacks** beyond the constant-time tag comparison —
  timing, cache, speculative-execution, power analysis. Not formally
  hardened.
- **Denial of service** from oversize input (extremely large GGUF
  models, malformed huge log files, etc.). Olorin is single-process and
  single-user; DoS is not a confidentiality or integrity threat in
  scope.
- **Third-party model weights.** Olorin loads GGUF files the user
  supplies; the project does not audit or attest model behaviour.
- **Cloud fallback (`ANTHROPIC_API_KEY`).** When the Anthropic cloud
  path is configured by the operator, request content leaves the host
  by design. That is the operator's policy choice, not an Olorin bug.

## Disclosure

Public disclosure happens **after a fix is released**, via the GitHub
Security Advisory and a `CHANGELOG.md` entry under the relevant
version. Coordinated disclosure with the reporter is the default;
researchers are credited unless they request otherwise.
