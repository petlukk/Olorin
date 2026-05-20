#!/usr/bin/env bash
# Olorin installer (Linux). Downloads the latest release binary, optionally
# configures the Anthropic cloud fallback key, and optionally installs the
# WhatsApp bridge. Pipe-safe (curl | sh) — reads prompts from /dev/tty.
#
#   curl -fsSL https://raw.githubusercontent.com/petlukk/Olorin/main/scripts/install.sh | sh
#
# Honour these env vars to skip the prompts (useful in CI / scripted setup):
#   OLORIN_VERSION       — release tag to install (default: latest)
#   OLORIN_INSTALL_DIR   — install location (default: ~/.local/bin)
#   OLORIN_WITH_BRIDGE   — "yes" / "no" to skip the bridge prompt
#   OLORIN_WITH_PATH     — "yes" / "no" to skip the PATH-update prompt
#   ANTHROPIC_API_KEY    — if already exported, the prompt is skipped

set -eu

REPO="petlukk/Olorin"
INSTALL_DIR="${OLORIN_INSTALL_DIR:-$HOME/.local/bin}"
OLORIN_HOME="$HOME/.olorin"
ENV_FILE="$OLORIN_HOME/env"

err()  { printf 'error: %s\n' "$*" >&2; exit 1; }
info() { printf '  %s\n' "$*"; }
step() { printf '\n[%s] %s\n' "$1" "$2"; }

# stdin may be the curl pipe; read prompts from the controlling tty instead.
# In dash the redirect-setup error fires before `2>/dev/null` takes effect
# on `exec`, so the probe lives in a subshell where its stderr is contained.
# If that subshell can open /dev/tty, the real exec in the parent shell will
# also succeed (same controlling-terminal context).
PROMPT_FD=
if [ -t 0 ]; then
  PROMPT_FD=0
elif ( : </dev/tty ) >/dev/null 2>&1; then
  exec 3</dev/tty
  PROMPT_FD=3
fi

ask() {
  # ask "Prompt" "default" → echoes the answer
  prompt="$1"; default="$2"
  if [ -z "$PROMPT_FD" ]; then
    printf '%s\n' "$default"
    return
  fi
  printf '%s' "$prompt" >&2
  IFS= read -r ans <&"$PROMPT_FD" || ans=""
  if [ -z "$ans" ]; then
    printf '%s\n' "$default"
  else
    printf '%s\n' "$ans"
  fi
}

ask_yn() {
  prompt="$1"; default="$2"
  ans=$(ask "$prompt" "$default")
  case "$ans" in
    [Yy]|[Yy][Ee][Ss]) echo yes ;;
    *)                 echo no  ;;
  esac
}

# 1. Detect platform.
os=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)
case "$os" in
  linux) ;;
  *)     err "Unsupported OS: $os. Use install.ps1 for Windows or build from source." ;;
esac
case "$arch" in
  x86_64|amd64)         target="linux-x86_64" ;;
  aarch64|arm64)        target="linux-aarch64" ;;
  *)                    err "Unsupported architecture: $arch" ;;
esac
echo "Olorin installer"
info "Platform: $target"

# 2. Resolve release tag.
tag="${OLORIN_VERSION:-}"
if [ -z "$tag" ]; then
  step 1 "Resolving latest release"
  tag=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
        | grep -oE '"tag_name":\s*"[^"]+"' | head -1 | cut -d'"' -f4)
  [ -n "$tag" ] || err "Could not resolve latest release tag from GitHub."
fi
info "Version: $tag"

# 3. Download olorin binary.
base="https://github.com/$REPO/releases/download/$tag"
binary_name="olorin-$target"
step 2 "Downloading olorin"
mkdir -p "$INSTALL_DIR"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL --retry 3 -o "$tmp/olorin" "$base/$binary_name" \
  || err "Failed to download $base/$binary_name"

# 4. Verify checksum if SHA256SUMS is published with the release.
if curl -fsSL --retry 2 -o "$tmp/SHA256SUMS" "$base/SHA256SUMS" 2>/dev/null; then
  expected=$(grep "  $binary_name\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
  if [ -n "$expected" ]; then
    actual=$(sha256sum "$tmp/olorin" | cut -d' ' -f1)
    [ "$expected" = "$actual" ] || err "Checksum mismatch for $binary_name"
    info "Checksum verified"
  fi
fi

# 5. Install.
chmod +x "$tmp/olorin"
mv "$tmp/olorin" "$INSTALL_DIR/olorin"
info "Installed: $INSTALL_DIR/olorin"

# 6. Cloud fallback (Anthropic API key).
mkdir -p "$OLORIN_HOME"
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  info "ANTHROPIC_API_KEY already set in environment — skipping prompt."
else
  step 3 "Cloud fallback (optional)"
  echo "  Olorin runs a local Gemma 4 model by default. If you also want"
  echo "  Anthropic Claude as a cloud fallback (used when no local model"
  echo "  is loaded), enter your API key. Leave blank to skip."
  key=$(ask "  ANTHROPIC_API_KEY (blank to skip): " "")
  if [ -n "$key" ]; then
    if [ -f "$ENV_FILE" ] && grep -q '^ANTHROPIC_API_KEY=' "$ENV_FILE"; then
      info "ANTHROPIC_API_KEY already in $ENV_FILE — leaving it untouched."
    else
      printf 'ANTHROPIC_API_KEY=%s\n' "$key" >> "$ENV_FILE"
      chmod 600 "$ENV_FILE"
      info "Wrote $ENV_FILE (mode 0600). Olorin reads this at startup."
    fi
  fi
fi

# 7. Bridge (WhatsApp gateway, optional).
step 4 "WhatsApp gateway (optional)"
echo "  /teleport launches a WhatsApp bridge subprocess (~15 MB)."
echo "  Skip this if you only use the terminal REPL and web UI."
want_bridge="${OLORIN_WITH_BRIDGE:-}"
if [ -z "$want_bridge" ]; then
  want_bridge=$(ask_yn "  Install bridge? [y/N]: " "no")
fi
if [ "$want_bridge" = "yes" ]; then
  bridge_name="wa-bridge-$target"
  mkdir -p "$INSTALL_DIR"
  if curl -fsSL --retry 3 -o "$tmp/wa-bridge" "$base/$bridge_name" 2>/dev/null; then
    chmod +x "$tmp/wa-bridge"
    mv "$tmp/wa-bridge" "$INSTALL_DIR/wa-bridge"
    info "Installed: $INSTALL_DIR/wa-bridge"
  else
    info "Bridge binary not in release $tag — skipping. Build from source if needed."
  fi
fi

# 8. PATH.
case ":$PATH:" in
  *":$INSTALL_DIR:"*)
    info "PATH already includes $INSTALL_DIR." ;;
  *)
    step 5 "Shell PATH"
    want_path="${OLORIN_WITH_PATH:-}"
    if [ -z "$want_path" ]; then
      want_path=$(ask_yn "  Add $INSTALL_DIR to PATH via shell rc? [Y/n]: " "yes")
    fi
    if [ "$want_path" = "yes" ]; then
      line='export PATH="'$INSTALL_DIR':$PATH"'
      for rc in "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ -f "$rc" ] && ! grep -qF "$line" "$rc"; then
          printf '\n# Added by Olorin installer\n%s\n' "$line" >> "$rc"
          info "Updated $rc"
        fi
      done
      echo
      echo "  Restart your shell, or run:  export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
    ;;
esac

# 9. Quickstart.
echo
echo "Done. Try one of:"
echo "  olorin                 # terminal REPL"
echo "  olorin --serve         # web UI on http://127.0.0.1:8080"
echo "  olorin --strict        # deterministic dispatch only, no LLM (~25 ms)"
echo
echo "On first run you'll be prompted to set a vault passphrase."
echo "Docs: https://github.com/$REPO"
