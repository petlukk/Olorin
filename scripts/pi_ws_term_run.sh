#!/usr/bin/env bash
# Boots olorin --serve in an isolated HOME (throwaway vault, live vault untouched),
# runs the headless WS term test, then tears everything down. Run ON the Pi.
set -u
BIN="$HOME/olorin-term-test"
MODEL="$HOME/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf"
TESTHOME="$HOME/olorin-ws-testhome"
LOG="$HOME/olorin-ws-test.log"
PORT=8741

cleanup() {
  [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null
  sleep 1
  [ -n "${SRV_PID:-}" ] && kill -9 "$SRV_PID" 2>/dev/null
  rm -rf "$TESTHOME"
}
trap cleanup EXIT

rm -rf "$TESTHOME"; mkdir -p "$TESTHOME/.olorin/models"
# Symlink the real model into the isolated home so --model still resolves.
ln -sf "$MODEL" "$TESTHOME/.olorin/models/$(basename "$MODEL")"

HOME="$TESTHOME" OLORIN_PASSPHRASE="ws-test-throwaway" OLORIN_BIND=127.0.0.1 \
  "$BIN" --serve --strict --port "$PORT" --model "$MODEL" >"$LOG" 2>&1 &
SRV_PID=$!

# Wait up to 25s for the web UI to come up.
for i in $(seq 1 50); do
  if grep -q "Web UI at" "$LOG" 2>/dev/null; then break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "SERVER DIED during boot:"; cat "$LOG"; exit 1
  fi
  sleep 0.5
done
if ! grep -q "Web UI at" "$LOG"; then
  echo "SERVER did not announce Web UI in time:"; cat "$LOG"; exit 1
fi
echo "  server up: $(grep 'Web UI at' "$LOG")  search=$(grep -o 'search=[a-z0-9]*' "$LOG" | head -1)"

python3 "$HOME/pi_ws_term_test.py"
RC=$?
echo "  ws test rc=$RC"
exit $RC
