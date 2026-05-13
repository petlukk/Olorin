#!/usr/bin/env bash
# Pi-side companion to tests/cross_arch_bit_identity.rs.
#
# The Rust integration test runs natively on the build host (x86 WSL).
# Cross-compiled Rust *test* binaries can't run on Pi 5 because the host
# glibc (2.39) is newer than the Pi's (2.36); but the *olorin* binary,
# linked against the Pi's glibc by the deploy workflow, runs fine. So we
# reproduce the same fixture matrix here by piping `/rune ... --json` to
# that binary and diffing against the goldens checked in alongside.
#
# Usage:  scripts/cross_arch_parity_pi.sh /path/to/olorin
# Exit:   0 = every case bit-identical to its x86 golden, 1 = drift detected.
#
# Run after cross-compiling and scp'ing olorin + the tests/fixtures/runes
# tree onto the Pi. Cases mirror the Rust test exactly minus eadiff
# (pure-Rust scalar, no SIMD path — Rust test covers it).

set -eu

OLORIN="${1:?usage: $0 /path/to/olorin}"
FIXTURES_DIR="${FIXTURES_DIR:-$(dirname "$0")/../tests/fixtures/runes}"
GOLDENS_DIR="${GOLDENS_DIR:-$FIXTURES_DIR/golden}"

[ -x "$OLORIN" ]            || { echo "not executable: $OLORIN" >&2; exit 2; }
[ -d "$FIXTURES_DIR" ]      || { echo "missing fixtures dir: $FIXTURES_DIR" >&2; exit 2; }
[ -d "$GOLDENS_DIR" ]       || { echo "missing goldens dir: $GOLDENS_DIR"   >&2; exit 2; }

# Match the Rust harness's two scrubbings exactly: scan_us → 0, path → "<fixture>".
normalize() {
    sed -E -e 's/"scan_us":[0-9]+/"scan_us":0/g' \
           -e 's/"path":"[^"]*"/"path":"<fixture>"/g'
}

# Single (rune, fixture, golden) case. Returns 0 on parity, 1 on diff.
run_case() {
    local case_name="$1"
    local rune="$2"
    local fixture="$3"

    local staged="/tmp/olorin_parity_$fixture"
    cp "$FIXTURES_DIR/$fixture" "$staged"

    # Match Rust harness: spawn `olorin --strict`, pipe /rune + /quit.
    local raw
    raw=$(printf '/rune %s --json %s\n/quit\n' "$rune" "$staged" \
        | "$OLORIN" --strict 2>/dev/null \
        | grep -m1 '^olorin> {"schema_version":' \
        | sed 's/^olorin> //')

    rm -f "$staged"

    if [ -z "$raw" ]; then
        echo "FAIL $case_name: no JSON line in stdout"
        return 1
    fi

    local actual
    actual=$(printf '%s' "$raw" | normalize)
    local golden_file="$GOLDENS_DIR/$case_name.json"

    if ! [ -f "$golden_file" ]; then
        echo "FAIL $case_name: missing golden $golden_file"
        return 1
    fi

    local expected
    expected=$(cat "$golden_file")

    if [ "$actual" = "$expected" ]; then
        echo "OK   $case_name"
        return 0
    fi

    echo "FAIL $case_name: bytes differ"
    echo "  golden:  $expected"
    echo "  actual:  $actual"
    return 1
}

fail_count=0
run_case eacrunch_tiny    eacrunch  tiny.csv         || fail_count=$((fail_count + 1))
run_case eajson_tiny      eajson    tiny.jsonl       || fail_count=$((fail_count + 1))
run_case eaparquet_tiny   eaparquet tiny.parquet     || fail_count=$((fail_count + 1))
run_case ealog_parity     ealog     parity_log.log   || fail_count=$((fail_count + 1))
run_case eatime_parity    eatime    parity_times.log || fail_count=$((fail_count + 1))

echo "---"
if [ "$fail_count" -eq 0 ]; then
    echo "cross-arch parity: 5/5 OK"
    exit 0
fi
echo "cross-arch parity: $fail_count case(s) drifted"
exit 1
