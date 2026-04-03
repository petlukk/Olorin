# Audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all Hard Rule violations, UB, security holes, and hot-path allocations found in the 2026-03-31 audit.

**Architecture:** Seven phases — split the 495-line file, fix UB/bugs, harden the server, fix data corruption bugs, eliminate hot-path allocations, deduplicate code, add missing tool tests. Each phase is independently committable and testable.

**Tech Stack:** Rust (no deps), SIMD kernels via FFI, libc, std::net

---

## Phase 1: Split `ffi_inference.rs` (Hard Rule — 500 lines)

### Task 1.1: Extract type aliases to `kernels/ffi_inference_types.rs`

**Files:**
- Create: `src/kernels/ffi_inference_types.rs`
- Modify: `src/kernels/ffi_inference.rs`
- Modify: `src/kernels/mod.rs` (if it exists, else `src/lib.rs`)

- [ ] **Step 1: Create `ffi_inference_types.rs` with all type aliases**

Move lines 9-89 from `ffi_inference.rs` (all `type XxxFn = unsafe extern "C" fn(...)` declarations) into the new file. Add `pub(crate)` visibility.

```rust
//! Type aliases for inference kernel FFI function pointers.

pub(crate) type I2DotI8Fn = unsafe extern "C" fn(*const u8, *const i8, i32) -> i32;
pub(crate) type I2DotI8_4RowFn = unsafe extern "C" fn(*const u8, *const i8, *mut i32, i32, i32);
// ... all remaining type aliases from lines 9-89
```

- [ ] **Step 2: Update `ffi_inference.rs` to import from the new file**

Replace the type alias block with:

```rust
use crate::kernels::ffi_inference_types::*;
```

- [ ] **Step 3: Add module declaration**

In `src/kernels/mod.rs` (or wherever kernel modules are declared), add:

```rust
pub(crate) mod ffi_inference_types;
```

- [ ] **Step 4: Verify build**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1`
Expected: Compiles with no new warnings.

- [ ] **Step 5: Verify line counts**

Run: `wc -l src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs`
Expected: `ffi_inference.rs` < 420 lines, `ffi_inference_types.rs` ~85 lines.

- [ ] **Step 6: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1`
Expected: All existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs src/kernels/mod.rs
git commit -m "refactor: split type aliases from ffi_inference.rs (495→~415 lines)"
```

---

## Phase 2: Fix Undefined Behavior

### Task 2.1: Fix `generate.rs` — writing through immutable pointer

**Files:**
- Modify: `src/inference/generate.rs:100-108`

- [ ] **Step 1: Make `generated` and `gen_tokens` mutable**

At their declaration sites (find with grep), change `let generated` → `let mut generated` and `let gen_tokens` → `let mut gen_tokens`. Then fix the wipe block:

```rust
// Wipe token buffers — no plaintext residue
unsafe {
    std::ptr::write_bytes(tokens.as_mut_ptr(), 0, tokens.len());
    std::ptr::write_bytes(generated.as_mut_ptr(), 0, generated.len());
    std::ptr::write_bytes(gen_tokens.as_mut_ptr(), 0, gen_tokens.len());
    let mut out = output.lock().unwrap();
    std::ptr::write_bytes(out.as_mut_ptr(), 0, out.len());
}
```

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/inference/generate.rs
git commit -m "fix: UB in generate.rs — use as_mut_ptr() for buffer wipe"
```

### Task 2.2: Fix `crypto.rs` — Vec<u8> alignment violation

**Files:**
- Modify: `src/storage/crypto.rs`

- [ ] **Step 1: Replace `Vec<u8>` with `Vec<i32>` for SIMD buffers**

Change the `chacha20_xor` function. Replace:

```rust
let input_copy: Vec<u8> = buf.to_vec();
let mut output: Vec<u8> = vec![0u8; buf.len()];
```

With properly aligned buffers:

```rust
// i32-aligned staging buffers for SIMD kernel (CLAUDE.md: "Use Vec<i32> not Vec<u8>")
let i32_len = (buf.len() + 3) / 4;
let mut input_i32: Vec<i32> = vec![0i32; i32_len];
let mut output_i32: Vec<i32> = vec![0i32; i32_len];
// Copy input bytes into aligned buffer
unsafe {
    std::ptr::copy_nonoverlapping(buf.as_ptr(), input_i32.as_mut_ptr() as *mut u8, buf.len());
}
```

Then update the FFI call to use `input_i32.as_ptr()` / `output_i32.as_mut_ptr()` and copy result back:

```rust
unsafe {
    ffi::chacha20_encrypt(
        key_i32.as_ptr(),
        nonce_i32.as_ptr(),
        counter,
        input_i32.as_ptr() as *const u8,
        output_i32.as_mut_ptr() as *mut u8,
        buf.len() as i32,
        scratch.as_mut_ptr(),
        scratch.as_mut_ptr() as *mut u8,
        input_i32.as_mut_ptr(),
        output_i32.as_mut_ptr(),
    );
    std::ptr::copy_nonoverlapping(output_i32.as_ptr() as *const u8, buf.as_mut_ptr(), buf.len());
}
```

- [ ] **Step 2: Build and run vault tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test vault 2>&1`
Expected: All vault/crypto tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/storage/crypto.rs
git commit -m "fix: alignment violation in crypto — use Vec<i32> for SIMD buffers"
```

---

## Phase 3: Server Hardening

### Task 3.1: Add body-size limit to `read_body`

**Files:**
- Modify: `src/interface/server.rs:162-174`

- [ ] **Step 1: Add max body size constant and check**

```rust
const MAX_BODY_SIZE: usize = 1024 * 1024; // 1 MB

pub(crate) fn read_body(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize) -> Vec<u8> {
    let content_len = parse_content_length(req);
    if content_len > MAX_BODY_SIZE {
        return Vec::new();
    }
    let header_end  = req.find("\r\n\r\n").unwrap_or(n) + 4;
    // ... rest unchanged
}
```

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test server 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/interface/server.rs
git commit -m "fix: add 1MB body-size limit to prevent OOM via Content-Length"
```

### Task 3.2: Add PTY session limit

**Files:**
- Modify: `src/interface/term_stream.rs:24-39`

- [ ] **Step 1: Add session count check in `handle_term_open`**

```rust
const MAX_TERM_SESSIONS: usize = 8;

pub fn handle_term_open(stream: &mut std::net::TcpStream) {
    {
        let sessions = term_sessions().lock().unwrap();
        if sessions.len() >= MAX_TERM_SESSIONS {
            serve_json(stream, r#"{"error":"too many sessions"}"#);
            return;
        }
    }
    let id = NEXT_TERM_ID.fetch_add(1, Ordering::Relaxed);
    // ... rest unchanged
}
```

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test pty 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/interface/term_stream.rs
git commit -m "fix: limit PTY sessions to 8 to prevent fork-bomb via API"
```

### Task 3.3: Bind to localhost by default

**Files:**
- Modify: `src/interface/server.rs` — find the `0.0.0.0` bind line (~line 33)

- [ ] **Step 1: Change default bind to 127.0.0.1**

Replace `0.0.0.0` with `127.0.0.1`. If the user needs remote access, they can use an env var or config option. Add a check:

```rust
let bind_addr = std::env::var("OLORIN_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
let addr = format!("{bind_addr}:{port}");
```

- [ ] **Step 2: Build and test server**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test server 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/interface/server.rs
git commit -m "fix: bind to 127.0.0.1 by default, OLORIN_BIND env for override"
```

---

## Phase 4: Fix Data Corruption Bugs

### Task 4.1: Fix `/teleport` command routing

**Files:**
- Modify: `src/core/dispatch.rs:46`

- [ ] **Step 1: Update CMD_TOOL_LAST to include CMD_TELEPORT**

```rust
pub const CMD_TOOL_LAST:  i32 = CMD_TELEPORT; // was CMD_REMIND
```

- [ ] **Step 2: Build and test dispatch**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test dispatch 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/core/dispatch.rs
git commit -m "fix: CMD_TOOL_LAST includes CMD_TELEPORT — /teleport was silently dropped"
```

### Task 4.2: Fix JSON parser UTF-8 corruption

**Files:**
- Modify: `src/storage/json.rs` — `parse_string` function (~line 233)

- [ ] **Step 1: Replace single-byte char push with UTF-8 accumulation**

Replace the catch-all arm:
```rust
Some(b) => s.push(b as char),
```

With proper UTF-8 decoding:

```rust
Some(b) if b < 0x80 => s.push(b as char),
Some(b) => {
    // UTF-8 multi-byte: accumulate bytes
    let (need, mut cp) = if b < 0xE0 {
        (1, (b & 0x1F) as u32)
    } else if b < 0xF0 {
        (2, (b & 0x0F) as u32)
    } else {
        (3, (b & 0x07) as u32)
    };
    for _ in 0..need {
        match self.advance() {
            Some(cont) if cont & 0xC0 == 0x80 => {
                cp = (cp << 6) | (cont & 0x3F) as u32;
            }
            _ => { s.push('\u{FFFD}'); break; }
        }
    }
    if let Some(c) = char::from_u32(cp) {
        s.push(c);
    } else {
        s.push('\u{FFFD}');
    }
}
```

- [ ] **Step 2: Write test in `tests/json_test.rs` or existing JSON test file**

Find the existing JSON test file and add:

```rust
#[test]
fn json_utf8_string() {
    // Swedish characters
    let json = r#"{"name":"Ölörin"}"#;
    let val = parse_json(json.as_bytes());
    assert_eq!(extract_string(&val, "name"), Some("Ölörin".to_string()));
}
```

- [ ] **Step 3: Build and run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test json 2>&1`

- [ ] **Step 4: Commit**

```bash
git add src/storage/json.rs tests/
git commit -m "fix: JSON parser handles multi-byte UTF-8 — was corrupting non-ASCII"
```

### Task 4.3: Fix `read_file.rs` UTF-8 boundary panic

**Files:**
- Modify: `src/tools/read_file.rs:14`

- [ ] **Step 1: Find char boundary before truncating**

Replace:
```rust
&content[..max_len]
```

With:
```rust
{
    let mut end = max_len;
    while end > 0 && !content.is_char_boundary(end) { end -= 1; }
    &content[..end]
}
```

- [ ] **Step 2: Build**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/tools/read_file.rs
git commit -m "fix: read_file truncation respects UTF-8 char boundaries"
```

### Task 4.4: Fix `exec.rs` CString NUL panic

**Files:**
- Modify: `src/interface/exec.rs:26`

- [ ] **Step 1: Replace `unwrap()` with error handling**

Find the `CString::new(*s).unwrap()` call. Replace with:

```rust
let c_str = match CString::new(*s) {
    Ok(c) => c,
    Err(_) => return Err(io::Error::new(io::ErrorKind::InvalidInput, "NUL byte in argument")),
};
```

Do the same for any other `CString::new().unwrap()` in the file (check line 179 too).

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/interface/exec.rs
git commit -m "fix: CString creation returns error instead of panicking on NUL bytes"
```

---

## Phase 5: Hot-Path Allocation Elimination

### Task 5.1: Pre-allocate sampling buffers on `InferenceState`

**Files:**
- Modify: `src/inference/forward.rs`

- [ ] **Step 1: Add sampling buffers to `InferenceState` struct**

Add fields after the existing Vec fields (~line 28):

```rust
pub(crate) sample_logits_buf: Vec<f32>,
pub(crate) sample_probs: Vec<f32>,
pub(crate) sample_indices: Vec<usize>,
```

- [ ] **Step 2: Initialize in `InferenceState::new`**

In the `new()` function, after `logits` init, add:

```rust
sample_logits_buf: vec![0.0f32; vocab_size],
sample_probs: vec![0.0f32; vocab_size],
sample_indices: (0..vocab_size).collect(),
```

- [ ] **Step 3: Refactor `sample` to take pre-allocated buffers**

Change `sample` signature to accept mutable slices instead of allocating:

```rust
pub(crate) fn sample_into(
    logits: &[f32],
    logits_buf: &mut [f32],
    probs: &mut [f32],
    indices: &mut [usize],
    temperature: f32,
    top_k: usize,
    top_p: f32,
) -> u32 {
    if temperature <= 0.0 { return argmax(logits); }
    let n = logits.len();
    logits_buf[..n].copy_from_slice(logits);
    apply_top_k(logits_buf, indices, k);  // indices reset inside
    // ... rest uses probs[..n] instead of allocating
}
```

Update `apply_top_k` to take `indices: &mut [usize]` and reset `for i in 0..n { indices[i] = i; }` at the start instead of allocating.

Update `apply_top_p` similarly.

- [ ] **Step 4: Update `sample_logits` to use struct buffers**

```rust
pub fn sample_logits(&mut self, temperature: f32, top_k: usize, top_p: f32) -> u32 {
    sample_into(
        &self.logits,
        &mut self.sample_logits_buf,
        &mut self.sample_probs,
        &mut self.sample_indices,
        temperature, top_k, top_p,
    )
}
```

Note: `sample_logits` changes from `&self` to `&mut self`. Update callers in `generate()`.

- [ ] **Step 5: Do the same for `LlamaState` in `forward_llama.rs`**

Add the same three fields, same initialization, same `sample_into` call.

- [ ] **Step 6: Build and run inference tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test inference 2>&1`

- [ ] **Step 7: Commit**

```bash
git add src/inference/forward.rs src/inference/forward_llama.rs
git commit -m "perf: pre-allocate sampling buffers — eliminate 3 allocs per token"
```

### Task 5.2: Pre-allocate KV-cache scratch buffers

**Files:**
- Modify: `src/inference/cache.rs`

- [ ] **Step 1: Add scratch fields to `EakvCache`**

Add after existing fields:

```rust
append_tmp: Vec<i32>,
append_rot_buf: Vec<f32>,
attn_rot_q: Vec<f32>,
```

- [ ] **Step 2: Initialize in `EakvCache::new`**

```rust
let max_head_elems = head_dim as usize;  // largest head
let max_n_gphead = (head_dim / 64) as usize;
// ...
append_tmp: vec![0i32; max_n_gphead * 32],
append_rot_buf: vec![0.0f32; max_head_elems],
attn_rot_q: vec![0.0f32; n_q_heads as usize * head_dim as usize],
```

- [ ] **Step 3: Update `append` to use struct scratch**

Replace lines 241-242:
```rust
let mut tmp = vec![0i32; n_gphead * 32];
let mut rot_buf = vec![0.0f32; head_elems];
```

With:
```rust
let tmp = &mut self.append_tmp[..n_gphead * 32];
let rot_buf = &mut self.append_rot_buf[..head_elems];
```

Zero them at start of call: `tmp.fill(0); rot_buf.fill(0.0);`

- [ ] **Step 4: Update `attention_scores` to use struct scratch**

Change `attention_scores` to take `&mut self` (or accept `attn_rot_q: &mut [f32]` from caller). Replace line 324:
```rust
let mut rot_q = queries[..q_elems].to_vec();
```

With:
```rust
cache.attn_rot_q[..q_elems].copy_from_slice(&queries[..q_elems]);
let rot_q = &mut cache.attn_rot_q[..q_elems];
```

Note: `attention_scores` is currently `fn(cache: &EakvCache, ...)`. It needs to become `fn(cache: &mut EakvCache, ...)`. Update all callers in `forward.rs` and `forward_llama.rs`.

- [ ] **Step 5: Build and run cache tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test cache 2>&1`

- [ ] **Step 6: Commit**

```bash
git add src/inference/cache.rs src/inference/forward.rs src/inference/forward_llama.rs
git commit -m "perf: pre-allocate KV-cache scratch — eliminate allocs per layer per token"
```

### Task 5.3: Pre-allocate crypto scratch in Vault

**Files:**
- Modify: `src/storage/vault.rs` (add scratch fields)
- Modify: `src/storage/crypto.rs` (accept scratch from caller)

- [ ] **Step 1: Add scratch buffers to Vault struct**

```rust
crypto_scratch: Vec<i32>,
crypto_input: Vec<i32>,
crypto_output: Vec<i32>,
```

Initialize with reasonable size (e.g., `BLOCK_SIZE / 4 + 1` i32 elements) in `Vault::new`/`Vault::open_existing`.

- [ ] **Step 2: Change `chacha20_xor` to accept scratch buffers**

```rust
pub fn chacha20_xor(
    buf: &mut [u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    counter: u32,
    scratch: &mut [i32],
    input_buf: &mut [i32],
    output_buf: &mut [i32],
)
```

- [ ] **Step 3: Update all call sites in vault.rs to pass scratch**

- [ ] **Step 4: Build and run vault tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test vault 2>&1`

- [ ] **Step 5: Commit**

```bash
git add src/storage/crypto.rs src/storage/vault.rs
git commit -m "perf: pre-allocate crypto scratch in Vault — eliminate 3 allocs per encrypt/decrypt"
```

### Task 5.4: Eliminate allocations in `shell_guard.rs`

**Files:**
- Modify: `src/core/shell_guard.rs`

- [ ] **Step 1: Replace format!() allocations with contains() chains**

Replace the loop at ~line 119:
```rust
if command.contains(&format!(">{target}")) || command.contains(&format!("> {target}"))
```

With a single-pass approach. Pre-build the check strings as `const` arrays or check inline:

```rust
const DANGEROUS_TARGETS: &[&str] = &["/dev/sda", "/dev/nvme", /* ... */];

// Single pass: scan for '>' then check what follows
fn targets_dangerous_write(command: &str) -> bool {
    for (i, _) in command.match_indices('>') {
        let rest = command[i + 1..].trim_start();
        for &target in DANGEROUS_TARGETS {
            if rest.starts_with(target) { return true; }
        }
    }
    false
}
```

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test shell 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/core/shell_guard.rs
git commit -m "perf: eliminate format! allocations in shell_guard classify"
```

---

## Phase 6: Code Deduplication

### Task 6.1: Extract shared `softmax_rows` and `wipe_*` into `inference/math.rs`

**Files:**
- Create: `src/inference/math.rs`
- Modify: `src/inference/forward.rs`
- Modify: `src/inference/forward_llama.rs`

- [ ] **Step 1: Create `inference/math.rs` with shared functions**

```rust
//! Shared math utilities for inference forward passes.

pub(crate) fn softmax_rows(data: &mut [f32], n_rows: usize, seq_len: usize) {
    for r in 0..n_rows {
        let row = &mut data[r * seq_len..(r + 1) * seq_len];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}

pub(crate) fn wipe_f32(buf: &mut [f32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn wipe_i8(buf: &mut [i8]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}
```

- [ ] **Step 2: Add module declaration**

In `src/inference/mod.rs`:
```rust
pub(crate) mod math;
```

- [ ] **Step 3: Replace duplicates in `forward.rs`**

Delete the `softmax_rows`, `wipe_f32` functions. Add:
```rust
use crate::inference::math::{softmax_rows, wipe_f32};
```

- [ ] **Step 4: Replace duplicates in `forward_llama.rs`**

Delete `softmax_rows`, `wipe_f32`, `wipe_i8`. Add:
```rust
use crate::inference::math::{softmax_rows, wipe_f32, wipe_i8};
```

- [ ] **Step 5: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test inference 2>&1`

- [ ] **Step 6: Commit**

```bash
git add src/inference/math.rs src/inference/mod.rs src/inference/forward.rs src/inference/forward_llama.rs
git commit -m "refactor: extract shared softmax_rows and wipe_* into inference/math.rs"
```

### Task 6.2: Deduplicate `f16_to_f32`

**Files:**
- Modify: `src/inference/engine.rs` — remove inline closure version
- Keep: `src/inference/matmul.rs` — canonical `f16_to_f32` stays here

- [ ] **Step 1: In `engine.rs`, replace inline f16 conversion with import**

Find the closure/inline function. Replace with:
```rust
use crate::inference::matmul::f16_to_f32;
```

Make sure `f16_to_f32` in `matmul.rs` is `pub(crate)`.

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/inference/engine.rs src/inference/matmul.rs
git commit -m "refactor: deduplicate f16_to_f32 — use matmul.rs version everywhere"
```

### Task 6.3: Deduplicate cloud fallback msg_pairs in `router.rs`

**Files:**
- Modify: `src/core/router.rs`

- [ ] **Step 1: Extract msg_pairs construction into a helper method**

Find the duplicated message pair construction at ~line 247 and ~line 437. Extract to:

```rust
fn build_cloud_messages(&self, recall_context: Option<&str>) -> Vec<(String, String)> {
    let sys = if let Some(ctx) = recall_context {
        format!("{}\n\n{ctx}", self.system_prompt)
    } else {
        self.system_prompt.clone()
    };
    let mut pairs = vec![("system".to_string(), sys)];
    for msg in &self.history {
        pairs.push((msg.role.clone(), msg.text.clone()));
    }
    pairs
}
```

Replace both duplicated blocks with calls to this method.

- [ ] **Step 2: Build and test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test router 2>&1`

- [ ] **Step 3: Commit**

```bash
git add src/core/router.rs
git commit -m "refactor: extract build_cloud_messages — deduplicate msg_pairs construction"
```

---

## Phase 7: Missing Tool Tests

### Task 7.1: Add tool tests — batch 1 (file/shell tools)

**Files:**
- Modify: `tests/tools.rs` (or create new test file if preferred)

- [ ] **Step 1: Write tests for read_file, write_file, grep, shell**

```rust
#[test]
fn tool_read_file() {
    let tmp = std::env::temp_dir().join("olorin_test_read.txt");
    std::fs::write(&tmp, "hello olorin").unwrap();
    let result = execute_tool("read_file", &tmp.to_string_lossy());
    assert!(result.success);
    assert!(result.output.contains("hello olorin"));
    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn tool_write_file() {
    let tmp = std::env::temp_dir().join("olorin_test_write.txt");
    let arg = format!("{} test content", tmp.display());
    let result = execute_tool("write_file", &arg);
    assert!(result.success);
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert_eq!(content, "test content");
    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn tool_grep() {
    let tmp = std::env::temp_dir().join("olorin_test_grep.txt");
    std::fs::write(&tmp, "line one\nfind me\nline three").unwrap();
    let arg = format!("find me {}", tmp.display());
    let result = execute_tool("grep", &arg);
    assert!(result.success);
    assert!(result.output.contains("find me"));
    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn tool_shell() {
    let result = execute_tool("shell", "echo hello_olorin");
    assert!(result.success);
    assert!(result.output.contains("hello_olorin"));
}
```

- [ ] **Step 2: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test tool_ 2>&1`

- [ ] **Step 3: Commit**

```bash
git add tests/tools.rs
git commit -m "test: add e2e tests for read_file, write_file, grep, shell tools"
```

### Task 7.2: Add tool tests — batch 2 (info tools)

**Files:**
- Modify: `tests/tools.rs`

- [ ] **Step 1: Write tests for cpu, bench, tokens, json_tool, memory**

```rust
#[test]
fn tool_cpu() {
    let result = execute_tool("cpu", "");
    assert!(result.success);
    // Should contain CPU info
    assert!(result.output.len() > 10);
}

#[test]
fn tool_bench() {
    let result = execute_tool("bench", "");
    assert!(result.success);
}

#[test]
fn tool_tokens() {
    let result = execute_tool("tokens", "hello world");
    assert!(result.success);
    // Should output token count
    assert!(result.output.contains("token"));
}

#[test]
fn tool_json_tool() {
    let result = execute_tool("json", r#"{"a":1,"b":2}"#);
    assert!(result.success);
}

#[test]
fn tool_memory() {
    let result = execute_tool("memory", "");
    assert!(result.success);
}
```

- [ ] **Step 2: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test tool_ 2>&1`

- [ ] **Step 3: Commit**

```bash
git add tests/tools.rs
git commit -m "test: add e2e tests for cpu, bench, tokens, json, memory tools"
```

### Task 7.3: Add tool tests — batch 3 (network/misc tools)

**Files:**
- Modify: `tests/tools.rs`

- [ ] **Step 1: Write tests for remaining tools**

```rust
#[test]
fn tool_time() {
    let result = execute_tool("time", "");
    assert!(result.success);
}

#[test]
fn tool_define() {
    let result = execute_tool("define", "hello");
    // May fail without network, but should not panic
    assert!(!result.output.is_empty());
}

#[test]
fn tool_remind() {
    let result = execute_tool("remind", "5s test reminder");
    assert!(result.success);
}

#[test]
fn tool_git() {
    let result = execute_tool("git", "status");
    // Should work in the repo directory
    assert!(result.success || result.output.contains("fatal"));
}

#[test]
fn tool_http() {
    // Test with invalid URL — should fail gracefully, not panic
    let result = execute_tool("http", "http://localhost:1");
    assert!(!result.output.is_empty());
}

#[test]
fn tool_summarize() {
    let result = execute_tool("summarize", "This is a long text that should be summarized.");
    assert!(!result.output.is_empty());
}

#[test]
fn tool_translate() {
    let result = execute_tool("translate", "hello to swedish");
    assert!(!result.output.is_empty());
}

#[test]
fn tool_weather() {
    let result = execute_tool("weather", "Stockholm");
    assert!(!result.output.is_empty());
}
```

- [ ] **Step 2: Run all tool tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test tool_ 2>&1`

- [ ] **Step 3: Commit**

```bash
git add tests/tools.rs
git commit -m "test: add e2e tests for remaining tools — all 19 now covered"
```

---

## Verification

After all phases:

- [ ] **Final line count check**: `wc -l src/**/*.rs | sort -n | tail -20` — no file > 500
- [ ] **Full test suite**: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1`
- [ ] **Release build clean**: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1`
