# eakv C→Rust Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eakv's C orchestration layer with pure Rust, loading Ea kernels via libloading.

**Architecture:** Four Rust modules (kernels, cache, attention, io) replace five C files. Kernels loaded dynamically from `~/.olorin/lib/` via libloading, same pattern as olorin-core's ffi.rs. The flat-buffer memory layout and Q4 split-pack format are preserved exactly.

**Tech Stack:** Rust, libloading, Ea SIMD kernels (.so), std::io for binary format

---

### File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `crates/eakv/src/kernels.rs` | Load all eakv .so files via libloading, expose typed function pointers |
| Create | `crates/eakv/src/cache.rs` | EakvCache struct, allocation, quantize, append, checkpoint/restore |
| Create | `crates/eakv/src/attention.rs` | attention_scores and attention_output with 4-way kernel dispatch |
| Create | `crates/eakv/src/io.rs` | Binary .eakv save/load, backwards compatible |
| Rewrite | `crates/eakv/src/lib.rs` | Public API re-exports, error types |
| Rewrite | `crates/eakv/build.rs` | Remove cc, just rerun-if-changed |
| Rewrite | `crates/eakv/Cargo.toml` | Remove cc build-dep, remove libc dep |
| Delete | `crates/eakv/csrc/*` | All C files and headers |

---

### Task 1: Kernel loading module

**Files:**
- Create: `crates/eakv/src/kernels.rs`
- Modify: `crates/eakv/src/lib.rs`
- Modify: `crates/eakv/Cargo.toml`
- Modify: `crates/eakv/build.rs`

- [ ] **Step 1: Strip build.rs and Cargo.toml**

Remove cc build-dep and C compilation. New `build.rs`:

```rust
fn main() {
    println!("cargo:rerun-if-changed=src/");
}
```

New `Cargo.toml`:

```toml
[package]
name = "eakv"
version.workspace = true
edition.workspace = true

[dependencies]
libloading.workspace = true
thiserror.workspace = true
```

- [ ] **Step 2: Write kernels.rs with type aliases and KernelTable**

```rust
//! Dynamic loader for eakv Ea SIMD kernels.

use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};

// Kernel function signatures (match Ea export signatures)
type QuantizeFn = unsafe extern "C" fn(
    *const f32, *mut i32, *mut f32, *mut f32, i32,
);
type RotateFn = unsafe extern "C" fn(*mut f32, *const f32, i32);
type FwhtFn = unsafe extern "C" fn(*mut f32, i32);
type SignFlipFn = unsafe extern "C" fn(*mut f32, *const f32, i32);
type KScoreMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32,
);
type KScoreGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32, i32,
);
type VSumMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32,
);
type VSumGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32, i32,
);

pub struct KernelTable {
    _libs: Vec<Library>,
    pub quantize: QuantizeFn,
    pub turbo_rotate: RotateFn,
    pub fwht_inplace: FwhtFn,
    pub sign_flip: SignFlipFn,
    pub k_score_mha: KScoreMhaFn,
    pub k_score_mha_64: KScoreMhaFn,
    pub k_score_gqa: KScoreGqaFn,
    pub k_score_gqa_64: KScoreGqaFn,
    pub v_sum_mha: VSumMhaFn,
    pub v_sum_mha_64: VSumMhaFn,
    pub v_sum_gqa: VSumGqaFn,
    pub v_sum_gqa_64: VSumGqaFn,
}

unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

/// Find the most recent `~/.olorin/lib/` directory containing eakv kernels.
pub fn find_kernel_dir() -> Result<PathBuf, String> {
    let home = home::home_dir()
        .ok_or_else(|| "cannot determine home directory".to_string())?;
    let lib_base = home.join(".olorin/lib");
    if !lib_base.is_dir() {
        return Err(format!("{} does not exist", lib_base.display()));
    }
    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;
    for entry in std::fs::read_dir(&lib_base)
        .map_err(|e| format!("cannot read {}: {e}", lib_base.display()))?
    {
        let entry = entry.map_err(|e| e.to_string())?;
        let so = entry.path().join("libquantize_simd.so");
        if so.is_file() {
            let mtime = so.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            if best.as_ref().map_or(true, |(_, t)| mtime > *t) {
                best = Some((entry.path(), mtime));
            }
        }
    }
    best.map(|(p, _)| p)
        .ok_or_else(|| "no eakv kernels found in ~/.olorin/lib/".to_string())
}

pub fn load(lib_dir: &Path) -> Result<KernelTable, String> {
    let open = |name: &str| -> Result<Library, String> {
        let path = lib_dir.join(format!("lib{name}.so"));
        unsafe {
            Library::new(&path)
                .map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    let sym = |lib: &Library, name: &str| -> Result<*const (), String> {
        unsafe {
            let s: Symbol<*const ()> = lib.get(name.as_bytes())
                .map_err(|e| format!("symbol {name}: {e}"))?;
            Ok(*s)
        }
    };

    let quantize_lib = open("quantize_simd")?;
    let turbo_lib = open("turbo_rotate")?;
    let k_score_lib = open("fused_k_score")?;
    let k_score_64_lib = open("fused_k_score_64")?;
    let k_score_gqa_lib = open("fused_k_score_gqa")?;
    let k_score_gqa_64_lib = open("fused_k_score_gqa_64")?;
    let v_sum_lib = open("fused_v_sum")?;
    let v_sum_64_lib = open("fused_v_sum_64")?;

    let table = KernelTable {
        quantize: unsafe { std::mem::transmute(sym(&quantize_lib, "q4_quantize_split_f32\0")?) },
        turbo_rotate: unsafe { std::mem::transmute(sym(&turbo_lib, "turbo_rotate\0")?) },
        fwht_inplace: unsafe { std::mem::transmute(sym(&turbo_lib, "fwht_inplace\0")?) },
        sign_flip: unsafe { std::mem::transmute(sym(&turbo_lib, "sign_flip\0")?) },
        k_score_mha: unsafe { std::mem::transmute(sym(&k_score_lib, "q4_fused_k_score_multi_f32\0")?) },
        k_score_mha_64: unsafe { std::mem::transmute(sym(&k_score_64_lib, "q4_fused_k_score_multi_64_f32\0")?) },
        k_score_gqa: unsafe { std::mem::transmute(sym(&k_score_gqa_lib, "q4_k_score_gqa_f32\0")?) },
        k_score_gqa_64: unsafe { std::mem::transmute(sym(&k_score_gqa_64_lib, "q4_k_score_gqa_64_f32\0")?) },
        v_sum_mha: unsafe { std::mem::transmute(sym(&v_sum_lib, "q4_fused_v_sum_multi_f32\0")?) },
        v_sum_mha_64: unsafe { std::mem::transmute(sym(&v_sum_64_lib, "q4_fused_v_sum_multi_64_f32\0")?) },
        v_sum_gqa: unsafe { std::mem::transmute(sym(&k_score_gqa_lib, "q4_v_sum_gqa_f32\0")?) },
        v_sum_gqa_64: unsafe { std::mem::transmute(sym(&k_score_gqa_64_lib, "q4_v_sum_gqa_64_f32\0")?) },
        _libs: vec![
            quantize_lib, turbo_lib,
            k_score_lib, k_score_64_lib,
            k_score_gqa_lib, k_score_gqa_64_lib,
            v_sum_lib, v_sum_64_lib,
        ],
    };
    Ok(table)
}
```

Note: `turbo_rotate.so` contains `turbo_rotate`, `fwht_inplace`, and `sign_flip` symbols. The GQA libs contain both K-score and V-sum symbols (`q4_k_score_gqa_f32` + `q4_v_sum_gqa_f32` in same .so).

- [ ] **Step 3: Write minimal lib.rs that declares modules**

```rust
//! eakv — Q4 KV cache quantization for LLM inference.

pub mod kernels;

pub use kernels::KernelTable;
```

- [ ] **Step 4: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build -p eakv 2>&1`
Expected: compiles with no errors

- [ ] **Step 5: Write kernel loading test**

Add to bottom of `kernels.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_and_load_kernels() {
        let dir = find_kernel_dir().expect("kernel dir not found");
        let table = load(&dir).expect("kernel load failed");
        // Verify function pointers are non-null
        assert!(table.quantize as usize != 0);
        assert!(table.turbo_rotate as usize != 0);
        assert!(table.k_score_mha as usize != 0);
        assert!(table.v_sum_mha as usize != 0);
    }
}
```

- [ ] **Step 6: Run test**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/eakv/src/kernels.rs crates/eakv/src/lib.rs crates/eakv/Cargo.toml crates/eakv/build.rs
git commit -m "feat(eakv): kernel loading module via libloading"
```

---

### Task 2: Cache module — struct, create, checkpoint, restore

**Files:**
- Create: `crates/eakv/src/cache.rs`
- Modify: `crates/eakv/src/lib.rs`

- [ ] **Step 1: Write failing test for cache create + checkpoint + restore**

Add to `cache.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;

    fn test_cache(n_layers: i32, n_kv_heads: i32, head_dim: i32, max_seq: i32) -> EakvCache {
        let dir = kernels::find_kernel_dir().expect("kernel dir");
        let kt = kernels::load(&dir).expect("kernel load");
        EakvCache::new(n_layers, n_kv_heads, head_dim, max_seq, kt).expect("create")
    }

    #[test]
    fn test_create_and_checkpoint() {
        let mut cache = test_cache(2, 4, 64, 128);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.checkpoint(), 0);
        assert_eq!(cache.n_layers(), 2);
        assert_eq!(cache.n_heads(), 4);
        assert_eq!(cache.head_dim(), 64);
        assert_eq!(cache.max_seq_len(), 128);
    }

    #[test]
    fn test_restore() {
        let mut cache = test_cache(2, 4, 64, 128);
        cache.restore(0).unwrap();
        assert!(cache.restore(1).is_err());
    }

    #[test]
    fn test_invalid_params() {
        let dir = kernels::find_kernel_dir().unwrap();
        let kt = kernels::load(&dir).unwrap();
        // head_dim not divisible by 64
        assert!(EakvCache::new(2, 4, 63, 128, kt).is_none());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: FAIL — `EakvCache` not defined

- [ ] **Step 3: Write cache.rs implementation — struct and lifecycle**

```rust
//! Q4 KV cache — allocation, quantization, checkpoint/restore.

use crate::kernels::KernelTable;
use std::sync::Arc;

/// Per-layer, per-KV (K or V) slice info pointing into the flat data buffer.
struct KvSlice {
    weights_offset: usize,
    scales_offset: usize,
    biases_offset: usize,
}

pub struct EakvCache {
    n_layers: i32,
    n_kv_heads: i32,
    head_dim: i32,
    max_seq_len: i32,
    seq_len: i32,
    groups_per_token: i32,
    max_groups: i32,
    data_buf: Vec<u8>,
    kv: Vec<KvSlice>,
    jl_signs: [f32; 64],
    kernels: Arc<KernelTable>,
}

fn gen_jl_signs() -> [f32; 64] {
    let mut signs = [0.0f32; 64];
    let mut rng: u64 = 0x4F6C6F72696E4A4C; // "OlorinJL"
    for s in &mut signs {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *s = if rng % 2 == 0 { 1.0 } else { -1.0 };
    }
    signs
}

/// Bytes per group: 32 (weights) + 4 (scale) + 4 (bias) = 40
fn block_size(max_groups: i32) -> usize {
    max_groups as usize * (32 + 4 + 4)
}

impl EakvCache {
    pub fn new(
        n_layers: i32,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        kernels: KernelTable,
    ) -> Option<Self> {
        if n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0 {
            return None;
        }
        if (n_kv_heads * head_dim) % 64 != 0 {
            return None;
        }

        let groups_per_token = (n_kv_heads * head_dim) / 64;
        let max_groups = groups_per_token * max_seq_len;
        let blk = block_size(max_groups);
        let n_slots = n_layers as usize * 2;
        let total = blk * n_slots;

        let data_buf = vec![0u8; total];
        let mut kv = Vec::with_capacity(n_slots);

        let mg = max_groups as usize;
        for i in 0..n_slots {
            let base = i * blk;
            kv.push(KvSlice {
                weights_offset: base,
                scales_offset: base + mg * 32,
                biases_offset: base + mg * 32 + mg * 4,
            });
        }

        Some(Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            groups_per_token,
            max_groups,
            data_buf,
            kv,
            jl_signs: gen_jl_signs(),
            kernels: Arc::new(kernels),
        })
    }

    pub fn seq_len(&self) -> i32 { self.seq_len }
    pub fn n_layers(&self) -> i32 { self.n_layers }
    pub fn n_heads(&self) -> i32 { self.n_kv_heads }
    pub fn head_dim(&self) -> i32 { self.head_dim }
    pub fn max_seq_len(&self) -> i32 { self.max_seq_len }

    pub fn checkpoint(&self) -> i32 { self.seq_len }

    pub fn restore(&mut self, seq_len: i32) -> Result<(), String> {
        if seq_len < 0 || seq_len > self.seq_len {
            return Err(format!("invalid seq_len {seq_len} (current {})", self.seq_len));
        }
        self.seq_len = seq_len;
        Ok(())
    }

    pub fn advance(&mut self, n_tokens: i32) -> Result<(), String> {
        if n_tokens <= 0 {
            return Err("n_tokens must be positive".to_string());
        }
        if self.seq_len + n_tokens > self.max_seq_len {
            return Err("would exceed max_seq_len".to_string());
        }
        self.seq_len += n_tokens;
        Ok(())
    }

    pub fn clear(&mut self) { self.seq_len = 0; }

    pub fn compression_ratio(&self) -> f32 {
        if self.seq_len == 0 { 0.0 } else { 40.0 / 256.0 }
    }

    // -- internal helpers for attention/io access --

    pub(crate) fn weights(&self, layer: i32, kv_idx: i32) -> &[u8] {
        let s = &self.kv[(layer * 2 + kv_idx) as usize];
        let mg = self.max_groups as usize;
        &self.data_buf[s.weights_offset..s.weights_offset + mg * 32]
    }

    pub(crate) fn scales(&self, layer: i32, kv_idx: i32) -> &[f32] {
        let s = &self.kv[(layer * 2 + kv_idx) as usize];
        let mg = self.max_groups as usize;
        let bytes = &self.data_buf[s.scales_offset..s.scales_offset + mg * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, mg) }
    }

    pub(crate) fn biases(&self, layer: i32, kv_idx: i32) -> &[f32] {
        let s = &self.kv[(layer * 2 + kv_idx) as usize];
        let mg = self.max_groups as usize;
        let bytes = &self.data_buf[s.biases_offset..s.biases_offset + mg * 4];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, mg) }
    }

    fn weights_mut(&mut self, layer: i32, kv_idx: i32) -> &mut [u8] {
        let s = &self.kv[(layer * 2 + kv_idx) as usize];
        let mg = self.max_groups as usize;
        let off = s.weights_offset;
        &mut self.data_buf[off..off + mg * 32]
    }

    fn scales_mut(&mut self, layer: i32, kv_idx: i32) -> &mut [f32] {
        let s = &self.kv[(layer * 2 + kv_idx) as usize];
        let mg = self.max_groups as usize;
        let off = s.scales_offset;
        let bytes = &mut self.data_buf[off..off + mg * 4];
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, mg) }
    }

    fn biases_mut(&mut self, layer: i32, kv_idx: i32) -> &mut [f32] {
        let s = &self.kv[(layer * 2 + kv_idx) as usize];
        let mg = self.max_groups as usize;
        let off = s.biases_offset;
        let bytes = &mut self.data_buf[off..off + mg * 4];
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, mg) }
    }

    pub(crate) fn jl_signs(&self) -> &[f32; 64] { &self.jl_signs }
    pub(crate) fn kernels(&self) -> &KernelTable { &self.kernels }
    pub(crate) fn groups_per_token(&self) -> i32 { self.groups_per_token }

    pub(crate) fn data_buf(&self) -> &[u8] { &self.data_buf }
    pub(crate) fn data_buf_mut(&mut self) -> &mut [u8] { &mut self.data_buf }
    pub(crate) fn kv_slice(&self, idx: usize) -> (usize, usize, usize) {
        let s = &self.kv[idx];
        (s.weights_offset, s.scales_offset, s.biases_offset)
    }
    pub(crate) fn set_seq_len(&mut self, s: i32) { self.seq_len = s; }
}

unsafe impl Send for EakvCache {}
```

- [ ] **Step 4: Update lib.rs**

```rust
//! eakv — Q4 KV cache quantization for LLM inference.

pub mod kernels;
pub mod cache;

pub use cache::EakvCache;
pub use kernels::KernelTable;
```

- [ ] **Step 5: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/eakv/src/cache.rs crates/eakv/src/lib.rs
git commit -m "feat(eakv): cache module — struct, create, checkpoint, restore"
```

---

### Task 3: Cache quantize — load_raw and append

**Files:**
- Modify: `crates/eakv/src/cache.rs`

- [ ] **Step 1: Write failing test for load_raw roundtrip**

Add to `cache.rs` tests:

```rust
    #[test]
    fn test_load_raw_sets_seq_len() {
        let mut cache = test_cache(1, 1, 64, 16);
        // 1 layer, 1 head, 64 dim, 16 max_seq
        // data: [layer=0][kv=0,1] × [head=0] × [pos=0..3] × [dim=0..63]
        let seq_len = 4;
        let elems = 1 * 2 * 1 * seq_len * 64; // n_layers * 2 * n_heads * seq * dim
        let data: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01).collect();
        cache.load_raw(&data, seq_len as i32).unwrap();
        assert_eq!(cache.seq_len(), seq_len as i32);
    }

    #[test]
    fn test_append_and_advance() {
        let mut cache = test_cache(1, 1, 64, 16);
        // append 2 tokens for layer 0, K (kv_idx=0)
        let data: Vec<f32> = (0..2 * 64).map(|i| i as f32 * 0.1).collect();
        cache.append(&data, 0, 0, 2).unwrap();
        // append 2 tokens for layer 0, V (kv_idx=1)
        cache.append(&data, 0, 1, 2).unwrap();
        cache.advance(2).unwrap();
        assert_eq!(cache.seq_len(), 2);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: FAIL — `load_raw` and `append` not defined

- [ ] **Step 3: Implement rotate_groups, inverse_rotate_groups, load_raw, append**

Add to `cache.rs` impl block:

```rust
    fn rotate_groups(&self, buf: &mut [f32], n_groups: i32) {
        let rotate = self.kernels.turbo_rotate;
        for g in 0..n_groups as usize {
            unsafe { rotate(buf[g * 64..].as_mut_ptr(), self.jl_signs.as_ptr(), 64) };
        }
    }

    fn inverse_rotate_groups(&self, buf: &mut [f32], n_groups: i32) {
        let fwht = self.kernels.fwht_inplace;
        let flip = self.kernels.sign_flip;
        for g in 0..n_groups as usize {
            let ptr = buf[g * 64..].as_mut_ptr();
            unsafe {
                fwht(ptr, 64);
                flip(ptr, self.jl_signs.as_ptr(), 64);
            }
        }
    }

    pub fn load_raw(&mut self, data: &[f32], seq_len: i32) -> Result<(), String> {
        if seq_len <= 0 || seq_len > self.max_seq_len {
            return Err(format!("invalid seq_len {seq_len}"));
        }

        let hd = self.head_dim as usize;
        let nh = self.n_kv_heads as usize;
        let gpd = hd / 64;
        let gph = self.max_seq_len as usize * gpd;
        let n_groups_per_head = seq_len as usize * gpd;
        let head_elems = seq_len as usize * hd;
        let elems_per_lkv = nh * head_elems;

        let mut tmp = vec![0i32; n_groups_per_head * 32];
        let mut rot_buf = vec![0.0f32; head_elems];

        for l in 0..self.n_layers {
            for kv in 0..2i32 {
                let lkv_offset = (l * 2 + kv) as usize * elems_per_lkv;
                for h in 0..nh {
                    let src = &data[lkv_offset + h * head_elems..];
                    let group_base = h * gph;

                    rot_buf[..head_elems].copy_from_slice(&src[..head_elems]);
                    self.rotate_groups(&mut rot_buf, n_groups_per_head as i32);

                    let scales = self.scales_mut(l, kv);
                    let biases = self.biases_mut(l, kv);
                    unsafe {
                        (self.kernels.quantize)(
                            rot_buf.as_ptr(),
                            tmp.as_mut_ptr(),
                            scales[group_base..].as_mut_ptr(),
                            biases[group_base..].as_mut_ptr(),
                            n_groups_per_head as i32,
                        );
                    }

                    let weights = self.weights_mut(l, kv);
                    for i in 0..n_groups_per_head * 32 {
                        weights[group_base * 32 + i] = tmp[i] as u8;
                    }
                }
            }
        }

        self.seq_len = seq_len;
        Ok(())
    }

    pub fn append(
        &mut self,
        data: &[f32],
        layer: i32,
        kv_idx: i32,
        n_tokens: i32,
    ) -> Result<(), String> {
        if n_tokens <= 0 { return Err("n_tokens must be positive".to_string()); }
        if layer < 0 || layer >= self.n_layers { return Err("invalid layer".to_string()); }
        if kv_idx < 0 || kv_idx > 1 { return Err("kv_idx must be 0 or 1".to_string()); }
        if self.seq_len + n_tokens > self.max_seq_len {
            return Err("would exceed max_seq_len".to_string());
        }

        let hd = self.head_dim as usize;
        let nh = self.n_kv_heads as usize;
        let gpd = hd / 64;
        let gph = self.max_seq_len as usize * gpd;
        let n_groups_per_head = n_tokens as usize * gpd;
        let head_elems = n_tokens as usize * hd;

        let mut tmp = vec![0i32; n_groups_per_head * 32];
        let mut rot_buf = vec![0.0f32; head_elems];

        for h in 0..nh {
            let src = &data[h * head_elems..];
            let group_base = h * gph + self.seq_len as usize * gpd;

            rot_buf[..head_elems].copy_from_slice(&src[..head_elems]);
            self.rotate_groups(&mut rot_buf, n_groups_per_head as i32);

            let scales = self.scales_mut(layer, kv_idx);
            let biases = self.biases_mut(layer, kv_idx);
            unsafe {
                (self.kernels.quantize)(
                    rot_buf.as_ptr(),
                    tmp.as_mut_ptr(),
                    scales[group_base..].as_mut_ptr(),
                    biases[group_base..].as_mut_ptr(),
                    n_groups_per_head as i32,
                );
            }

            let weights = self.weights_mut(layer, kv_idx);
            for i in 0..n_groups_per_head * 32 {
                weights[group_base * 32 + i] = tmp[i] as u8;
            }
        }

        Ok(())
    }
```

- [ ] **Step 4: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/eakv/src/cache.rs
git commit -m "feat(eakv): load_raw and append with TurboQuant rotation"
```

---

### Task 4: Attention module

**Files:**
- Create: `crates/eakv/src/attention.rs`
- Modify: `crates/eakv/src/lib.rs`

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::EakvCache;
    use crate::kernels;

    #[test]
    fn test_attention_scores_shape() {
        let dir = kernels::find_kernel_dir().unwrap();
        let kt = kernels::load(&dir).unwrap();
        let mut cache = EakvCache::new(1, 2, 64, 8, kt).unwrap();

        // Load 4 tokens of data
        let data: Vec<f32> = (0..1 * 2 * 2 * 4 * 64)
            .map(|i| ((i % 100) as f32) * 0.01)
            .collect();
        cache.load_raw(&data, 4).unwrap();

        // Query: 2 heads × 64 dim
        let queries: Vec<f32> = (0..2 * 64).map(|i| (i as f32) * 0.01).collect();
        let mut scores = vec![0.0f32; 2 * 4]; // n_q_heads × seq_len
        attention_scores(&cache, &queries, 0, 2, 2, &mut scores);

        // Scores should be non-zero (data and queries are non-zero)
        assert!(scores.iter().any(|&s| s != 0.0), "scores should be non-zero");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: FAIL — module not found

- [ ] **Step 3: Implement attention.rs**

```rust
//! Attention on Q4 data — scores (Q·K) and output (weights·V).

use crate::cache::EakvCache;

pub fn attention_scores(
    cache: &EakvCache,
    queries: &[f32],
    layer: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    scores_out: &mut [f32],
) {
    let k = cache.kernels();
    let hd = cache.head_dim();
    let groups_per_head = cache.max_seq_len() * (hd / 64);
    let q_elems = (n_q_heads * hd) as usize;
    let n_q_groups = q_elems / 64;

    let mut rot_q = vec![0.0f32; q_elems];
    rot_q.copy_from_slice(&queries[..q_elems]);
    for g in 0..n_q_groups {
        unsafe {
            (k.turbo_rotate)(
                rot_q[g * 64..].as_mut_ptr(),
                cache.jl_signs().as_ptr(),
                64,
            );
        }
    }

    let weights = cache.weights(layer, 0);
    let scales = cache.scales(layer, 0);
    let biases = cache.biases(layer, 0);

    unsafe {
        if hd == 64 {
            if n_q_heads == n_kv_heads {
                (k.k_score_mha_64)(
                    rot_q.as_ptr(), weights.as_ptr(), scales.as_ptr(), biases.as_ptr(),
                    scores_out.as_mut_ptr(), cache.seq_len(), n_q_heads, groups_per_head,
                );
            } else {
                (k.k_score_gqa_64)(
                    rot_q.as_ptr(), weights.as_ptr(), scales.as_ptr(), biases.as_ptr(),
                    scores_out.as_mut_ptr(), cache.seq_len(), n_q_heads, n_kv_heads,
                    groups_per_head,
                );
            }
        } else if n_q_heads == n_kv_heads {
            (k.k_score_mha)(
                rot_q.as_ptr(), weights.as_ptr(), scales.as_ptr(), biases.as_ptr(),
                scores_out.as_mut_ptr(), cache.seq_len(), n_q_heads, groups_per_head,
            );
        } else {
            (k.k_score_gqa)(
                rot_q.as_ptr(), weights.as_ptr(), scales.as_ptr(), biases.as_ptr(),
                scores_out.as_mut_ptr(), cache.seq_len(), n_q_heads, n_kv_heads,
                groups_per_head,
            );
        }
    }
}

pub fn attention_output(
    cache: &EakvCache,
    weights_in: &[f32],
    layer: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    output_out: &mut [f32],
) {
    let k = cache.kernels();
    let hd = cache.head_dim();
    let groups_per_head = cache.max_seq_len() * (hd / 64);

    let v_weights = cache.weights(layer, 1);
    let v_scales = cache.scales(layer, 1);
    let v_biases = cache.biases(layer, 1);

    unsafe {
        if hd == 64 {
            if n_q_heads == n_kv_heads {
                (k.v_sum_mha_64)(
                    weights_in.as_ptr(), v_weights.as_ptr(), v_scales.as_ptr(), v_biases.as_ptr(),
                    output_out.as_mut_ptr(), cache.seq_len(), n_q_heads, groups_per_head,
                );
            } else {
                (k.v_sum_gqa_64)(
                    weights_in.as_ptr(), v_weights.as_ptr(), v_scales.as_ptr(), v_biases.as_ptr(),
                    output_out.as_mut_ptr(), cache.seq_len(), n_q_heads, n_kv_heads,
                    groups_per_head,
                );
            }
        } else if n_q_heads == n_kv_heads {
            (k.v_sum_mha)(
                weights_in.as_ptr(), v_weights.as_ptr(), v_scales.as_ptr(), v_biases.as_ptr(),
                output_out.as_mut_ptr(), cache.seq_len(), n_q_heads, groups_per_head,
            );
        } else {
            (k.v_sum_gqa)(
                weights_in.as_ptr(), v_weights.as_ptr(), v_scales.as_ptr(), v_biases.as_ptr(),
                output_out.as_mut_ptr(), cache.seq_len(), n_q_heads, n_kv_heads,
                groups_per_head,
            );
        }
    }

    // Inverse-rotate output to undo V pre-rotation
    let out_elems = (n_q_heads * hd) as usize;
    let n_out_groups = out_elems / 64;
    for g in 0..n_out_groups {
        unsafe {
            (k.fwht_inplace)(output_out[g * 64..].as_mut_ptr(), 64);
            (k.sign_flip)(output_out[g * 64..].as_mut_ptr(), cache.jl_signs().as_ptr(), 64);
        }
    }
}
```

- [ ] **Step 4: Add to lib.rs**

```rust
pub mod attention;
```

- [ ] **Step 5: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/eakv/src/attention.rs crates/eakv/src/lib.rs
git commit -m "feat(eakv): attention module — scores and output with 4-way dispatch"
```

---

### Task 5: IO module — save and load

**Files:**
- Create: `crates/eakv/src/io.rs`
- Modify: `crates/eakv/src/lib.rs`

- [ ] **Step 1: Write failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::EakvCache;
    use crate::kernels;

    #[test]
    fn test_save_load_roundtrip() {
        let dir = kernels::find_kernel_dir().unwrap();
        let kt = kernels::load(&dir).unwrap();
        let mut cache = EakvCache::new(1, 1, 64, 16, kt).unwrap();

        let data: Vec<f32> = (0..1 * 2 * 1 * 4 * 64)
            .map(|i| (i as f32) * 0.01)
            .collect();
        cache.load_raw(&data, 4).unwrap();

        let tmp = std::env::temp_dir().join("eakv_test_roundtrip.eakv");
        save(&cache, &tmp).unwrap();

        let kt2 = kernels::load(&dir).unwrap();
        let loaded = load(&tmp, kt2).unwrap();
        assert_eq!(loaded.seq_len(), 4);
        assert_eq!(loaded.n_layers(), 1);
        assert_eq!(loaded.n_heads(), 1);
        assert_eq!(loaded.head_dim(), 64);

        // Verify data matches
        assert_eq!(cache.weights(0, 0), loaded.weights(0, 0));
        assert_eq!(cache.scales(0, 0), loaded.scales(0, 0));
        assert_eq!(cache.biases(0, 0), loaded.biases(0, 0));

        std::fs::remove_file(&tmp).ok();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: FAIL

- [ ] **Step 3: Implement io.rs**

```rust
//! Binary .eakv format — save/load, backwards compatible with C version.

use crate::cache::EakvCache;
use crate::kernels::KernelTable;
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;

const MAGIC: [u8; 4] = *b"EAKV";
const HEADER_SIZE: usize = 512;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Header {
    magic: [u8; 4],
    version: u16,
    quant_scheme: u16,
    group_size: u32,
    orig_dtype: u16,
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    compression: i16,
    model_hash: [u8; 32],
    tokenizer_hash: [u8; 32],
    checksum: u64,
}

fn align64(x: usize) -> usize {
    (x + 63) & !63
}

pub fn save(cache: &EakvCache, path: &Path) -> Result<(), String> {
    let mut f = std::fs::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;

    let n_groups = (cache.n_heads() * cache.head_dim() * cache.seq_len()) / 64;
    let weights_size = n_groups as usize * 32;
    let scales_size = n_groups as usize * 4;
    let biases_size = n_groups as usize * 4;
    let block_raw = weights_size + scales_size + biases_size;
    let block_aligned = align64(block_raw);

    let mut header_buf = [0u8; HEADER_SIZE];
    let h = unsafe { &mut *(header_buf.as_mut_ptr() as *mut Header) };
    h.magic = MAGIC;
    h.version = 1;
    h.quant_scheme = 0;
    h.group_size = 64;
    h.orig_dtype = 0;
    h.n_layers = cache.n_layers() as u32;
    h.n_heads = cache.n_heads() as u32;
    h.head_dim = cache.head_dim() as u32;
    h.seq_len = cache.seq_len() as u32;
    h.max_seq_len = cache.seq_len() as u32;
    h.compression = 0;

    f.write_all(&header_buf).map_err(|e| e.to_string())?;

    let idx_table_size = cache.n_layers() as usize * 2 * 8;
    let data_start = align64(HEADER_SIZE + idx_table_size);

    let mut cur = data_start;
    for _l in 0..cache.n_layers() {
        let k_off = cur as u64;
        cur += block_aligned;
        let v_off = cur as u64;
        cur += block_aligned;
        f.write_all(&k_off.to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&v_off.to_le_bytes()).map_err(|e| e.to_string())?;
    }

    let pos = HEADER_SIZE + idx_table_size;
    if pos < data_start {
        let zeros = vec![0u8; data_start - pos];
        f.write_all(&zeros).map_err(|e| e.to_string())?;
    }

    let ng = n_groups as usize;
    for l in 0..cache.n_layers() {
        for kv in 0..2i32 {
            let w = cache.weights(l, kv);
            f.write_all(&w[..weights_size]).map_err(|e| e.to_string())?;
            let s = cache.scales(l, kv);
            let s_bytes = unsafe {
                std::slice::from_raw_parts(s[..ng].as_ptr() as *const u8, scales_size)
            };
            f.write_all(s_bytes).map_err(|e| e.to_string())?;
            let b = cache.biases(l, kv);
            let b_bytes = unsafe {
                std::slice::from_raw_parts(b[..ng].as_ptr() as *const u8, biases_size)
            };
            f.write_all(b_bytes).map_err(|e| e.to_string())?;

            let pad = block_aligned - block_raw;
            if pad > 0 {
                let zeros = vec![0u8; pad];
                f.write_all(&zeros).map_err(|e| e.to_string())?;
            }
        }
    }

    Ok(())
}

pub fn load(path: &Path, kernels: KernelTable) -> Result<EakvCache, String> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;

    let mut header_buf = [0u8; HEADER_SIZE];
    f.read_exact(&mut header_buf).map_err(|e| e.to_string())?;

    let h = unsafe { &*(header_buf.as_ptr() as *const Header) };
    if h.magic != MAGIC { return Err("invalid magic".to_string()); }
    if h.version != 1 { return Err(format!("unsupported version {}", h.version)); }

    let mut cache = EakvCache::new(
        h.n_layers as i32, h.n_heads as i32,
        h.head_dim as i32, h.seq_len as i32,
        kernels,
    ).ok_or_else(|| "invalid cache params in header".to_string())?;

    let n_groups = (h.n_heads * h.head_dim * h.seq_len) as usize / 64;
    let weights_size = n_groups * 32;

    let idx_table_size = h.n_layers as usize * 2 * 8;
    let mut offsets = vec![0u64; h.n_layers as usize * 2];
    let offset_bytes = unsafe {
        std::slice::from_raw_parts_mut(
            offsets.as_mut_ptr() as *mut u8,
            idx_table_size,
        )
    };
    f.read_exact(offset_bytes).map_err(|e| e.to_string())?;

    for l in 0..h.n_layers as i32 {
        for kv in 0..2i32 {
            let off = offsets[(l * 2 + kv) as usize];
            f.seek(SeekFrom::Start(off)).map_err(|e| e.to_string())?;

            let w = cache.weights_mut_pub(l, kv);
            f.read_exact(&mut w[..weights_size]).map_err(|e| e.to_string())?;

            let s = cache.scales_mut_pub(l, kv);
            let s_bytes = unsafe {
                std::slice::from_raw_parts_mut(s[..n_groups].as_mut_ptr() as *mut u8, n_groups * 4)
            };
            f.read_exact(s_bytes).map_err(|e| e.to_string())?;

            let b = cache.biases_mut_pub(l, kv);
            let b_bytes = unsafe {
                std::slice::from_raw_parts_mut(b[..n_groups].as_mut_ptr() as *mut u8, n_groups * 4)
            };
            f.read_exact(b_bytes).map_err(|e| e.to_string())?;
        }
    }

    cache.set_seq_len(h.seq_len as i32);
    Ok(cache)
}
```

Note: `weights_mut_pub`, `scales_mut_pub`, `biases_mut_pub` need to be added as `pub(crate)` aliases on `EakvCache` for the io module to write into the cache. Add these to cache.rs:

```rust
    pub(crate) fn weights_mut_pub(&mut self, layer: i32, kv_idx: i32) -> &mut [u8] {
        self.weights_mut(layer, kv_idx)
    }
    pub(crate) fn scales_mut_pub(&mut self, layer: i32, kv_idx: i32) -> &mut [f32] {
        self.scales_mut(layer, kv_idx)
    }
    pub(crate) fn biases_mut_pub(&mut self, layer: i32, kv_idx: i32) -> &mut [f32] {
        self.biases_mut(layer, kv_idx)
    }
```

- [ ] **Step 4: Add to lib.rs**

```rust
pub mod io;
```

- [ ] **Step 5: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add crates/eakv/src/io.rs crates/eakv/src/cache.rs crates/eakv/src/lib.rs
git commit -m "feat(eakv): io module — .eakv binary save/load"
```

---

### Task 6: Delete C code, final lib.rs, full test suite

**Files:**
- Delete: `crates/eakv/csrc/` (entire directory)
- Modify: `crates/eakv/src/lib.rs`

- [ ] **Step 1: Delete all C files**

```bash
rm -rf crates/eakv/csrc/
```

- [ ] **Step 2: Write final lib.rs with full public API**

```rust
//! eakv — Q4 KV cache quantization for LLM inference.
//!
//! Pure Rust orchestration layer. Loads Ea SIMD kernels at runtime
//! via libloading from ~/.olorin/lib/.

pub mod kernels;
pub mod cache;
pub mod attention;
pub mod io;

pub use cache::EakvCache;
pub use kernels::KernelTable;
pub use attention::{attention_scores, attention_output};
pub use io::{save, load};
```

- [ ] **Step 3: Run full test suite**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eakv 2>&1`
Expected: PASS — all tests green

- [ ] **Step 4: Build full workspace to verify nothing is broken**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1`
Expected: compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add -A crates/eakv/
git commit -m "refactor(eakv): remove C code, pure Rust with Ea SIMD kernels

Delete csrc/ entirely (cache.c, attention.c, io.c, ggml_type.c,
llama_bridge.c, all headers). No more cc build-dep or C compiler.
Kernels loaded via libloading from ~/.olorin/lib/."
```

- [ ] **Step 6: Push**

```bash
git push origin web-ui
```
