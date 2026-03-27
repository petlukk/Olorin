//! KV cache — quantized storage with TurboQuant rotation.
//!
//! Port of csrc/cache.c. Single flat allocation, per-group Q4 packing.

use crate::KernelTable;

/// Offsets into `data_buf` for one KV slot (K or V for one layer).
#[derive(Clone, Copy, Debug)]
pub struct KvSlice {
    pub weights_offset: usize,
    pub scales_offset: usize,
    pub biases_offset: usize,
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
    kernels: KernelTable,
}

// SAFETY: EakvCache holds function pointers (via KernelTable) that are
// valid for the lifetime of the loaded libraries. KernelTable is Send+Sync.
unsafe impl Send for EakvCache {}

/// Generate the TurboQuant sign mask using xorshift64 PRNG.
/// Seed = 0x4F6C6F72696E4A4C ("OlorinJL"). Must be byte-identical to C.
fn gen_jl_signs() -> [f32; 64] {
    let mut signs = [0.0f32; 64];
    let mut rng: u64 = 0x4F6C6F72696E4A4C;
    for s in signs.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *s = if rng % 2 == 0 { 1.0 } else { -1.0 };
    }
    signs
}

impl EakvCache {
    /// Create a new cache. Returns `None` on invalid params.
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

        // Per-slot size: max_groups * 32 (weights) + max_groups * 4 (scales) + max_groups * 4 (biases)
        let blk = max_groups as usize * 32
            + max_groups as usize * 4
            + max_groups as usize * 4;
        let n_slots = n_layers as usize * 2;
        let total = blk * n_slots;

        let data_buf = vec![0u8; total];

        let mut kv = Vec::with_capacity(n_slots);
        let mut offset = 0usize;
        for _ in 0..n_layers {
            for _ in 0..2 {
                let weights_offset = offset;
                offset += max_groups as usize * 32;
                let scales_offset = offset;
                offset += max_groups as usize * 4;
                let biases_offset = offset;
                offset += max_groups as usize * 4;
                kv.push(KvSlice {
                    weights_offset,
                    scales_offset,
                    biases_offset,
                });
            }
        }
        debug_assert_eq!(offset, total);

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
            kernels,
        })
    }

    // ── Simple accessors ──

    pub fn seq_len(&self) -> i32 {
        self.seq_len
    }
    pub fn n_layers(&self) -> i32 {
        self.n_layers
    }
    pub fn n_heads(&self) -> i32 {
        self.n_kv_heads
    }
    pub fn head_dim(&self) -> i32 {
        self.head_dim
    }
    pub fn max_seq_len(&self) -> i32 {
        self.max_seq_len
    }
    pub fn groups_per_token(&self) -> i32 {
        self.groups_per_token
    }

    // ── Lifecycle ──

    pub fn checkpoint(&self) -> i32 {
        self.seq_len
    }

    pub fn restore(&mut self, seq_len: i32) -> Result<(), String> {
        if seq_len < 0 || seq_len > self.seq_len {
            return Err(format!(
                "restore: seq_len {seq_len} out of range [0, {}]",
                self.seq_len
            ));
        }
        self.seq_len = seq_len;
        Ok(())
    }

    pub fn advance(&mut self, n_tokens: i32) -> Result<(), String> {
        if n_tokens <= 0 {
            return Err("advance: n_tokens must be > 0".into());
        }
        if self.seq_len + n_tokens > self.max_seq_len {
            return Err(format!(
                "advance: {} + {} > max {}",
                self.seq_len, n_tokens, self.max_seq_len
            ));
        }
        self.seq_len += n_tokens;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.seq_len = 0;
    }

    pub fn compression_ratio(&self) -> f32 {
        if self.seq_len == 0 {
            return 0.0;
        }
        40.0 / 256.0
    }

    // ── Rotation helpers ──

    fn rotate_groups(&self, buf: &mut [f32], n_groups: i32) {
        for g in 0..n_groups as usize {
            unsafe {
                (self.kernels.rotate)(
                    buf.as_mut_ptr().add(g * 64),
                    self.jl_signs.as_ptr(),
                    64,
                );
            }
        }
    }

    #[allow(dead_code)]
    fn inverse_rotate_groups(&self, buf: &mut [f32], n_groups: i32) {
        for g in 0..n_groups as usize {
            unsafe {
                let ptr = buf.as_mut_ptr().add(g * 64);
                (self.kernels.fwht)(ptr, 64);
                (self.kernels.sign_flip)(ptr, self.jl_signs.as_ptr(), 64);
            }
        }
    }

    // ── Bulk load (matches cache.c eakv_cache_load_raw) ──

    /// Load raw f32 data for all layers. Layout: [layer][kv_idx][head][seq][dim].
    pub fn load_raw(&mut self, data: &[f32], seq_len: i32) -> Result<(), String> {
        if seq_len <= 0 {
            return Err("load_raw: seq_len must be > 0".into());
        }
        if seq_len > self.max_seq_len {
            return Err(format!(
                "load_raw: seq_len {} > max {}",
                seq_len, self.max_seq_len
            ));
        }

        let hd = self.head_dim as usize;
        let nh = self.n_kv_heads as usize;
        let gpd = hd / 64; // groups per token per head
        let gph = self.max_seq_len as usize * gpd; // stride: groups per head in buffer
        let n_groups_per_head = seq_len as usize * gpd;
        let elems_per_lkv = nh * seq_len as usize * hd;
        let head_elems = seq_len as usize * hd;

        let expected = self.n_layers as usize * 2 * elems_per_lkv;
        if data.len() < expected {
            return Err(format!(
                "load_raw: data too short ({} < {expected})",
                data.len()
            ));
        }

        let mut tmp = vec![0i32; n_groups_per_head * 32];
        let mut rot_buf = vec![0.0f32; head_elems];

        for l in 0..self.n_layers as usize {
            for kv_idx in 0..2usize {
                let lkv_src = &data[(l * 2 + kv_idx) * elems_per_lkv..];
                let slot = self.kv[l * 2 + kv_idx];

                for h in 0..nh {
                    let src = &lkv_src[h * head_elems..h * head_elems + head_elems];
                    let group_base = h * gph;

                    rot_buf[..head_elems].copy_from_slice(src);
                    self.rotate_groups(&mut rot_buf[..head_elems], n_groups_per_head as i32);

                    // Quantize into tmp + scales + biases
                    let scales_ptr = unsafe {
                        (self.data_buf.as_mut_ptr().add(slot.scales_offset)
                            as *mut f32)
                            .add(group_base)
                    };
                    let biases_ptr = unsafe {
                        (self.data_buf.as_mut_ptr().add(slot.biases_offset)
                            as *mut f32)
                            .add(group_base)
                    };
                    unsafe {
                        (self.kernels.quantize)(
                            rot_buf.as_ptr(),
                            tmp.as_mut_ptr(),
                            scales_ptr,
                            biases_ptr,
                            n_groups_per_head as i32,
                        );
                    }

                    // Truncate i32 → u8 into weights buffer (matches C cast)
                    let w_start = slot.weights_offset + group_base * 32;
                    for i in 0..n_groups_per_head * 32 {
                        self.data_buf[w_start + i] = tmp[i] as u8;
                    }
                }
            }
        }

        self.seq_len = seq_len;
        Ok(())
    }

    // ── Incremental append (matches cache.c eakv_cache_append) ──

    /// Append tokens to one (layer, kv_idx) slot.
    /// Input layout: [head][token][dim]. Does NOT advance seq_len.
    pub fn append(
        &mut self,
        data: &[f32],
        layer: i32,
        kv_idx: i32,
        n_tokens: i32,
    ) -> Result<(), String> {
        if n_tokens <= 0 {
            return Err("append: n_tokens must be > 0".into());
        }
        if layer < 0 || layer >= self.n_layers {
            return Err(format!(
                "append: layer {} out of range [0, {})",
                layer, self.n_layers
            ));
        }
        if kv_idx < 0 || kv_idx > 1 {
            return Err("append: kv_idx must be 0 or 1".into());
        }
        if self.seq_len + n_tokens > self.max_seq_len {
            return Err(format!(
                "append: {} + {} > max {}",
                self.seq_len, n_tokens, self.max_seq_len
            ));
        }

        let hd = self.head_dim as usize;
        let nh = self.n_kv_heads as usize;
        let gpd = hd / 64;
        let gph = self.max_seq_len as usize * gpd;
        let n_groups_per_head = n_tokens as usize * gpd;
        let head_elems = n_tokens as usize * hd;

        let expected = nh * head_elems;
        if data.len() < expected {
            return Err(format!(
                "append: data too short ({} < {expected})",
                data.len()
            ));
        }

        let mut tmp = vec![0i32; n_groups_per_head * 32];
        let mut rot_buf = vec![0.0f32; head_elems];

        let slot = self.kv[layer as usize * 2 + kv_idx as usize];

        for h in 0..nh {
            let src = &data[h * head_elems..h * head_elems + head_elems];
            let group_base = h * gph + self.seq_len as usize * gpd;

            rot_buf[..head_elems].copy_from_slice(src);
            self.rotate_groups(&mut rot_buf[..head_elems], n_groups_per_head as i32);

            let scales_ptr = unsafe {
                (self.data_buf.as_mut_ptr().add(slot.scales_offset) as *mut f32)
                    .add(group_base)
            };
            let biases_ptr = unsafe {
                (self.data_buf.as_mut_ptr().add(slot.biases_offset) as *mut f32)
                    .add(group_base)
            };
            unsafe {
                (self.kernels.quantize)(
                    rot_buf.as_ptr(),
                    tmp.as_mut_ptr(),
                    scales_ptr,
                    biases_ptr,
                    n_groups_per_head as i32,
                );
            }

            let w_start = slot.weights_offset + group_base * 32;
            for i in 0..n_groups_per_head * 32 {
                self.data_buf[w_start + i] = tmp[i] as u8;
            }
        }

        Ok(())
    }

    // ── pub(crate) accessors for attention/dequant ──

    pub(crate) fn jl_signs(&self) -> &[f32; 64] {
        &self.jl_signs
    }

    pub(crate) fn kernels(&self) -> &KernelTable {
        &self.kernels
    }

    pub(crate) fn set_seq_len(&mut self, v: i32) {
        self.seq_len = v;
    }

    pub(crate) fn kv_slice(&self, layer: i32, kv_idx: i32) -> KvSlice {
        self.kv[layer as usize * 2 + kv_idx as usize]
    }

    /// Immutable weights slice for a (layer, kv_idx) slot.
    pub(crate) fn weights(&self, layer: i32, kv_idx: i32) -> &[u8] {
        let s = self.kv_slice(layer, kv_idx);
        let len = self.max_groups as usize * 32;
        &self.data_buf[s.weights_offset..s.weights_offset + len]
    }

    /// Mutable weights slice for a (layer, kv_idx) slot.
    #[allow(dead_code)]
    pub(crate) fn weights_mut(&mut self, layer: i32, kv_idx: i32) -> &mut [u8] {
        let s = self.kv_slice(layer, kv_idx);
        let len = self.max_groups as usize * 32;
        &mut self.data_buf[s.weights_offset..s.weights_offset + len]
    }

    /// Immutable scales slice (as f32) for a (layer, kv_idx) slot.
    pub(crate) fn scales(&self, layer: i32, kv_idx: i32) -> &[f32] {
        let s = self.kv_slice(layer, kv_idx);
        let len = self.max_groups as usize;
        let start = s.scales_offset;
        // SAFETY: data_buf was allocated aligned and sized for f32 at these offsets
        unsafe {
            std::slice::from_raw_parts(
                self.data_buf.as_ptr().add(start) as *const f32,
                len,
            )
        }
    }

    /// Mutable scales slice for a (layer, kv_idx) slot.
    #[allow(dead_code)]
    pub(crate) fn scales_mut(&mut self, layer: i32, kv_idx: i32) -> &mut [f32] {
        let s = self.kv_slice(layer, kv_idx);
        let len = self.max_groups as usize;
        let start = s.scales_offset;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data_buf.as_mut_ptr().add(start) as *mut f32,
                len,
            )
        }
    }

    /// Immutable biases slice (as f32) for a (layer, kv_idx) slot.
    pub(crate) fn biases(&self, layer: i32, kv_idx: i32) -> &[f32] {
        let s = self.kv_slice(layer, kv_idx);
        let len = self.max_groups as usize;
        let start = s.biases_offset;
        unsafe {
            std::slice::from_raw_parts(
                self.data_buf.as_ptr().add(start) as *const f32,
                len,
            )
        }
    }

    /// Mutable biases slice for a (layer, kv_idx) slot.
    #[allow(dead_code)]
    pub(crate) fn biases_mut(&mut self, layer: i32, kv_idx: i32) -> &mut [f32] {
        let s = self.kv_slice(layer, kv_idx);
        let len = self.max_groups as usize;
        let start = s.biases_offset;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data_buf.as_mut_ptr().add(start) as *mut f32,
                len,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;

    fn load_kernels() -> Option<KernelTable> {
        let dir = kernels::find_kernel_dir().ok()?;
        KernelTable::load(&dir).ok()
    }

    #[test]
    fn test_gen_jl_signs() {
        let signs = gen_jl_signs();
        // Every element must be +1 or -1
        for &s in &signs {
            assert!(s == 1.0 || s == -1.0, "bad sign: {s}");
        }
        // Not all the same (extremely unlikely with proper PRNG)
        let n_pos = signs.iter().filter(|&&s| s == 1.0).count();
        assert!(n_pos > 0 && n_pos < 64);
    }

    #[test]
    fn test_create_invalid_params() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        // head_dim not multiple of 64 (with n_kv_heads=1)
        assert!(EakvCache::new(1, 1, 65, 128, kt).is_none());
    }

    #[test]
    fn test_create_invalid_zero() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        assert!(EakvCache::new(0, 4, 64, 128, kt).is_none());
    }

    #[test]
    fn test_checkpoint_restore() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        let mut cache = EakvCache::new(2, 4, 64, 128, kt).unwrap();
        assert_eq!(cache.checkpoint(), 0);

        cache.advance(10).unwrap();
        assert_eq!(cache.checkpoint(), 10);

        let cp = cache.checkpoint();
        cache.advance(5).unwrap();
        assert_eq!(cache.seq_len(), 15);

        cache.restore(cp).unwrap();
        assert_eq!(cache.seq_len(), 10);

        // Cannot restore beyond current seq_len
        assert!(cache.restore(15).is_err());
        // Cannot restore negative
        assert!(cache.restore(-1).is_err());
    }

    #[test]
    fn test_load_raw_sets_seq_len() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        let n_layers = 1;
        let n_kv_heads = 1;
        let head_dim = 64;
        let max_seq = 32;
        let seq = 4;
        let mut cache = EakvCache::new(n_layers, n_kv_heads, head_dim, max_seq, kt).unwrap();

        let elems = n_layers as usize * 2 * n_kv_heads as usize * seq as usize * head_dim as usize;
        let data = vec![0.5f32; elems];

        cache.load_raw(&data, seq).unwrap();
        assert_eq!(cache.seq_len(), seq);
    }

    #[test]
    fn test_append_advance() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        let n_layers = 1;
        let n_kv_heads = 2;
        let head_dim = 64;
        let max_seq = 64;
        let mut cache = EakvCache::new(n_layers, n_kv_heads, head_dim, max_seq, kt).unwrap();

        let n_tokens = 1;
        let head_elems = n_tokens as usize * head_dim as usize;
        let data = vec![1.0f32; n_kv_heads as usize * head_elems];

        // Append K and V for layer 0
        cache.append(&data, 0, 0, n_tokens).unwrap();
        cache.append(&data, 0, 1, n_tokens).unwrap();
        cache.advance(n_tokens).unwrap();
        assert_eq!(cache.seq_len(), 1);

        // Append another token
        cache.append(&data, 0, 0, 1).unwrap();
        cache.append(&data, 0, 1, 1).unwrap();
        cache.advance(1).unwrap();
        assert_eq!(cache.seq_len(), 2);
    }

    #[test]
    fn test_clear() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        let mut cache = EakvCache::new(1, 1, 64, 32, kt).unwrap();
        cache.advance(5).unwrap();
        assert_eq!(cache.seq_len(), 5);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }
}
