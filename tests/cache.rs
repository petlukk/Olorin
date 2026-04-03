use olorin::kernels::ffi;
use olorin::inference::cache::F16KvCache;

#[test]
fn test_f16_cache_new() {
    ffi::init().unwrap();
    let cache = F16KvCache::new(1, 5, 128, 64).unwrap();
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.n_layers(), 1);
    assert_eq!(cache.n_kv_heads(), 5);
    assert_eq!(cache.head_dim(), 128);
}

#[test]
fn test_f16_cache_store_advance() {
    ffi::init().unwrap();
    let mut cache = F16KvCache::new(1, 2, 8, 64).unwrap();
    // One token, 2 heads × 8 dim = 16 floats
    let k = vec![1.0f32; 16];
    let v = vec![0.5f32; 16];
    cache.store(0, 0, &k, 1).unwrap();
    cache.store(0, 1, &v, 1).unwrap();
    cache.advance(1).unwrap();
    assert_eq!(cache.len(), 1);
}

#[test]
fn test_f16_cache_checkpoint_restore() {
    ffi::init().unwrap();
    let mut cache = F16KvCache::new(1, 2, 8, 64).unwrap();
    let data = vec![1.0f32; 16];
    cache.store(0, 0, &data, 1).unwrap();
    cache.store(0, 1, &data, 1).unwrap();
    cache.advance(1).unwrap();
    let cp = cache.checkpoint();
    assert_eq!(cp, 1);
    cache.restore(0).unwrap();
    assert_eq!(cache.len(), 0);
}
