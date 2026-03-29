use olorin::kernels::ffi;
use olorin::inference::cache::EakvCache;

#[test]
fn test_cache_new() {
    ffi::init().unwrap();
    let cache = EakvCache::new(1, 5, 128, 64,
        olorin::inference::cache::KernelTable::init().unwrap()).unwrap();
    assert_eq!(cache.len(), 0);
}

#[test]
fn test_cache_append_advance() {
    ffi::init().unwrap();
    let mut cache = EakvCache::new(1, 5, 128, 64,
        olorin::inference::cache::KernelTable::init().unwrap()).unwrap();
    let k = vec![0.1f32; 5 * 128];
    let v = vec![0.2f32; 5 * 128];
    cache.append(&k, 0, 0, 1).unwrap();
    cache.append(&v, 0, 1, 1).unwrap();
    cache.advance(1).unwrap();
    assert_eq!(cache.len(), 1);
}
