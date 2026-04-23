use olorin::kernels::ffi;

#[test]
fn test_kernel_init() {
    ffi::init().expect("kernel init failed");
}

#[test]
fn test_kernel_dir_exists() {
    ffi::init().unwrap();
    let dir = ffi::kernel_dir().unwrap();
    assert!(dir.exists());
}

#[test]
fn test_zeroize() {
    ffi::init().unwrap();
    let mut buf = vec![0xFFu8; 64];
    unsafe { ffi::zeroize(buf.as_mut_ptr(), buf.len() as i32) };
    assert!(buf.iter().all(|&b| b == 0));
}
