use std::path::Path;

fn model_path() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let p = Path::new(&home).join(".olorin/models");
    std::fs::read_dir(&p).ok()?
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .map(|e| e.path())
}

#[test]
fn test_gguf_parse() {
    olorin::kernels::ffi::init().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no model file found");
        return;
    };
    let gguf = olorin::inference::gguf::GgufFile::open(&path).unwrap();
    eprintln!("GGUF version={} tensors={} metadata_keys={}",
        gguf.version, gguf.tensors.len(), gguf.metadata.len());
    assert!(gguf.version >= 2 && gguf.version <= 3, "unexpected GGUF version");
    assert!(!gguf.tensors.is_empty(), "no tensors found");
}

#[test]
fn test_gguf_open_missing() {
    let result = olorin::inference::gguf::GgufFile::open(Path::new("/tmp/nonexistent_olorin_test.gguf"));
    assert!(result.is_err());
}

#[test]
fn test_gguf_open_bad_magic() {
    let path = std::path::PathBuf::from("/tmp/bad_magic_olorin.gguf");
    std::fs::write(&path, &[0x42u8, 0x41, 0x41, 0x44, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0,
                              0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
    let result = olorin::inference::gguf::GgufFile::open(&path);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("magic") || msg.contains("GGUF"), "expected magic error, got: {msg}");
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_gguf_minimal_roundtrip() {
    use olorin::inference::gguf::GgufFile;

    const GGUF_MAGIC: u32 = 0x4655_4747;
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&0u64.to_le_bytes()); // n_tensors
    buf.extend_from_slice(&1u64.to_le_bytes()); // n_kv

    let key = b"test.key";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&8u32.to_le_bytes()); // STRING type
    let val = b"hello";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val);

    let path = std::path::PathBuf::from("/tmp/olorin_minimal_test.gguf");
    std::fs::write(&path, &buf).unwrap();
    let gf = GgufFile::open(&path).unwrap();
    assert_eq!(gf.version, 3);
    assert_eq!(gf.tensors.len(), 0);
    assert_eq!(gf.get_str("test.key"), Some("hello"));
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_tokenizer_from_gguf() {
    olorin::kernels::ffi::init().unwrap();
    let Some(path) = model_path() else {
        eprintln!("SKIP: no model file found");
        return;
    };
    let gguf = olorin::inference::gguf::GgufFile::open(&path).unwrap();
    let tok = olorin::inference::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    eprintln!("Tokenizer loaded, bos_id={} eos_id={}", tok.bos_id, tok.eos_id);

    // Encode and decode a simple string
    let text = "Hello";
    let ids = tok.encode(text);
    assert!(!ids.is_empty(), "encode returned empty");
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, text, "roundtrip failed: {decoded:?} != {text:?}");
}

