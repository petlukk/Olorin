//! Step 3 of the file-drop analyst: the /api/analyze decode + staging helpers
//! (base64, filename sanitization, JSON-body → /tmp staging). The HTTP/SSE
//! wiring and the analysis itself are covered by step 2's tests; here we pin
//! the new decoding/staging logic that turns an upload into a /tmp file.

use olorin::interface::server_analyze::{base64_decode, parse_and_stage, sanitize_filename};

#[test]
fn base64_known_vectors() {
    assert_eq!(base64_decode("TWFu").unwrap(), b"Man");
    assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
    assert_eq!(base64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
    assert_eq!(base64_decode("").unwrap(), b"");
    // whitespace inside the payload is skipped
    assert_eq!(base64_decode("aGVs\nbG8=").unwrap(), b"hello");
    // data-URL prefix is tolerated
    assert_eq!(base64_decode("data:text/plain;base64,aGk=").unwrap(), b"hi");
}

#[test]
fn base64_rejects_invalid() {
    assert!(base64_decode("not valid !@#").is_none());
}

#[test]
fn sanitize_strips_paths_and_unsafe_chars() {
    assert_eq!(sanitize_filename("app.log"), "app.log");
    assert_eq!(sanitize_filename("/etc/passwd"), "passwd");
    assert_eq!(sanitize_filename("../../evil.sh"), "evil.sh");
    assert_eq!(sanitize_filename("a b.csv"), "a_b.csv");
    assert_eq!(sanitize_filename(".."), "dropped-file");
    assert_eq!(sanitize_filename(""), "dropped-file");
    assert_eq!(sanitize_filename("weird;name|x.log"), "weird_name_x.log");
}

#[test]
fn parse_and_stage_writes_tmp_file() {
    // b64("hello") = aGVsbG8=
    let body = br#"{"files":[{"name":"note.txt","b64":"aGVsbG8="}]}"#;
    let staged = parse_and_stage(body).expect("should stage");
    assert_eq!(staged.len(), 1);
    let (name, path) = &staged[0];
    assert_eq!(name, "note.txt");
    assert!(path.starts_with("/tmp/"), "staged under /tmp: {path}");
    assert_eq!(std::fs::read(path).unwrap(), b"hello");
}

#[test]
fn parse_and_stage_handles_multiple_files() {
    // b64("hi")=aGk=, b64("yo")=eW8=
    let body = br#"{"files":[{"name":"a.log","b64":"aGk="},{"name":"b.log","b64":"eW8="}]}"#;
    let staged = parse_and_stage(body).expect("should stage both");
    assert_eq!(staged.len(), 2);
    assert_eq!(staged[0].0, "a.log");
    assert_eq!(staged[1].0, "b.log");
    assert_eq!(std::fs::read(&staged[0].1).unwrap(), b"hi");
    assert_eq!(std::fs::read(&staged[1].1).unwrap(), b"yo");
    assert_ne!(staged[0].1, staged[1].1, "each file gets a distinct path");
}

#[test]
fn parse_and_stage_rejects_bad_input() {
    assert!(parse_and_stage(b"not json").is_err());
    assert!(parse_and_stage(br#"{"files":[]}"#).is_err());
    assert!(parse_and_stage(br#"{"nope":1}"#).is_err());
    assert!(parse_and_stage(br#"{"files":[{"name":"x.csv"}]}"#).is_err()); // no b64
}

#[test]
fn parse_and_stage_traversal_name_stays_in_tmp() {
    let body = br#"{"files":[{"name":"../../../etc/cron.d/evil","b64":"aGk="}]}"#;
    let staged = parse_and_stage(body).expect("should stage");
    let path = &staged[0].1;
    // The sanitized basename must keep the write under the /tmp drop dir.
    assert!(path.starts_with("/tmp/olorin-drop-"), "no escape: {path}");
    assert!(!path.contains(".."), "no traversal in path: {path}");
}
