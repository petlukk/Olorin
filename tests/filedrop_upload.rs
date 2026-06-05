//! Step 3 of the file-drop analyst: the /api/analyze decode + staging helpers
//! (base64, filename sanitization, JSON-body → /tmp staging). The HTTP/SSE
//! wiring and the analysis itself are covered by step 2's tests; here we pin
//! the new decoding/staging logic that turns an upload into a /tmp file.

use olorin::interface::server_analyze::{
    base64_decode, drain_to_file, header_value, parse_and_stage, sanitize_filename,
};

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
fn drain_to_file_writes_leftover_then_stream() {
    let path = "/tmp/filedrop_drain_a";
    // leftover holds the body bytes that arrived with the headers; the rest
    // comes from the reader.
    let mut reader = std::io::Cursor::new(b"world".to_vec());
    let written = drain_to_file(&mut reader, b"hello ", 11, path).unwrap();
    assert_eq!(written, 11);
    assert_eq!(std::fs::read(path).unwrap(), b"hello world");
}

#[test]
fn drain_to_file_stops_at_content_len() {
    let path = "/tmp/filedrop_drain_b";
    // content_len shorter than available bytes — must not over-read.
    let mut reader = std::io::Cursor::new(b"world EXTRA".to_vec());
    let written = drain_to_file(&mut reader, b"hello ", 8, path).unwrap();
    assert_eq!(written, 8);
    assert_eq!(std::fs::read(path).unwrap(), b"hello wo");
}

#[test]
fn header_value_is_case_insensitive() {
    let req = "POST /api/analyze_raw HTTP/1.1\r\nHost: x\r\nX-Filename: nasa_1gb.log\r\nContent-Length: 5\r\n\r\n";
    assert_eq!(header_value(req, "x-filename").as_deref(), Some("nasa_1gb.log"));
    assert_eq!(header_value(req, "content-length").as_deref(), Some("5"));
    assert_eq!(header_value(req, "nope"), None);
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
