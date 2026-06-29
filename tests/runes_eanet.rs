//! eanet rune tests: registration, output safety, selection wiring, and an
//! end-to-end synthetic-pcap triage where a scanner + exfil surface from
//! generic metrics — the same property the real CTU-13 capture validated.

use olorin::kernels::ffi;
use olorin::runes::narration::{is_grid_continuation, looks_like_data_dump};
use olorin::runes::output::RuneOutput;
use olorin::runes::select::pick_rune_name;
use olorin::runes::{run_rune, OutputSafety, RUNES};

fn ipv4_tcp(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, payload: usize) -> Vec<u8> {
    let mut p = vec![0u8; 12];
    p.extend_from_slice(&[0x08, 0x00]);
    let mut ip = vec![0u8; 20];
    ip[0] = 0x45;
    ip[9] = 6;
    ip[12..16].copy_from_slice(&src);
    ip[16..20].copy_from_slice(&dst);
    p.extend_from_slice(&ip);
    let mut tcp = vec![0u8; 20];
    tcp[0..2].copy_from_slice(&sport.to_be_bytes());
    tcp[2..4].copy_from_slice(&dport.to_be_bytes());
    p.extend_from_slice(&tcp);
    p.extend(std::iter::repeat(0u8).take(payload));
    p
}

fn record(pkt: &[u8]) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&[0u8; 8]);
    r.extend_from_slice(&(pkt.len() as u32).to_le_bytes());
    r.extend_from_slice(&(pkt.len() as u32).to_le_bytes());
    r.extend_from_slice(pkt);
    r
}

fn global_header() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(&0xa1b2c3d4u32.to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&4u16.to_le_bytes());
    h.extend_from_slice(&0i32.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes());
    h.extend_from_slice(&65535u32.to_le_bytes());
    h.extend_from_slice(&1u32.to_le_bytes());
    h
}

/// A capture with an exfil (heavy talker), a scanner (high fan-out), and benign
/// background traffic. Returns a /tmp path the allowlist accepts.
fn write_incident_pcap(name: &str) -> String {
    let mut cap = global_header();
    for _ in 0..100 {
        cap.extend(record(&ipv4_tcp([10, 0, 0, 99], [203, 0, 113, 7], 51000, 443, 1446)));
    }
    for i in 0..50u16 {
        cap.extend(record(&ipv4_tcp([10, 0, 0, 66], [172, 16, (i >> 8) as u8, i as u8], 40000, 445, 0)));
    }
    for h in 0..5u8 {
        for d in 0..2u8 {
            cap.extend(record(&ipv4_tcp([192, 168, 0, h], [8, 8, 8, d], 1234, 80, 40)));
        }
    }
    let path = std::env::temp_dir().join(name);
    std::fs::write(&path, &cap).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn eanet_is_registered() {
    ffi::init().unwrap();
    assert!(RUNES.iter().any(|r| r.name() == "eanet"), "eanet missing from registry");
}

#[test]
fn eanet_output_safety_is_untrusted() {
    let r = RUNES.iter().find(|r| r.name() == "eanet").expect("eanet registered");
    assert_eq!(r.output_safety(), OutputSafety::UntrustedQuoted);
}

#[test]
fn pcap_routes_to_eanet() {
    assert_eq!(pick_rune_name("capture.pcap", &[]), Some("eanet"));
    assert_eq!(pick_rune_name("capture.pcapng", &[]), Some("eanet"));
    // Extensionless capture detected by magic bytes.
    assert_eq!(pick_rune_name("capture", &[0xd4, 0xc3, 0xb2, 0xa1, 0, 0]), Some("eanet"));
}

#[test]
fn eanet_triage_surfaces_scanner_and_exfil() {
    ffi::init().unwrap();
    let path = write_incident_pcap("olorin_eanet_human.pcap");
    let r = run_rune("eanet", &path).expect("eanet runs");
    assert!(r.success, "rune failed: {}", r.answer);
    assert!(r.answer.contains("10.0.0.66"), "scanner host missing:\n{}", r.answer);
    assert!(r.answer.contains("10.0.0.99 -> 203.0.113.7"), "exfil pair missing:\n{}", r.answer);
    assert!(r.answer.contains("destinations"), "fan-out section missing:\n{}", r.answer);
    std::fs::remove_file(&path).ok();
}

#[test]
fn eanet_json_flags_scanner_anomaly() {
    ffi::init().unwrap();
    let path = write_incident_pcap("olorin_eanet_json.pcap");
    let r = run_rune("eanet", &format!("--json {path}")).expect("eanet runs");
    assert!(r.structured, "--json must set structured");
    let out = RuneOutput::from_json(r.answer.as_bytes()).expect("valid RuneOutput JSON");
    assert_eq!(out.rune, "eanet");
    assert_eq!(out.totals.rows, 160, "100 exfil + 50 scan + 10 benign");
    assert!(
        out.anomalies.iter().any(|a| a.bucket.contains("10.0.0.66")),
        "scanner anomaly missing: {:?}", out.anomalies
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn eanet_narration_survives_grid_filter() {
    ffi::init().unwrap();
    let path = write_incident_pcap("olorin_eanet_grid.pcap");
    let r = run_rune("eanet", &path).expect("eanet runs");
    std::fs::remove_file(&path).ok();
    let answer = &r.answer;

    // A plausible 1–2 sentence prose narration must survive BOTH filters.
    let prose = "Host 10.0.0.66 contacted thousands of distinct destinations, a likely \
                 horizontal port scan, while 10.0.0.99 moved hundreds of megabytes to a \
                 single external host — possible exfiltration.";
    assert!(!is_grid_continuation(answer, prose), "prose narration wrongly flagged as grid");
    assert!(!looks_like_data_dump(prose), "prose narration wrongly flagged as a data dump");

    // A model that echoes one of the ranking rows must be discarded.
    let grid_echo = "10.0.0.66   6000 destinations";
    assert!(is_grid_continuation(answer, grid_echo), "grid-row echo should be caught");
}

#[test]
fn eanet_answer_is_narratable_not_safety_blocked() {
    // build_narration_prompt returns None (silently killing narration) if the
    // rune answer trips safety::scan. eanet's output is full of "scan",
    // "exfiltration", and IPs — exactly the kind of content a safety classifier
    // might flag. Guard against the rune's own findings blocking its narration.
    ffi::init().unwrap();
    let path = write_incident_pcap("olorin_eanet_narr.pcap");
    let result = run_rune("eanet", &path).expect("eanet runs");
    std::fs::remove_file(&path).ok();
    let safety = RUNES.iter().find(|r| r.name() == "eanet").unwrap().output_safety();
    let prompt = olorin::runes::build_narration_prompt("eanet", safety, result);
    assert!(
        prompt.is_some(),
        "eanet answer was safety-blocked from narration — narration would silently never run"
    );
    assert!(prompt.unwrap().contains("findings:"), "narration prompt should carry the findings");
}

#[test]
fn eanet_rejects_pcapng_with_reason() {
    ffi::init().unwrap();
    let path = std::env::temp_dir().join("olorin_eanet_ng.pcap");
    let mut data = vec![0x0a, 0x0d, 0x0d, 0x0a];
    data.extend(std::iter::repeat(0u8).take(28));
    std::fs::write(&path, &data).unwrap();
    let r = run_rune("eanet", &path.to_string_lossy()).expect("eanet runs");
    assert!(!r.success);
    assert!(r.answer.contains("pcapng"), "expected pcapng reason: {}", r.answer);
    std::fs::remove_file(&path).ok();
}
