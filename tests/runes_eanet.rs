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
fn eanet_answer_compact_tables_in_details() {
    ffi::init().unwrap();
    let path = write_incident_pcap("olorin_eanet_grid.pcap");
    let r = run_rune("eanet", &path).expect("eanet runs");
    std::fs::remove_file(&path).ok();
    let answer = &r.answer;

    // The LLM-facing `answer` is compact (stats + findings), so the ranking
    // tables live in `details` — the model never sees the grid and can't be
    // drowned by it. The verbose tables are still shown to the user via details.
    assert!(answer.contains("findings:"), "answer should carry the findings");
    assert!(!answer.contains("top destination fan-in"), "ranking tables must NOT be in the answer");
    let details = r.details.as_deref().expect("eanet should populate details with the tables");
    assert!(details.contains("top destination fan-in"), "tables belong in details");

    // A plausible prose narration of the compact answer survives both filters.
    let prose = "Host 10.0.0.66 contacted thousands of distinct destinations, a likely \
                 horizontal port scan, while 10.0.0.99 moved data to a single external host.";
    assert!(!is_grid_continuation(answer, prose), "compact answer must not trip the grid filter");
    assert!(!looks_like_data_dump(prose));
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
fn eanet_output_is_chartable() {
    ffi::init().unwrap();
    let path = write_incident_pcap("olorin_eanet_chart.pcap");
    let r = run_rune("eanet", &format!("--json {path}")).expect("eanet runs");
    std::fs::remove_file(&path).ok();
    let out = RuneOutput::from_json(r.answer.as_bytes()).expect("valid json");

    // categories[] is the source fan-out ranking — the chartable bar series.
    assert!(olorin::runes::plot::is_chartable(&out), "eanet output should be chartable");
    assert_eq!(out.categories[0].name, "10.0.0.66", "scanner should top the fan-out ranking");
    // The scan anomaly's bucket equals its bar's name, so spike_flags highlights it.
    assert!(out.anomalies.iter().any(|a| a.bucket == "10.0.0.66"), "scan bucket must match its bar");
    // The report renders an inline SVG for it.
    assert!(olorin::runes::report::svg_chart(&out).is_some(), "report should render an eanet SVG chart");
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
