//! Integration test for the eanet flow-triage driver (step 2): build a full
//! pcap (global header + records) with a port-scan and a data-exfil buried in
//! benign traffic, run `netflow::triage`, and assert the generic rankings
//! surface both — the same property the real-capture validation relied on.

use olorin::kernels::ffi;
use olorin::runes::netflow::triage;

fn ipv4_tcp(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, payload: usize) -> Vec<u8> {
    let mut p = vec![0u8; 12]; // MACs
    p.extend_from_slice(&[0x08, 0x00]); // ethertype IPv4
    let mut ip = vec![0u8; 20];
    ip[0] = 0x45;
    ip[9] = 6; // TCP
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
    r.extend_from_slice(&[0u8; 8]); // ts, ts_usec
    r.extend_from_slice(&(pkt.len() as u32).to_le_bytes()); // incl_len
    r.extend_from_slice(&(pkt.len() as u32).to_le_bytes()); // orig_len
    r.extend_from_slice(pkt);
    r
}

fn global_header() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(&0xa1b2c3d4u32.to_le_bytes()); // native-LE classic pcap
    h.extend_from_slice(&2u16.to_le_bytes()); // version major
    h.extend_from_slice(&4u16.to_le_bytes()); // version minor
    h.extend_from_slice(&0i32.to_le_bytes()); // thiszone
    h.extend_from_slice(&0u32.to_le_bytes()); // sigfigs
    h.extend_from_slice(&65535u32.to_le_bytes()); // snaplen
    h.extend_from_slice(&1u32.to_le_bytes()); // linktype = Ethernet
    h
}

fn be(ip: [u8; 4]) -> u32 {
    u32::from_be_bytes(ip)
}

#[test]
fn triage_surfaces_scanner_and_exfil() {
    ffi::init().unwrap();

    let mut cap = global_header();
    // Exfil: 10.0.0.99 -> 203.0.113.7, 100 packets of 1500 B on the wire.
    for _ in 0..100 {
        cap.extend(record(&ipv4_tcp([10, 0, 0, 99], [203, 0, 113, 7], 51000, 443, 1446)));
    }
    // Port scan: 10.0.0.66 -> 50 distinct destinations on 445, tiny packets.
    for i in 0..50u16 {
        cap.extend(record(&ipv4_tcp([10, 0, 0, 66], [172, 16, (i >> 8) as u8, i as u8], 40000, 445, 0)));
    }
    // Benign: 5 hosts, each talking to 2 destinations.
    for h in 0..5u8 {
        for d in 0..2u8 {
            cap.extend(record(&ipv4_tcp([192, 168, 0, h], [8, 8, 8, d], 1234, 80, 40)));
        }
    }

    let mut cur = std::io::Cursor::new(cap);
    let t = triage(&mut cur).expect("triage should succeed");

    assert_eq!(t.packets, 160, "100 exfil + 50 scan + 10 benign");
    assert_eq!(t.tcp, 160);
    assert_eq!(t.udp, 0);

    // Top talker = the exfil conversation, 100 × 1500 B.
    assert_eq!(t.top_talkers[0].0, be([10, 0, 0, 99]));
    assert_eq!(t.top_talkers[0].1, be([203, 0, 113, 7]));
    assert_eq!(t.top_talkers[0].2, 100 * 1500);

    // Top fan-out = the scanner, 50 distinct destinations; benign hosts ≤ 2.
    assert_eq!(t.top_fanout[0].0, be([10, 0, 0, 66]));
    assert_eq!(t.top_fanout[0].1, 50);
    assert!(t.top_fanout[1].1 <= 2, "benign fan-out must be far below the scanner");
}

#[test]
fn triage_rejects_non_pcap() {
    ffi::init().unwrap();
    let mut cur = std::io::Cursor::new(b"this is not a pcap file at all!!".to_vec());
    let err = triage(&mut cur).unwrap_err();
    assert!(err.contains("pcap"), "expected a pcap-shaped error, got: {err}");
}

#[test]
fn triage_rejects_pcapng() {
    ffi::init().unwrap();
    // pcapng Section Header Block magic (0x0a0d0d0a) in the first 4 bytes.
    let mut data = vec![0x0a, 0x0d, 0x0d, 0x0a];
    data.extend(std::iter::repeat(0u8).take(28));
    let mut cur = std::io::Cursor::new(data);
    let err = triage(&mut cur).unwrap_err();
    assert!(err.contains("pcapng"), "expected pcapng rejection, got: {err}");
}
