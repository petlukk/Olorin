//! Kernel test for pcap_scan.ea (eanet step 1): hand-built pcap packet region,
//! assert exact IPv4 5-tuple extraction across TCP, UDP, VLAN, and that non-IP
//! (ARP) packets are skipped. The kernel takes the region AFTER the 24-byte
//! pcap global header (the rune strips it + verifies endianness/linktype).

use olorin::kernels::ffi;

/// Build an Ethernet IPv4 TCP/UDP packet (ports at L4 offset 0/2). VLAN inserts
/// one 802.1Q tag before the ethertype.
fn ipv4(proto: u8, src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16, vlan: bool) -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 12]); // dst+src MAC
    if vlan {
        p.extend_from_slice(&[0x81, 0x00, 0x00, 0x64]); // 802.1Q, vid 100
    }
    p.extend_from_slice(&[0x08, 0x00]); // ethertype IPv4
    let mut ip = vec![0u8; 20];
    ip[0] = 0x45; // version 4, IHL 5
    ip[9] = proto;
    ip[12..16].copy_from_slice(&src);
    ip[16..20].copy_from_slice(&dst);
    p.extend_from_slice(&ip);
    let mut l4 = vec![0u8; if proto == 6 { 20 } else { 8 }];
    l4[0..2].copy_from_slice(&sport.to_be_bytes());
    l4[2..4].copy_from_slice(&dport.to_be_bytes());
    p.extend_from_slice(&l4);
    p
}

fn arp() -> Vec<u8> {
    let mut p = Vec::new();
    p.extend_from_slice(&[0u8; 12]);
    p.extend_from_slice(&[0x08, 0x06]); // ethertype ARP
    p.extend_from_slice(&[0u8; 28]);
    p
}

/// Wrap a packet in a classic-pcap record header (ts=0, incl=orig=len, LE).
fn record(pkt: &[u8]) -> Vec<u8> {
    let mut r = Vec::new();
    r.extend_from_slice(&[0u8; 8]); // ts_sec, ts_usec
    r.extend_from_slice(&(pkt.len() as u32).to_le_bytes()); // incl_len
    r.extend_from_slice(&(pkt.len() as u32).to_le_bytes()); // orig_len
    r.extend_from_slice(pkt);
    r
}

fn be(ip: [u8; 4]) -> u32 {
    u32::from_be_bytes(ip)
}

#[test]
fn pcap_scan_extracts_5tuples_and_skips_non_ip() {
    ffi::init().unwrap();

    let r1 = ipv4(6, [10, 0, 0, 1], [10, 0, 0, 2], 1234, 80, false); // TCP
    let r2 = ipv4(17, [192, 168, 1, 1], [8, 8, 8, 8], 5000, 53, false); // UDP
    let r3 = arp(); // skipped
    let r4 = ipv4(6, [172, 16, 0, 9], [172, 16, 0, 1], 44321, 445, true); // VLAN+TCP

    let mut region = Vec::new();
    for pkt in [&r1, &r2, &r3, &r4] {
        region.extend_from_slice(&record(pkt));
    }

    let max = 16usize;
    let mut out = vec![0i32; 6 * max];
    let mut n = 0i32;
    let mut consumed = 0i32;
    unsafe {
        ffi::pcap_scan(region.as_ptr(), region.len() as i32, out.as_mut_ptr(), max as i32, &mut n, &mut consumed);
    }

    // ARP skipped → exactly 3 records, in capture order (TCP, UDP, VLAN+TCP).
    assert_eq!(n, 3, "expected 3 IPv4 TCP/UDP records (ARP skipped)");
    // Whole region walked: consumed reaches the end (no straddling record).
    assert_eq!(consumed, region.len() as i32, "full walk should consume the region");

    let rec = |k: usize| -> (i32, u32, u32, i32, i32, i32) {
        let b = k * 6;
        (out[b], out[b + 1] as u32, out[b + 2] as u32, out[b + 3], out[b + 4], out[b + 5])
    };

    assert_eq!(rec(0), (6, be([10, 0, 0, 1]), be([10, 0, 0, 2]), 1234, 80, r1.len() as i32));
    assert_eq!(rec(1), (17, be([192, 168, 1, 1]), be([8, 8, 8, 8]), 5000, 53, r2.len() as i32));
    // VLAN tag transparently parsed; 5-tuple recovered from the inner IPv4 header.
    assert_eq!(rec(2), (6, be([172, 16, 0, 9]), be([172, 16, 0, 1]), 44321, 445, r4.len() as i32));
}

#[test]
fn pcap_scan_clamps_at_capacity() {
    ffi::init().unwrap();
    let pkt = ipv4(6, [1, 1, 1, 1], [2, 2, 2, 2], 1, 2, false);
    let mut region = Vec::new();
    for _ in 0..10 {
        region.extend_from_slice(&record(&pkt));
    }
    let max = 4usize;
    let mut out = vec![0i32; 6 * max];
    let mut n = 0i32;
    let mut consumed = 0i32;
    unsafe {
        ffi::pcap_scan(region.as_ptr(), region.len() as i32, out.as_mut_ptr(), max as i32, &mut n, &mut consumed);
    }
    assert_eq!(n, 4, "kernel must clamp record count at max_records");
    // Out filled at record 4 → consumed stops before the 5th record (resume point).
    let one = record(&pkt).len() as i32;
    assert_eq!(consumed, one * 4, "consumed must stop at the first un-emitted record");
}
