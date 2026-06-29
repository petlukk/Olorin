//! pcap → flow-triage aggregation for the `eanet` rune.
//!
//! Reads the 24-byte pcap global header, then drives the `pcap_scan` kernel
//! across fixed-size chunks — carrying the trailing partial record into the
//! next chunk via the kernel's `consumed` offset — and aggregates the emitted
//! `(proto, src, dst, sport, dport, bytes)` records into top-talker / fan-out /
//! fan-in rankings.
//!
//! The kernel does the per-byte parse (zero scalar-Rust fallback); this module
//! does only the hash bookkeeping — the same split as `timestamp_scan` →
//! `stream.rs`. The flow hash is the measured bottleneck (memory-bound), kept
//! in Rust for v1; a SIMD batched-hash kernel is a possible future optimisation.

use std::collections::{HashMap, HashSet};
use std::io::Read;

use crate::kernels::ffi;

const TOP_N: usize = 5;
const CHUNK: usize = 16 << 20; // 16 MiB read window
const REC_INTS: usize = 6; // i32s the kernel emits per record
const OUT_RECORDS: usize = CHUNK / 40; // > max emittable records per chunk

/// One ranked entry: an address (network-order bits) and its metric.
pub type Ranked = (u32, u64);

#[derive(Debug)]
pub struct Triage {
    pub packets: u64,
    pub bytes: u64,
    pub conversations: usize, // distinct (src, dst) pairs
    pub tcp: u64,
    pub udp: u64,
    pub top_talkers: Vec<(u32, u32, u64)>, // (src, dst, bytes)
    pub top_fanout: Vec<Ranked>,           // (src, distinct destinations)
    pub top_fanin: Vec<Ranked>,            // (dst, distinct sources)
    /// Robust baselines: the median across all hosts/pairs EXCLUDING the single
    /// top entry, so a lone scanner or exfil flow never sets its own baseline
    /// (which would make its ratio ~1 and hide it on small captures).
    pub fanout_median: u64,
    pub talker_median: u64,
}

/// Median of `v` after dropping its single largest element. The candidate being
/// scored is always the max, so excluding it keeps the baseline honest even when
/// there are only a handful of hosts.
fn median_excluding_max(mut v: Vec<u64>) -> u64 {
    if v.len() <= 1 {
        return 0;
    }
    v.sort_unstable();
    v.pop(); // drop the max (the candidate)
    v[v.len() / 2]
}

#[derive(Default)]
struct Agg {
    packets: u64,
    bytes: u64,
    tcp: u64,
    udp: u64,
    pair_bytes: HashMap<(u32, u32), u64>,
    fanout: HashMap<u32, HashSet<u32>>,
    fanin: HashMap<u32, HashSet<u32>>,
}

impl Agg {
    #[inline]
    fn record(&mut self, proto: u32, src: u32, dst: u32, bytes: u64) {
        self.packets += 1;
        self.bytes += bytes;
        if proto == 6 {
            self.tcp += 1;
        } else if proto == 17 {
            self.udp += 1;
        }
        *self.pair_bytes.entry((src, dst)).or_insert(0) += bytes;
        self.fanout.entry(src).or_default().insert(dst);
        self.fanin.entry(dst).or_default().insert(src);
    }

    fn finish(self) -> Triage {
        let fanout_median = median_excluding_max(self.fanout.values().map(|s| s.len() as u64).collect());
        let talker_median = median_excluding_max(self.pair_bytes.values().copied().collect());

        // Rankings break ties on the address(es) so the order is deterministic
        // — same output every run AND bit-identical across arches (HashMap
        // iteration order is neither). Without this the cross-arch golden and
        // the chart's bar order would flap on tied counts.
        let mut talkers: Vec<(u32, u32, u64)> =
            self.pair_bytes.iter().map(|(&(s, d), &b)| (s, d, b)).collect();
        talkers.sort_unstable_by(|a, b| b.2.cmp(&a.2).then((a.0, a.1).cmp(&(b.0, b.1))));
        talkers.truncate(TOP_N);

        let mut fanout: Vec<Ranked> =
            self.fanout.iter().map(|(&s, set)| (s, set.len() as u64)).collect();
        fanout.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        fanout.truncate(TOP_N);

        let mut fanin: Vec<Ranked> =
            self.fanin.iter().map(|(&d, set)| (d, set.len() as u64)).collect();
        fanin.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        fanin.truncate(TOP_N);

        Triage {
            packets: self.packets,
            bytes: self.bytes,
            conversations: self.pair_bytes.len(),
            tcp: self.tcp,
            udp: self.udp,
            top_talkers: talkers,
            top_fanout: fanout,
            top_fanin: fanin,
            fanout_median,
            talker_median,
        }
    }
}

/// Validate a 24-byte classic-pcap global header. v1 accepts native-endian
/// Ethernet captures; everything else returns a user-facing reason.
fn check_global_header(h: &[u8]) -> Result<(), String> {
    if h.len() < 24 {
        return Err("not a pcap file (header too short)".into());
    }
    let magic = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
    match magic {
        0xa1b2_c3d4 | 0xa1b2_3c4d => {} // microsecond / nanosecond, native-endian
        0xd4c3_b2a1 | 0x4d3c_b2a1 => {
            return Err("big-endian pcap not supported yet (v1 reads native-endian captures)".into())
        }
        0x0a0d_0d0a => {
            return Err("pcapng not supported yet — convert with `editcap -F pcap <in> <out>`".into())
        }
        _ => return Err("not a pcap file (bad magic number)".into()),
    }
    let linktype = u32::from_le_bytes([h[20], h[21], h[22], h[23]]);
    if linktype != 1 {
        return Err(format!("unsupported link type {linktype} (v1 handles Ethernet only)"));
    }
    Ok(())
}

fn drain(agg: &mut Agg, out: &[i32], n_rec: usize) {
    for k in 0..n_rec {
        let b = k * REC_INTS;
        agg.record(out[b] as u32, out[b + 1] as u32, out[b + 2] as u32, out[b + 5] as u32 as u64);
    }
}

/// Triage a pcap stream: header validation + chunked kernel scan + aggregation.
pub fn triage<R: Read>(reader: &mut R) -> Result<Triage, String> {
    let mut hdr = [0u8; 24];
    read_exact(reader, &mut hdr)?;
    check_global_header(&hdr)?;

    let mut agg = Agg::default();
    let mut buf = vec![0u8; CHUNK];
    let mut out = vec![0i32; OUT_RECORDS * REC_INTS];
    let mut filled = 0usize;

    loop {
        // Top up the window (after any carried partial record) up to CHUNK.
        let mut eof = false;
        while filled < CHUNK {
            let n = reader.read(&mut buf[filled..CHUNK]).map_err(|e| e.to_string())?;
            if n == 0 {
                eof = true;
                break;
            }
            filled += n;
        }
        if filled == 0 {
            break;
        }

        // Drain the window. The kernel may stop early (output full or a record
        // straddling the window end); `consumed` is always a record boundary.
        let mut pos = 0usize;
        while pos < filled {
            let mut n_rec = 0i32;
            let mut consumed = 0i32;
            unsafe {
                ffi::pcap_scan(
                    buf[pos..filled].as_ptr(),
                    (filled - pos) as i32,
                    out.as_mut_ptr(),
                    OUT_RECORDS as i32,
                    &mut n_rec,
                    &mut consumed,
                );
            }
            drain(&mut agg, &out, n_rec as usize);
            if consumed == 0 {
                break; // record straddles the window end — needs more bytes
            }
            pos += consumed as usize;
        }

        // Carry the unconsumed tail to the front; a truncated final record at
        // EOF is correctly dropped here.
        buf.copy_within(pos..filled, 0);
        filled -= pos;
        if eof {
            break;
        }
    }

    Ok(agg.finish())
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<(), String> {
    let mut got = 0;
    while got < buf.len() {
        let n = r.read(&mut buf[got..]).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("truncated pcap (incomplete global header)".into());
        }
        got += n;
    }
    Ok(())
}
