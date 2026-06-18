//! Palantír — ambient log watcher. v1 member: **logwatch**, a foreground
//! trigger-during-lag early-warning watcher.
//!
//! `olorin palantir --alert <file>` tails a log and, when a deploy-class trigger
//! line appears, alerts that an error cascade is likely incoming — before the
//! errors hit — using a lag learned from the file's own history. See
//! `watch.rs` for the detector and the honest scope (it predicts the
//! trigger+lag class of incidents, not precursor-free instant crashes).

pub mod sink;
pub mod tail;
pub mod watch;

use crate::runes::stream::{self, Format};
use sink::Sink;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::Duration;
use watch::{Detector, RateDetector, Sensitivity};

/// Cap on the history read for the learn pass: the recent tail is the relevant
/// part and bounds memory on a multi-GB log.
const LEARN_CAP: u64 = 16 * 1024 * 1024;

pub struct Opts {
    pub path:        String,
    pub sensitivity: Sensitivity,
    pub poll_secs:   u64,
    pub learn:       bool,
    pub sinks:       Vec<Sink>,
}

/// Parse `palantir` args. Returns Err(usage) on a bad invocation.
pub fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut path: Option<String> = None;
    let mut sensitivity = Sensitivity::Medium;
    let mut poll_secs = 1u64;
    let mut learn = true;
    let mut sinks: Vec<Sink> = Vec::new();
    let mut it = args.iter();
    while let Some(tok) = it.next() {
        match tok.as_str() {
            "--alert" => path = Some(it.next().ok_or("missing path after --alert")?.clone()),
            "--sensitivity" => {
                let v = it.next().ok_or("missing value after --sensitivity")?;
                sensitivity = Sensitivity::parse(v)
                    .ok_or(format!("unknown --sensitivity: {v} (expected low|med|high)"))?;
            }
            "--poll" => {
                let v = it.next().ok_or("missing value after --poll")?;
                poll_secs = v.parse().map_err(|_| format!("bad --poll seconds: {v}"))?;
                if poll_secs == 0 { return Err("--poll must be >= 1".to_string()); }
            }
            "--notify" => {
                let v = it.next().ok_or("missing value after --notify")?;
                sinks.push(Sink::parse(v)?);
            }
            "--no-learn" => learn = false,
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    let path = path.ok_or("missing --alert <file>")?;
    if sinks.is_empty() {
        sinks.push(Sink::Stdout); // default sink
    }
    Ok(Opts { path, sensitivity, poll_secs, learn, sinks })
}

pub const USAGE: &str =
    "usage: olorin palantir --alert <file> [--sensitivity low|med|high] [--poll SECS]\n         \
     [--notify stdout|webhook:URL|exec:CMD]... [--no-learn]\n  \
     e.g. olorin palantir --alert /app/log/system.log --notify webhook:https://hooks.slack.com/…";

/// Run the foreground watcher. Diverges: loops until the process is killed
/// (Ctrl-C). Kernel init happens here, as for the rune/report subcommands.
pub fn run(opts: Opts) -> ! {
    crate::kernels::ffi::init().expect("kernel init failed");

    // Learn pass over existing history: pick the format and a trigger→error lag.
    let history = read_tail(Path::new(&opts.path));
    let mut fmt: Option<Format> = if history.is_empty() { None } else { Some(stream::detect_format(&history)) };
    let lag = match (opts.learn, fmt) {
        (true, Some(f)) => watch::learn_lag(&history, f),
        _ => None,
    };
    let mut detector = Detector::new(lag, opts.sensitivity);

    eprintln!(
        "[palantír] watching {}  format={}  lag={}  sensitivity={:?}  window={}s  sinks={}  (Ctrl-C to stop)",
        opts.path,
        fmt.map(Format::tag).unwrap_or("pending"),
        lag.map(|l| format!("~{l}s")).unwrap_or_else(|| "unknown (no history pattern)".to_string()),
        opts.sensitivity,
        detector.window(),
        opts.sinks.len(),
    );

    let mut rate = RateDetector::new(opts.sensitivity);
    let mut tailer = tail::Tailer::at_end(&opts.path);
    loop {
        std::thread::sleep(Duration::from_secs(opts.poll_secs));
        let now = now_epoch();
        let lines = tailer.poll();
        let (triggers, errors) = if lines.is_empty() {
            (0, 0) // empty tick still drives window expiry and the rate baseline
        } else {
            let chunk = lines.join("\n");
            let f = *fmt.get_or_insert_with(|| stream::detect_format(chunk.as_bytes()));
            watch::classify_chunk(chunk.as_bytes(), f)
        };

        let mut alerts = detector.observe(now, triggers, errors);
        // Rate-anomaly runs every tick to keep its baseline current, but its
        // alert is suppressed while a trigger incident is active so one incident
        // isn't reported by both detectors.
        if let Some(a) = rate.observe(now, errors as u64) {
            if !detector.is_active() {
                alerts.push(a);
            }
        }

        for a in alerts {
            for s in &opts.sinks {
                s.deliver(&a);
            }
        }
    }
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Read up to the last `LEARN_CAP` bytes of a file, dropping a partial first
/// line so the history starts on a line boundary. Empty on missing/unreadable.
fn read_tail(path: &Path) -> Vec<u8> {
    let Ok(mut f) = File::open(path) else { return Vec::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let start = len.saturating_sub(LEARN_CAP);
    if start > 0 && f.seek(SeekFrom::Start(start)).is_err() {
        return Vec::new();
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return Vec::new();
    }
    if start > 0 {
        if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
            buf.drain(0..=nl);
        }
    }
    buf
}
