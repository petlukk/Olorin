//! Palantír — ambient log watcher. v1 member: **logwatch**, a foreground
//! trigger-during-lag early-warning watcher.
//!
//! `olorin palantir --alert <file>` tails a log and, when a deploy-class trigger
//! line appears, alerts that an error cascade is likely incoming — before the
//! errors hit — using a lag learned from the file's own history. See
//! `watch.rs` for the detector and the honest scope (it predicts the
//! trigger+lag class of incidents, not precursor-free instant crashes).

pub mod daemon;
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
    pub daemon:      bool,
    pub name:        Option<String>,
    pub sinks:       Vec<Sink>,
}

/// What the invocation asks for: watch a file, or a lifecycle query/command.
pub enum Mode {
    Watch(Opts),
    Status(Option<String>),
    Stop(Option<String>),
}

/// Parse `palantir` args. Returns Err(usage) on a bad invocation.
pub fn parse_args(args: &[String]) -> Result<Mode, String> {
    let mut path: Option<String> = None;
    let mut sensitivity = Sensitivity::Medium;
    let mut poll_secs = 1u64;
    let mut learn = true;
    let mut daemon = false;
    let mut name: Option<String> = None;
    let mut sinks: Vec<Sink> = Vec::new();
    let mut lifecycle: Option<&str> = None;
    let mut it = args.iter();
    while let Some(tok) = it.next() {
        match tok.as_str() {
            "--alert" => path = Some(it.next().ok_or("missing path after --alert")?.clone()),
            "--name" => name = Some(it.next().ok_or("missing value after --name")?.clone()),
            "--status" => lifecycle = Some("status"),
            "--stop" => lifecycle = Some("stop"),
            "--daemon" => daemon = true,
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
    match lifecycle {
        Some("status") => return Ok(Mode::Status(name)),
        Some("stop") => return Ok(Mode::Stop(name)),
        _ => {}
    }
    let path = path.ok_or("missing --alert <file> (or --status / --stop)")?;
    if sinks.is_empty() {
        sinks.push(Sink::Stdout); // default sink
    }
    Ok(Mode::Watch(Opts { path, sensitivity, poll_secs, learn, daemon, name, sinks }))
}

pub const USAGE: &str =
    "usage: olorin palantir --alert <file> [--daemon] [--name NAME] [--sensitivity low|med|high]\n         \
     [--poll SECS] [--notify stdout|webhook:URL|exec:CMD]... [--no-learn]\n       \
     olorin palantir --status [--name NAME]\n       \
     olorin palantir --stop   [--name NAME]\n  \
     e.g. olorin palantir --alert /app/log/system.log --daemon --notify webhook:https://hooks.slack.com/…";

/// Dispatch a parsed mode. Diverges (watch loops forever; status/stop exit).
pub fn run(mode: Mode) -> ! {
    match mode {
        Mode::Status(name) => std::process::exit(daemon::status(name.as_deref())),
        Mode::Stop(name) => std::process::exit(daemon::stop(name.as_deref())),
        Mode::Watch(opts) => run_watch(opts),
    }
}

/// Watch a file. Foreground by default; `--daemon` detaches first. Loops until
/// the process is killed (Ctrl-C, or `--stop`).
fn run_watch(opts: Opts) -> ! {
    crate::kernels::ffi::init().expect("kernel init failed");
    let name = daemon::watcher_name(&opts.path, opts.name.as_deref());

    if let Some(pid) = daemon::already_running(&name) {
        eprintln!("[palantír] '{name}' is already watching (pid {pid}) — `--stop` it first");
        std::process::exit(1);
    }

    // Learn pass over existing history: pick the format and a trigger→error lag.
    let history = read_tail(Path::new(&opts.path));
    let mut fmt: Option<Format> = if history.is_empty() { None } else { Some(stream::detect_format(&history)) };
    let lag = match (opts.learn, fmt) {
        (true, Some(f)) => watch::learn_lag(&history, f),
        _ => None,
    };

    // Detach BEFORE the loop, while stderr still reaches the terminal — tell the
    // user where the log went, then redirect stdio into it.
    if opts.daemon {
        let log = daemon::log_path(&name);
        eprintln!("[palantír] '{name}' → background; log {}, `--status`/`--stop` to manage", log.display());
        if let Err(e) = daemon::daemonize(&log) {
            eprintln!("[palantír] daemonize failed: {e}");
            std::process::exit(1);
        }
    }
    let _ = daemon::write_pid(&name); // the (now possibly detached) daemon's pid

    let mut detector = Detector::new(lag, opts.sensitivity);
    eprintln!(
        "[palantír] '{name}' watching {}  format={}  lag={}  sensitivity={:?}  window={}s  sinks={}",
        opts.path,
        fmt.map(Format::tag).unwrap_or("pending"),
        lag.map(|l| format!("~{l}s")).unwrap_or_else(|| "unknown (no history pattern)".to_string()),
        opts.sensitivity,
        detector.window(),
        opts.sinks.len(),
    );

    // Initial snapshot so `--status` is meaningful from the first moment, before
    // any alert or heartbeat.
    let fmt0 = fmt.map(Format::tag).unwrap_or("pending");
    daemon::write_snapshot(&name, &opts.path, fmt0, lag, None, now_epoch());

    let mut rate = RateDetector::new(opts.sensitivity);
    let mut tailer = tail::Tailer::at_end(&opts.path);
    let mut last_alert: Option<watch::Alert> = None;
    let mut tick: u64 = 0;
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

        let had_alert = !alerts.is_empty();
        for a in alerts {
            for s in &opts.sinks {
                s.deliver(&a);
            }
            last_alert = Some(a);
        }

        // Snapshot on every alert, plus a heartbeat, so `--status` stays fresh.
        tick += 1;
        if had_alert || tick % 10 == 0 {
            let fmt_tag = fmt.map(Format::tag).unwrap_or("pending");
            daemon::write_snapshot(&name, &opts.path, fmt_tag, lag, last_alert.as_ref(), now);
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
