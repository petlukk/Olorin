use olorin::interface;
use olorin::kernels::ffi;

fn main() {
    olorin::config::load_env_file();

    let args: Vec<String> = std::env::args().collect();

    // One-shot rune subcommand: `olorin rune <name> [rune args…]` runs a rune
    // and writes ONLY its answer to stdout — no banner, no model load, no REPL
    // chrome — so `olorin rune eatime --bucket series --json file.log > out.json`
    // yields clean JSON for downstream tools (matplotlib, jq, …). Kernel-init
    // diagnostics go to stderr, so stdout stays pure.
    if args.get(1).map(String::as_str) == Some("rune") {
        run_rune_cli(&args[2..]); // never returns
    }

    // One-shot report subcommand: `olorin report <files…> [-o out.html]` runs
    // the deterministic file-drop pipeline (pick_rune per file + eacorrelate
    // across them) and writes one self-contained HTML report. No model.
    if args.get(1).map(String::as_str) == Some("report") {
        run_report_cli(&args[2..]); // never returns
    }

    let serve    = args.contains(&"--serve".into());
    let whatsapp = args.contains(&"--whatsapp".into());
    let strict   = args.contains(&"--strict".into());
    let model_arg = get_opt(&args, "--model");
    let audit_path = get_opt(&args, "--audit");
    let port: u16 = get_opt(&args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    println!("[Olorin] v{} — The Wakeful Mind in Ea", env!("CARGO_PKG_VERSION"));
    if strict {
        println!("[Olorin] strict mode: LLM disabled, deterministic dispatch only.");
    }
    if let Some(p) = audit_path {
        println!("[Olorin] audit: writing JSON Lines to {p}");
    }

    // Init SIMD kernels
    ffi::init().expect("kernel init failed");

    // Setup directories
    let home        = olorin::home_dir().unwrap_or_default();
    let olorin_home = home.join(".olorin");
    std::fs::create_dir_all(olorin_home.join("vault")).ok();
    std::fs::create_dir_all(olorin_home.join("models")).ok();

    if serve {
        interface::server::run(port, model_arg, strict, audit_path);
    } else if whatsapp {
        interface::whatsapp::run_whatsapp(model_arg);
    } else {
        interface::terminal::run(model_arg, strict, audit_path);
    }
}

/// Run a single rune non-interactively and print only its answer to stdout.
/// `rest` is everything after `rune` on the command line: `<name> [args…]`.
/// Exit codes: 0 success, 1 rune failure / unknown rune, 2 usage error.
fn run_rune_cli(rest: &[String]) -> ! {
    let Some(name) = rest.first() else {
        eprintln!(
            "usage: olorin rune <name> [args…]\n  \
             e.g. olorin rune eatime --bucket series --json access.log > out.json"
        );
        std::process::exit(2);
    };
    // Kernels are required for every rune; init prints only to stderr.
    ffi::init().expect("kernel init failed");
    let rune_args = rest[1..].join(" ");
    match olorin::runes::run_rune(name, &rune_args) {
        Some(result) => {
            println!("{}", result.answer);
            std::process::exit(if result.success { 0 } else { 1 });
        }
        None => {
            eprintln!("unknown rune: {name}");
            std::process::exit(1);
        }
    }
}

/// Build a self-contained HTML report from 1–8 files and write it.
/// Exit codes match `run_rune_cli`: 0 ok, 1 analysis failure, 2 usage.
fn run_report_cli(rest: &[String]) -> ! {
    let mut out_path = "olorin-report.html".to_string();
    let mut paths: Vec<String> = Vec::new();
    let mut it = rest.iter();
    while let Some(tok) = it.next() {
        if tok == "-o" || tok == "--out" {
            match it.next() {
                Some(p) => out_path = p.clone(),
                None => {
                    eprintln!("missing value after {tok}");
                    std::process::exit(2);
                }
            }
        } else {
            paths.push(tok.clone());
        }
    }
    if paths.is_empty() {
        eprintln!(
            "usage: olorin report <file> [file…] [-o out.html]\n  \
             e.g. olorin report syslog.log deploys.csv access.log -o incident.html"
        );
        std::process::exit(2);
    }
    ffi::init().expect("kernel init failed");
    let files: Vec<(String, String)> = paths
        .iter()
        .map(|p| (basename(p), p.clone()))
        .collect();
    match olorin::runes::report::build_report(&files, concat!("v", env!("CARGO_PKG_VERSION"))) {
        Ok((html, summary)) => {
            if let Err(e) = std::fs::write(&out_path, &html) {
                eprintln!("cannot write {out_path}: {e}");
                std::process::exit(1);
            }
            eprint!("{summary}");
            eprintln!("\nwrote {out_path} ({:.1} KB, self-contained)", html.len() as f64 / 1024.0);
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("report failed: {e}");
            std::process::exit(1);
        }
    }
}

fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// Return the value immediately after `flag` in args, if present.
fn get_opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}
