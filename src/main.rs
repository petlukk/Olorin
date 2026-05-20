use olorin::interface;
use olorin::kernels::ffi;

fn main() {
    olorin::config::load_env_file();

    let args: Vec<String> = std::env::args().collect();

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

/// Return the value immediately after `flag` in args, if present.
fn get_opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}
