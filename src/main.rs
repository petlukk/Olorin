use olorin::interface;
use olorin::kernels::ffi;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let serve    = args.contains(&"--serve".into());
    let whatsapp = args.contains(&"--whatsapp".into());
    let model_arg = get_opt(&args, "--model");
    let draft_arg = get_opt(&args, "--draft");
    let draft_k: Option<usize> = get_opt(&args, "--draft-k")
        .and_then(|s| s.parse().ok());
    let port: u16 = get_opt(&args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);

    println!("[Olorin] v0.6.0 — The Wakeful Mind in Ea");

    // Init SIMD kernels
    ffi::init().expect("kernel init failed");

    // Setup directories
    let home        = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_default());
    let olorin_home = home.join(".olorin");
    std::fs::create_dir_all(olorin_home.join("vault")).ok();
    std::fs::create_dir_all(olorin_home.join("models")).ok();

    if serve {
        interface::server::run(port, model_arg, draft_arg, draft_k);
    } else if whatsapp {
        interface::whatsapp::run_whatsapp(model_arg, draft_arg, draft_k);
    } else {
        interface::terminal::run(model_arg, draft_arg, draft_k);
    }
}

/// Return the value immediately after `flag` in args, if present.
fn get_opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}
