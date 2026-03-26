use std::env;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse flags
    let serve = args.contains(&"--serve".into());
    let interactive = args.contains(&"--interactive".into());
    let whatsapp = args.contains(&"--whatsapp".into());

    // Parse options
    let backend = get_opt(&args, "--backend").unwrap_or("auto".into());
    let model_arg = get_opt(&args, "--model");
    let port: u16 = get_opt(&args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let max_seq_len: usize = get_opt(&args, "--max-seq-len")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    // Banner
    println!("[Olorin] v0.5.0 — The Wakeful Mind in Eä");

    // Initialize Olorin home directory
    let olorin_home = home_dir().join(".olorin");
    std::fs::create_dir_all(&olorin_home).ok();
    std::fs::create_dir_all(olorin_home.join("vault")).ok();
    std::fs::create_dir_all(olorin_home.join("models")).ok();

    // Model resolution
    let model_path = resolve_model_path(model_arg.as_deref(), &olorin_home);

    // Print status
    println!("[Olorin] Home: {}", olorin_home.display());
    if let Some(ref p) = model_path {
        println!("[Olorin] Model: {}", p.display());
    }
    println!("[Olorin] Backend: {}", backend);
    println!("[Olorin] Max sequence length: {}", max_seq_len);

    // Mode dispatch
    if serve {
        println!("[Olorin] Starting web UI on port {}...", port);
    }

    if whatsapp {
        println!("[Olorin] WhatsApp bridge not yet connected.");
    }

    if interactive || (!serve && !whatsapp) {
        println!("[Olorin] Interactive mode.");
        println!("[Olorin] Ready. (REPL not yet wired — use --serve for web UI)");
    }
}

fn get_opt(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn home_dir() -> PathBuf {
    home::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"))
}

fn resolve_model_path(arg: Option<&str>, olorin_home: &std::path::Path) -> Option<PathBuf> {
    match arg {
        Some("bitnet") => {
            let p = olorin_home.join("models/ggml-model-i2_s.gguf");
            if p.exists() { Some(p) } else { None }
        }
        Some("llama") => {
            let p = olorin_home.join("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
            if p.exists() { Some(p) } else { None }
        }
        Some(path) => {
            let p = PathBuf::from(path);
            if p.exists() { Some(p) } else { None }
        }
        None => {
            let paths = [
                olorin_home.join("models/ggml-model-i2_s.gguf"),
                home_dir().join(".cougar/models/ggml-model-i2_s.gguf"),
            ];
            paths.into_iter().find(|p| p.exists())
        }
    }
}
