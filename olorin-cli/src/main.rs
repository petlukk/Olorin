use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
mod embedded_kernels {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let serve = args.contains(&"--serve".into());
    let interactive = args.contains(&"--interactive".into());
    let whatsapp = args.contains(&"--whatsapp".into());

    let backend = get_opt(&args, "--backend").unwrap_or("auto".into());
    let model_arg = get_opt(&args, "--model");
    let port: u16 = get_opt(&args, "--port")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let max_seq_len: usize = get_opt(&args, "--max-seq-len")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2048);

    println!("[Olorin] v0.5.0 — The Wakeful Mind in Eä");

    let olorin_home = home_dir().join(".olorin");
    std::fs::create_dir_all(&olorin_home).ok();
    std::fs::create_dir_all(olorin_home.join("vault")).ok();
    std::fs::create_dir_all(olorin_home.join("models")).ok();

    let model_path = resolve_model_path(model_arg.as_deref(), &olorin_home);

    println!("[Olorin] Home: {}", olorin_home.display());
    println!("[Olorin] Backend: {}", backend);
    println!("[Olorin] Max sequence length: {}", max_seq_len);

    // Load model if available and backend is local/auto
    let engine = if backend != "cloud" {
        match model_path {
            Some(ref p) => {
                println!("[Olorin] Loading model: {}", p.display());
                match load_cougar_model(p, max_seq_len) {
                    Some(e) => {
                        println!("[Olorin] Model loaded. Ready for inference.");
                        Some(Arc::new(e))
                    }
                    None => {
                        eprintln!("[Olorin] Model loading failed. Cloud-only mode.");
                        None
                    }
                }
            }
            None => {
                println!("[Olorin] No model found. Cloud-only mode.");
                None
            }
        }
    } else {
        println!("[Olorin] Cloud-only backend selected.");
        None
    };

    if serve {
        println!("[Olorin] Starting web UI on port {}...", port);
        let web = olorin_core::channel::web::WebChannel::new(port);
        let engine_ref = engine.clone();
        let handler = move |prompt: &str, on_token: &dyn Fn(&str)| -> String {
            match engine_ref {
                Some(ref eng) => eng.generate_text(prompt, on_token),
                None => {
                    let msg = "[Olorin] No local model loaded. Set --model or use --backend cloud.";
                    on_token(msg);
                    msg.to_string()
                }
            }
        };
        if let Err(e) = web.run(handler) {
            eprintln!("[Olorin] Web server error: {}", e);
        }
        return;
    }

    if whatsapp {
        println!("[Olorin] WhatsApp bridge not yet connected.");
    }

    if interactive || !whatsapp {
        println!("[Olorin] Interactive mode.");
        run_repl(engine);
    }
}

struct CougarEngine {
    gguf: cougar_engine::gguf::GgufFile,
    max_seq_len: usize,
}

impl CougarEngine {
    fn generate_text(&self, prompt: &str, on_token: &dyn Fn(&str)) -> String {
        let model = match cougar_engine::model::BitNetModel::from_gguf(&self.gguf) {
            Ok(m) => m,
            Err(e) => {
                let msg = format!("[Olorin] Model error: {}", e);
                on_token(&msg);
                return msg;
            }
        };
        let tokenizer = match cougar_engine::tokenizer::Tokenizer::from_gguf(&self.gguf) {
            Ok(t) => t,
            Err(e) => {
                let msg = format!("[Olorin] Tokenizer error: {}", e);
                on_token(&msg);
                return msg;
            }
        };

        let chat_prompt = format!(
            "<|im_start|>system\nYou are Olorin, a helpful AI assistant.\n<|im_end|>\n\
             <|im_start|>user\n{}\n<|im_end|>\n\
             <|im_start|>assistant\n",
            prompt
        );
        let tokens = tokenizer.encode(&chat_prompt);
        let output = Arc::new(Mutex::new(Vec::<u32>::new()));
        let output_ref = output.clone();
        let tok_ref = &tokenizer;

        let (generated, _prefill_ms, _decode_ms) =
            cougar_engine::forward::InferenceState::generate(
                &model,
                &tokens,
                256,
                0.7,
                1.1,
                tokenizer.eos_id,
                self.max_seq_len,
                |tok_id| {
                    let text = tok_ref.decode(&[tok_id]);
                    on_token(&text);
                    output_ref.lock().unwrap().push(tok_id);
                },
            );

        let gen_tokens: Vec<u32> = generated[tokens.len()..].to_vec();
        tokenizer.decode(&gen_tokens)
    }
}

fn load_cougar_model(path: &std::path::Path, max_seq_len: usize) -> Option<CougarEngine> {
    // Initialize Cougar kernels
    let lib_dir = home_dir().join(".olorin/lib");
    std::fs::create_dir_all(&lib_dir).ok();
    if let Err(e) = cougar_engine::embed::extract(&lib_dir) {
        eprintln!("[Olorin] Kernel extraction warning: {}", e);
    }

    let path_str = path.to_str().unwrap_or("");
    let gguf = match cougar_engine::gguf::GgufFile::open(path_str) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[Olorin] Failed to open GGUF: {}", e);
            return None;
        }
    };

    Some(CougarEngine { gguf, max_seq_len })
}

fn run_repl(engine: Option<Arc<CougarEngine>>) {
    use std::io::{self, BufRead, Write};
    use olorin_core::kernels::command_router as cmd_router;

    let stdin = io::stdin();
    let stdout = io::stdout();

    println!("[Olorin] Type a message (Ctrl+D to exit):");
    loop {
        print!("you> ");
        stdout.lock().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let input = line.trim();
                if input.is_empty() { continue; }

                // Route through SIMD command router
                let (cmd_id, cmd_arg) = cmd_router::match_command_verified(input.as_bytes());

                if cmd_id == cmd_router::CMD_QUIT {
                    break;
                }

                if cmd_id == cmd_router::CMD_HELP {
                    println!("olorin> Commands:");
                    println!("  /help     — this message");
                    println!("  /model    — show/switch backend");
                    println!("  /teleport — session handoff");
                    println!("  /time     — current time");
                    println!("  /shell    — run shell command");
                    println!("  /quit     — exit");
                    println!("  (any text) — chat with Cougar");
                    continue;
                }

                if cmd_id == cmd_router::CMD_MODEL {
                    let arg = std::str::from_utf8(cmd_arg).unwrap_or("").trim();
                    match arg {
                        "local" => println!("olorin> Backend: local (Cougar)"),
                        "cloud" => println!("olorin> Backend: cloud (Anthropic)"),
                        "auto" => println!("olorin> Backend: auto"),
                        "" => {
                            println!("olorin> Backend: auto");
                            if engine.is_some() {
                                println!("  Local: Cougar BitNet 2B (loaded)");
                            } else {
                                println!("  Local: no model loaded");
                            }
                        }
                        other => println!("olorin> Unknown backend '{}'. Use: local|cloud|auto", other),
                    }
                    continue;
                }

                if cmd_id == cmd_router::CMD_TELEPORT {
                    let target = std::str::from_utf8(cmd_arg).unwrap_or("").trim();
                    if target.is_empty() {
                        println!("olorin> Usage: /teleport <whatsapp|web>");
                    } else {
                        println!("olorin> Teleporting to {}...", target);
                    }
                    continue;
                }

                if cmd_id == cmd_router::CMD_TIME {
                    println!("olorin> {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"));
                    continue;
                }

                if cmd_id == cmd_router::CMD_SHELL {
                    let shell_cmd = std::str::from_utf8(cmd_arg).unwrap_or("").trim();
                    if shell_cmd.is_empty() {
                        println!("olorin> Usage: /shell <command>");
                    } else {
                        match std::process::Command::new("sh").arg("-c").arg(shell_cmd).output() {
                            Ok(out) => {
                                let text = String::from_utf8_lossy(&out.stdout);
                                let err = String::from_utf8_lossy(&out.stderr);
                                if !text.is_empty() { print!("{}", text); }
                                if !err.is_empty() { eprint!("{}", err); }
                            }
                            Err(e) => println!("olorin> Shell error: {}", e),
                        }
                    }
                    continue;
                }

                // Known command but not handled in REPL
                if cmd_id != cmd_router::CMD_NONE {
                    let name = cmd_router::command_name(cmd_id).unwrap_or("unknown");
                    println!("olorin> /{} — not available in REPL mode", name);
                    continue;
                }

                // Not a command — send to inference
                print!("olorin> ");
                stdout.lock().flush().ok();
                match &engine {
                    Some(eng) => {
                        eng.generate_text(input, &|tok| {
                            print!("{}", tok);
                            stdout.lock().flush().ok();
                        });
                        println!();
                    }
                    None => println!("No local model loaded."),
                }
            }
            Err(_) => break,
        }
    }
    println!("[Olorin] Goodbye.");
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
