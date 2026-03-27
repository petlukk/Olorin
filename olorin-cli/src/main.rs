mod repl;
mod repl_commands;

use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use repl::{OlorinRepl, ReplAction};

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
        let quant: String = engine.as_ref().map(|e| e.quant_type_str().to_string()).unwrap_or("unknown".into());
        let repl = Mutex::new(OlorinRepl::new(engine.clone(), &quant));
        let web = olorin_core::channel::web::WebChannel::new(port);
        let handler = move |prompt: &str, on_token: &dyn Fn(&str)| -> String {
            let mut repl = repl.lock().unwrap();
            match repl.process(prompt) {
                ReplAction::Quit => "Goodbye.".to_string(),
                ReplAction::Print(msg) => {
                    on_token(&msg);
                    msg
                }
                ReplAction::Generate(p) => repl.generate_for_web(&p, on_token),
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

// ---------------------------------------------------------------------------
// CougarEngine — thin wrapper around cougar-engine for direct inference
// ---------------------------------------------------------------------------

pub(crate) struct CougarEngine {
    gguf: cougar_engine::gguf::GgufFile,
    max_seq_len: usize,
}

impl CougarEngine {
    pub fn quant_type_str(&self) -> &str {
        let idx = self.gguf.tensor_map.get("blk.0.attn_q.weight")
            .or_else(|| self.gguf.tensor_map.get("blk.0.attn_qkv.weight"));
        match idx {
            Some(&i) => match self.gguf.tensors[i].dtype {
                36 => "I2S",
                12 | 14 => "Q4K",
                _ => "unknown",
            },
            None => "unknown",
        }
    }

    pub fn generate_text(&self, prompt: &str, on_token: &dyn Fn(&str)) -> String {
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

        use cougar_engine::model::QuantType;
        let is_q4k = model.quant_type == QuantType::Q4K;

        // Chat template depends on model family:
        // - Llama 3: <|start_header_id|>...<|end_header_id|>
        // - Qwen/ChatML: <|im_start|>...<|im_end|>
        // - BitNet (base model): no template, raw prompt
        let mut tokens = vec![tokenizer.bos_id];
        if is_q4k {
            // Detect Qwen vs Llama by checking if tokenizer knows ChatML tokens
            let has_chatml = tokenizer.token_to_id("<|im_start|>").is_some();
            let chat = if has_chatml {
                // Qwen-style ChatML
                format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n\
                     <|im_start|>assistant\n"
                )
            } else {
                // Llama 3 Instruct
                format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n\
                     {prompt}<|eot_id|>\
                     <|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            };
            tokens.extend(tokenizer.encode(&chat));
        } else {
            // BitNet: base model, no chat template
            tokens.extend(tokenizer.encode(prompt));
        }

        let tok_ref = &tokenizer;
        let output = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let output_ref = output.clone();

        let on_tok = |tok_id: u32| {
            let text = tok_ref.decode(&[tok_id]);
            on_token(&text);
            output_ref.lock().unwrap().push(tok_id);
        };

        let generated = if is_q4k {
            let (gen, _, _) = cougar_engine::forward_llama::generate(
                &model, &tokens, 256, 0.7, 1.1,
                tokenizer.eos_id, self.max_seq_len, on_tok,
            );
            gen
        } else {
            let (gen, _, _) = cougar_engine::forward::InferenceState::generate(
                &model, &tokens, 256, 0.7, 1.1,
                tokenizer.eos_id, self.max_seq_len, on_tok,
            );
            gen
        };

        let gen_tokens: Vec<u32> = generated[tokens.len()..].to_vec();
        tokenizer.decode(&gen_tokens)
    }
}

fn load_cougar_model(path: &std::path::Path, max_seq_len: usize) -> Option<CougarEngine> {
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

// ---------------------------------------------------------------------------
// REPL loop
// ---------------------------------------------------------------------------

fn run_repl(engine: Option<Arc<CougarEngine>>) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let quant: String = engine.as_ref().map(|e| e.quant_type_str().to_string()).unwrap_or("unknown".into());
    let mut repl = OlorinRepl::new(engine, &quant);

    println!("[Olorin] Type a message (Ctrl+D to exit):");
    loop {
        print!("you> ");
        stdout.lock().flush().ok();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }
                match repl.process(input) {
                    ReplAction::Quit => break,
                    ReplAction::Print(msg) => println!("olorin> {}", msg),
                    ReplAction::Generate(prompt) => {
                        print!("olorin> ");
                        stdout.lock().flush().ok();
                        repl.generate_streaming(&prompt);
                        println!();
                    }
                }
            }
            Err(_) => break,
        }
    }
    println!("[Olorin] Goodbye.");
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

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
            if p.exists() {
                Some(p)
            } else {
                None
            }
        }
        Some("llama") => {
            let p = olorin_home.join("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        }
        Some(path) => {
            let p = PathBuf::from(path);
            if p.exists() {
                Some(p)
            } else {
                None
            }
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
