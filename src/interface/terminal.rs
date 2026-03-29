//! Terminal REPL — synchronous blocking I/O, no rustyline.
//!
//! Reads lines from stdin, dispatches through the Olorin Pipe, prints response.
//! Handles /quit, Ctrl-C (SIGINT), and all slash commands via DispatchContext.

use std::io::{self, BufRead, Write};
use crate::core::router::{DispatchContext, Response};

// ── REPL entry point ──────────────────────────────────────────────────────────

/// Run the interactive REPL. Blocks until /quit or EOF.
pub fn run(model_arg: Option<&str>) {
    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let mut ctx = DispatchContext::new(api_key, model_arg);

    print_banner();

    let stdin = io::stdin();
    let stdout = io::stdout();

    loop {
        // Print prompt
        {
            let mut out = stdout.lock();
            let _ = out.write_all(b"olorin> ");
            let _ = out.flush();
        }

        // Read line
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                // EOF (Ctrl-D)
                println!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[olorin] read error: {e}");
                break;
            }
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        // Dispatch through the Pipe
        let resp = ctx.dispatch(input);

        // /quit — special exit case
        if resp.text == "Goodbye!" && !resp.blocked {
            println!("{}", resp.text);
            break;
        }

        print_response(&resp);
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_banner() {
    println!("[Olorin] v0.6.0 — The Wakeful Mind in Ea");
    println!("Type /help for commands, /quit to exit.");
    println!();
}

fn print_response(resp: &Response) {
    if resp.text.is_empty() {
        return;
    }
    if resp.blocked {
        eprintln!("[blocked] {}", resp.text);
    } else {
        println!("{}", resp.text);
    }
}
