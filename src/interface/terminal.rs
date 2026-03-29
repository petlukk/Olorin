//! Terminal REPL — synchronous blocking I/O, no rustyline.
//!
//! Reads lines from stdin, dispatches through the Olorin Pipe, prints response.
//! Handles /quit, Ctrl-C (SIGINT), and all slash commands via DispatchContext.

use std::io::{self, BufRead, Write};
use crate::core::router::{DispatchContext, Response};

// ── REPL entry point ──────────────────────────────────────────────────────────

/// Run the interactive REPL. Blocks until /quit or EOF.
pub fn run() {
    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let mut ctx = DispatchContext::new(api_key);

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_help_non_empty() {
        let mut ctx = DispatchContext::new(None);
        let r = ctx.dispatch("/help");
        assert!(!r.text.is_empty());
        assert!(!r.blocked);
    }

    #[test]
    fn test_dispatch_quit_returns_goodbye() {
        let mut ctx = DispatchContext::new(None);
        let r = ctx.dispatch("/quit");
        assert_eq!(r.text, "Goodbye!");
    }

    #[test]
    fn test_dispatch_empty_noop() {
        let mut ctx = DispatchContext::new(None);
        let r = ctx.dispatch("");
        assert_eq!(r.text, "");
        assert!(!r.blocked);
    }

    #[test]
    fn test_dispatch_unknown_slash_cmd() {
        let mut ctx = DispatchContext::new(None);
        let r = ctx.dispatch("/notacommand");
        assert!(r.text.contains("Unknown command"));
    }

    #[test]
    fn test_dispatch_clear() {
        let mut ctx = DispatchContext::new(None);
        let r = ctx.dispatch("/clear");
        assert_eq!(r.text, "Context cleared.");
    }

    #[test]
    fn test_dispatch_time_returns_something() {
        let mut ctx = DispatchContext::new(None);
        let r = ctx.dispatch("/time");
        assert!(!r.text.is_empty());
    }
}
