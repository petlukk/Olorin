//! Tool and command handling for the Olorin Pipe.
//!
//! Handles slash commands (/help, /tools, /clear, etc.), tool execution,
//! and intent-based tool dispatch. Split from router.rs for the 500-line rule.

use crate::core::dispatch;
use crate::core::handlers;
use crate::core::safety;
use crate::core::router::{DispatchContext, Response};
use std::time::Instant;

impl DispatchContext {
    // ── Meta command handling ────────────────────────────────────────────────

    pub(crate) fn handle_meta(&mut self, cmd_id: i32) -> Response {
        match cmd_id {
            dispatch::CMD_QUIT => Response::text("Goodbye!"),
            dispatch::CMD_HELP => Response::text(self.help_text()),
            dispatch::CMD_TOOLS => Response::text(
                "Available tools: time, calc, cpu, shell, http, memory, read, write, \
                 ls, json, tokens, bench, weather, translate, define, summarize, \
                 grep, git, remind"
            ),
            dispatch::CMD_CLEAR => {
                self.clear();
                Response::text("Context cleared.")
            }
            dispatch::CMD_MODEL => {
                let local = self.engine.as_ref().map(|e| e.quant_type_str()).unwrap_or("none");
                let cloud = if self.anthropic.is_some() { "Anthropic" } else { "none" };
                Response::text(format!("Local: {local}, Cloud: {cloud}"))
            }
            dispatch::CMD_PROFILE => {
                let msg = match &self.last_timing {
                    Some(t) => t.format(),
                    None => "No timing data yet.".to_string(),
                };
                Response::text(msg)
            }
            _ => Response::text(""),
        }
    }

    // ── Tool command handling ────────────────────────────────────────────────

    pub(crate) fn handle_tool_command(&mut self, cmd_id: i32, cmd_arg: &[u8]) -> Response {
        let arg_str = String::from_utf8_lossy(cmd_arg);
        let tool_start = Instant::now();

        match dispatch::build_tool_params(cmd_id, &arg_str) {
            Ok((name, params)) => {
                let result = self.execute_tool(name, &params);
                let tool_ms = tool_start.elapsed().as_millis() as u64;
                self.last_timing = Some(handlers::TurnTiming {
                    safety_scan_us: 0,
                    llm_call_ms: 0,
                    tool_execs: vec![(name.to_string(), tool_ms)],
                });
                match result {
                    Ok(output) => {
                        let scan = safety::scan(output.as_bytes());
                        if scan.blocked {
                            Response::blocked("Tool output blocked by safety scan.")
                        } else {
                            self.vault_save(b"user", arg_str.as_bytes());
                            self.vault_save(b"tool", output.as_bytes());
                            Response::text(output)
                        }
                    }
                    Err(e) => Response::text(format!("Tool error: {e}")),
                }
            }
            Err(e) => Response::text(format!("{e}")),
        }
    }

    // ── Intent execution ─────────────────────────────────────────────────────

    pub(crate) fn execute_intent(&mut self, tool_name: &str, intent: i32, arg_bytes: &[u8]) -> Response {
        let params = dispatch::intent_to_params(intent, arg_bytes);
        match self.execute_tool(tool_name, &params) {
            Ok(output) => {
                self.vault_save(b"user", arg_bytes);
                self.vault_save(b"tool", output.as_bytes());
                Response::text(output)
            }
            Err(e) => Response::text(format!("Tool error: {e}")),
        }
    }

    // ── Tool execution ───────────────────────────────────────────────────────

    pub(crate) fn execute_tool(
        &self,
        name: &str,
        params: &[(&str, String)],
    ) -> Result<String, String> {
        if name == "calc" {
            let expr = params.iter()
                .find(|(k, _)| *k == "expr")
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            return match dispatch::eval_expr(expr) {
                Ok(result) => Ok(result),
                Err(e) => Err(format!("{e}")),
            };
        }

        let args = params.iter()
            .map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        match crate::tools::run_tool(name, &args) {
            Some(r) => if r.success { Ok(r.output) } else { Err(r.output) },
            None => Err(format!("unknown tool: {name}")),
        }
    }

    // ── Help text ────────────────────────────────────────────────────────────

    pub(crate) fn help_text(&self) -> String {
        "\
Commands:
  /help    /quit    /tools   /clear   /model   /profile

Tools:
  /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu
  /memory <action> [key] [value]   /read <path>   /write <path> <content>
  /ls [path]  /json <action> <input>  /tokens <text>  /bench <target>
  /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>
  /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>
  /recall <query>

Agent: Olorin".to_string()
    }
}
