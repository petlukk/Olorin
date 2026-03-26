use crate::channel::Channel;
use crate::error::Result;
use crate::kernels::arg_tokenizer::ArgTokenizer;
use crate::kernels::command_router as cmd_router;
use std::time::Instant;

use super::{Agent, TurnTiming};

/// Build tool name and params from command ID and argument string.
/// Shared between REPL (Agent) and WhatsApp direct commands.
pub fn build_tool_params(
    cmd_id: i32,
    arg: &str,
    tokenizer: &mut ArgTokenizer,
) -> Result<(&'static str, serde_json::Value)> {
    match cmd_id {
        cmd_router::CMD_TIME => Ok(("time", serde_json::json!({}))),
        cmd_router::CMD_CALC => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool("usage: /calc <expression>".into()));
            }
            Ok(("calc", serde_json::json!({"expr": arg})))
        }
        cmd_router::CMD_HTTP => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool("usage: /http <url>".into()));
            }
            Ok(("http", serde_json::json!({"url": arg})))
        }
        cmd_router::CMD_SHELL => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool("usage: /shell <command>".into()));
            }
            Ok(("shell", serde_json::json!({"command": arg})))
        }
        cmd_router::CMD_MEMORY => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /memory <action> [key] [value]".into(),
                ));
            }
            let parts = tokenizer.tokenize_str(arg, 3);
            let action = parts.first().unwrap_or(&"");
            let key = parts.get(1).unwrap_or(&"");
            let value = parts.get(2).unwrap_or(&"");
            Ok((
                "memory",
                serde_json::json!({"action": action, "key": key, "value": value}),
            ))
        }
        cmd_router::CMD_READ => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool("usage: /read <path>".into()));
            }
            Ok(("read", serde_json::json!({"path": arg})))
        }
        cmd_router::CMD_WRITE => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /write <path> <content>".into(),
                ));
            }
            let parts = tokenizer.tokenize_str(arg, 2);
            let path = parts.first().unwrap_or(&"");
            let content = parts.get(1).unwrap_or(&"");
            Ok((
                "write",
                serde_json::json!({"path": path, "content": content}),
            ))
        }
        cmd_router::CMD_LS => {
            let path = if arg.is_empty() { "." } else { arg };
            Ok(("ls", serde_json::json!({"path": path})))
        }
        cmd_router::CMD_JSON => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /json <keys|get|pretty> <input> [path]".into(),
                ));
            }
            let parts = tokenizer.tokenize_str(arg, 3);
            let action = parts.first().unwrap_or(&"");
            let input = parts.get(1).unwrap_or(&"");
            let path = parts.get(2).unwrap_or(&"");
            let mut params = serde_json::json!({"action": action, "input": input});
            if !path.is_empty() {
                params["path"] = serde_json::json!(path);
            }
            Ok(("json", params))
        }
        cmd_router::CMD_CPU => Ok(("cpu", serde_json::json!({}))),
        cmd_router::CMD_TOKENS => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /tokens <text or file>".into(),
                ));
            }
            Ok(("tokens", serde_json::json!({"text": arg})))
        }
        cmd_router::CMD_BENCH => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /bench <safety|router>".into(),
                ));
            }
            Ok(("bench", serde_json::json!({"target": arg})))
        }
        cmd_router::CMD_WEATHER => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool("usage: /weather <city>".into()));
            }
            Ok(("weather", serde_json::json!({"city": arg})))
        }
        cmd_router::CMD_TRANSLATE => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /translate <lang> <text>".into(),
                ));
            }
            let parts = tokenizer.tokenize_str(arg, 2);
            let lang = parts.first().unwrap_or(&"");
            let text = parts.get(1).unwrap_or(&"");
            Ok(("translate", serde_json::json!({"lang": lang, "text": text})))
        }
        cmd_router::CMD_DEFINE => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool("usage: /define <word>".into()));
            }
            Ok(("define", serde_json::json!({"word": arg})))
        }
        cmd_router::CMD_SUMMARIZE => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /summarize <url>".into(),
                ));
            }
            Ok(("summarize", serde_json::json!({"url": arg})))
        }
        cmd_router::CMD_GREP => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /grep <pattern> [path]".into(),
                ));
            }
            let parts = tokenizer.tokenize_str(arg, 2);
            let pattern = parts.first().unwrap_or(&"");
            let path = parts.get(1).unwrap_or(&".");
            Ok(("grep", serde_json::json!({"pattern": pattern, "path": path})))
        }
        cmd_router::CMD_GIT => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /git <status|log|diff|branch|show|blame|stash> [args]".into(),
                ));
            }
            Ok(("git", serde_json::json!({"args": arg})))
        }
        cmd_router::CMD_REMIND => {
            if arg.is_empty() {
                return Err(crate::error::Error::Tool(
                    "usage: /remind <time> <message>".into(),
                ));
            }
            let parts = tokenizer.tokenize_str(arg, 2);
            let time = parts.first().unwrap_or(&"");
            let message = parts.get(1).unwrap_or(&"");
            Ok(("remind", serde_json::json!({"time": time, "message": message})))
        }
        _ => Err(crate::error::Error::Tool(format!(
            "unknown tool command: {cmd_id}"
        ))),
    }
}

/// Escape a string for safe use in shell commands (single-quote wrapping).
pub(crate) fn shell_escape(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

impl Agent {
    /// Execute a pipeline of tool commands separated by " | ".
    /// Output of each stage becomes the argument to the next.
    pub(crate) async fn execute_pipeline(
        &mut self,
        input: &str,
        channel: &dyn Channel,
    ) -> Result<()> {
        let stages: Vec<&str> = input.split(" | ").collect();
        if stages.len() < 2 {
            return Err(crate::error::Error::Tool("not a pipeline".into()));
        }

        let mut pipe_data: Option<String> = None;
        let mut tool_execs: Vec<(String, u64)> = Vec::new();
        let _pipeline_start = Instant::now();

        for (i, stage) in stages.iter().enumerate() {
            let stage = stage.trim();
            let (cmd_id, cmd_arg) = cmd_router::match_command_verified(stage.as_bytes());

            if cmd_id < cmd_router::CMD_TOOL_FIRST || cmd_id > cmd_router::CMD_TOOL_LAST {
                return Err(crate::error::Error::Tool(format!(
                    "stage {}: not a tool command: {stage}",
                    i + 1
                )));
            }

            let tool_name = cmd_router::command_name(cmd_id).unwrap_or("unknown");
            let arg_str = String::from_utf8_lossy(cmd_arg).into_owned();

            // Build params — for piped stages, use piped data as default arg
            let effective_arg = if pipe_data.is_some() && arg_str.is_empty() {
                "__piped__".to_string()
            } else {
                arg_str.clone()
            };
            let (name, mut params) = self.build_tool_params(cmd_id, &effective_arg)?;
            if let Some(ref piped) = pipe_data {
                self.inject_pipe_data(cmd_id, &mut params, piped, &arg_str);
            }

            let tool_start = Instant::now();
            let output = self.run_tool(name, params).await?;
            let tool_ms = tool_start.elapsed().as_millis() as u64;
            tool_execs.push((tool_name.to_string(), tool_ms));

            // Safety scan intermediate pipeline outputs
            let scan = self.safety.scan_output(&output);
            if let Some(reason) = scan.block_reason() {
                channel
                    .send(&format!("Pipeline stage {} blocked: {reason}.", i + 1))
                    .await;
                return Ok(());
            }

            pipe_data = Some(output);
        }

        // Safety scan on final output
        if let Some(ref output) = pipe_data {
            let scan = self.safety.scan_output(output);
            if let Some(reason) = scan.block_reason() {
                channel
                    .send(&format!("Pipeline output blocked: {reason}."))
                    .await;
            } else {
                channel.send(output).await;
            }
        }

        self.last_timing = Some(TurnTiming {
            safety_scan_us: 0,
            llm_call_ms: 0,
            tool_execs,
        });

        Ok(())
    }

    /// Inject piped data into tool parameters.
    /// Each tool has a primary input field that receives piped data.
    pub(crate) fn inject_pipe_data(
        &self,
        cmd_id: i32,
        params: &mut serde_json::Value,
        piped: &str,
        _arg: &str,
    ) {
        match cmd_id {
            cmd_router::CMD_CALC => {
                params["expr"] = serde_json::json!(piped);
            }
            cmd_router::CMD_SHELL => {
                if let Some(cmd) = params["command"].as_str() {
                    let wrapped = format!("echo {} | {cmd}", shell_escape(piped));
                    params["command"] = serde_json::json!(wrapped);
                }
            }
            cmd_router::CMD_HTTP => {
                params["url"] = serde_json::json!(piped.trim());
            }
            cmd_router::CMD_TOKENS => {
                params["text"] = serde_json::json!(piped);
            }
            cmd_router::CMD_JSON => {
                params["input"] = serde_json::json!(piped);
            }
            cmd_router::CMD_READ => {
                params["path"] = serde_json::json!(piped.trim());
            }
            cmd_router::CMD_WRITE => {
                params["content"] = serde_json::json!(piped);
            }
            cmd_router::CMD_MEMORY => {
                if params["action"].as_str() == Some("write") {
                    params["value"] = serde_json::json!(piped);
                } else if params["action"].as_str() == Some("read") {
                    params["key"] = serde_json::json!(piped.trim());
                }
            }
            _ => {}
        }
    }

    /// Execute a tool directly from a slash command, bypassing the LLM.
    pub(crate) async fn execute_direct_tool(
        &mut self,
        cmd_id: i32,
        arg: &str,
    ) -> Result<String> {
        let (name, params) = self.build_tool_params(cmd_id, arg)?;
        self.run_tool(name, params).await
    }

    /// Execute a streaming tool command, sending chunks directly to the channel.
    pub(crate) async fn execute_direct_tool_stream(
        &mut self,
        cmd_id: i32,
        arg: &str,
        channel: &dyn Channel,
    ) -> Result<()> {
        let (tool_name, params) = self.build_tool_params(cmd_id, arg)?;

        let tool = self
            .tools
            .get(tool_name)
            .ok_or_else(|| crate::error::Error::Tool(format!("{tool_name} tool not registered")))?;

        // Collect chunks, then leak-scan full output before sending to channel.
        let mut chunks: Vec<String> = Vec::new();
        let mut on_chunk = |chunk: &str| {
            chunks.push(chunk.to_string());
        };

        tool.execute_stream(params, &mut on_chunk).await?;

        // Safety scan the full output (injection + leak detection)
        let full_output: String = chunks.iter().map(|s| s.as_str()).collect();
        let scan = self.safety.scan_output(&full_output);
        if let Some(reason) = scan.block_reason() {
            channel
                .send(&format!("Tool output blocked: {reason}."))
                .await;
            return Ok(());
        }

        // Stream chunks to channel
        if !chunks.is_empty() {
            print!("\r\x1b[2K{} ", channel.response_prefix());
            for chunk in &chunks {
                channel.send_chunk(chunk).await;
            }
            channel.flush().await;
        }

        Ok(())
    }

    /// Build tool name and params from command ID and argument string.
    /// Delegates to the free function with this agent's tokenizer.
    pub(crate) fn build_tool_params(
        &mut self,
        cmd_id: i32,
        arg: &str,
    ) -> Result<(&'static str, serde_json::Value)> {
        build_tool_params(cmd_id, arg, &mut self.tokenizer)
    }

    pub(crate) async fn run_tool(
        &self,
        name: &str,
        params: serde_json::Value,
    ) -> Result<String> {
        match self.tools.get(name) {
            Some(tool) => tool.execute(params).await,
            None => Err(crate::error::Error::Tool(format!(
                "{name} tool not registered"
            ))),
        }
    }
}
