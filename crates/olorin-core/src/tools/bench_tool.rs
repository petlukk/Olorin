use super::Tool;
use async_trait::async_trait;
use std::time::Instant;

pub struct BenchTool;

#[async_trait]
impl Tool for BenchTool {
    fn name(&self) -> &str {
        "bench"
    }

    fn description(&self) -> &str {
        "Run a quick microbenchmark. Targets: safety, router."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["safety", "router"],
                    "description": "What to benchmark"
                }
            },
            "required": ["target"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let target = params["target"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'target' parameter".into()))?;

        match target {
            "safety" => bench_safety(),
            "router" => bench_router(),
            _ => Err(crate::error::Error::Tool(format!(
                "unknown target '{target}'"
            ))),
        }
    }
}

fn bench_safety() -> crate::error::Result<String> {
    use crate::safety::SafetyLayer;

    let input = "Hello, this is a normal user message for benchmarking purposes. ".repeat(16);
    let iterations = 10_000;

    let mut safety = SafetyLayer::with_capacity(input.len());

    // Warmup
    for _ in 0..100 {
        let _ = safety.scan_input(&input);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = safety.scan_input(&input);
    }
    let elapsed = start.elapsed();

    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let bytes_per_sec = input.len() as f64 * iterations as f64 / elapsed.as_secs_f64();
    let gb_per_sec = bytes_per_sec / 1e9;

    Ok(format!(
        "Safety scan ({} B input, {} iterations):\n  Per call: {:.0} ns\n  Throughput: {:.2} GB/s\n  Total: {:.1} ms",
        input.len(),
        iterations,
        per_call_ns,
        gb_per_sec,
        elapsed.as_secs_f64() * 1000.0,
    ))
}

fn bench_router() -> crate::error::Result<String> {
    use crate::kernels::command_router;

    let commands: &[&[u8]] = &[
        b"/help", b"/quit", b"/time", b"/calc 2+2", b"/shell ls",
        b"/read file.txt", b"/ls", b"/cpu", b"/json keys {}",
        b"hello world", b"/unknown", b"/tokens test",
    ];
    let iterations = 100_000;

    // Warmup
    for _ in 0..1000 {
        for cmd in commands {
            let _ = command_router::match_command_verified(cmd);
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        for cmd in commands {
            let _ = command_router::match_command_verified(cmd);
        }
    }
    let elapsed = start.elapsed();

    let total_calls = iterations * commands.len();
    let per_call_ns = elapsed.as_nanos() as f64 / total_calls as f64;

    Ok(format!(
        "Command router ({} commands × {} iterations):\n  Per call: {:.0} ns\n  Total calls: {}\n  Total: {:.1} ms",
        commands.len(),
        iterations,
        per_call_ns,
        total_calls,
        elapsed.as_secs_f64() * 1000.0,
    ))
}
