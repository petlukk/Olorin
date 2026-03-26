pub mod gateway;
pub mod repl;
pub mod types;
pub mod wa_loop;
pub mod whatsapp;

use async_trait::async_trait;

/// Channel trait for input/output.
#[async_trait]
pub trait Channel: Send + Sync {
    fn name(&self) -> &str;
    async fn recv(&self) -> Option<String>;
    async fn send(&self, content: &str);

    /// Prefix printed before the first chunk of a streamed response.
    fn response_prefix(&self) -> &str {
        ""
    }

    /// Send a partial chunk of streaming output (no trailing newline).
    async fn send_chunk(&self, chunk: &str) {
        // Default: just print inline
        print!("{chunk}");
    }

    /// Signal end of a streamed response.
    async fn flush(&self) {
        println!("\n");
    }
}
