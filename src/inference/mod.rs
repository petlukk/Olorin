pub mod engine;
mod engine_helpers;
pub mod gguf;
pub mod tokenizer;
pub mod cache;
pub mod matmul;
pub mod dequant;
pub mod generate;
pub mod threadpool;
pub mod forward;
mod forward_attn;
mod forward_attn_heads;

// Re-exports for integration tests that verify batched attention
// against looped single-query attention before the production
// wire-up in Task 19.
pub use forward_attn_heads::{attention_decode, attention_decode_batch};
