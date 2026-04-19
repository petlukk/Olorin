//! Runes — SIMD-backed tool calls. Each rune lives in its own file and
//! is auto-discovered by build.rs; the generated registry is included
//! below.

pub mod common;

/// Output safety classification for a rune's `answer` field.
/// Declared per-rune, never computed at runtime.
pub enum OutputSafety {
    /// Numeric/aggregate only; safe to inline in LLM context as-is.
    Trusted,
    /// Contains file-derived text; must be wrapped in
    /// `<rune_output untrusted="true">...</rune_output>` before reaching
    /// the model.
    UntrustedQuoted,
}

/// Result returned from every rune invocation.
pub struct RuneResult {
    /// Compact summary → eventually fed to the LLM.
    pub answer:    String,
    /// Verbose output → REPL/web UI only, never to the LLM.
    pub details:   Option<String>,
    pub success:   bool,
    /// Always populated, even on refusal/error — this is the flex.
    pub timing_us: u64,
}

/// The contract every rune implements.
pub trait Rune: Sync {
    fn name(&self)          -> &'static str;
    fn description(&self)   -> &'static str;
    fn usage(&self)         -> &'static str;
    fn output_safety(&self) -> OutputSafety;
    fn run(&self, args: &str) -> RuneResult;
}

/// Registry generated at build time — placeholder until build.rs
/// task lands. Replaced in Task 3 by `include!(...)`.
pub const RUNES: &[&(dyn Rune + Sync)] = &[];

/// Look up and invoke a rune by name. Returns None if not found.
pub fn run_rune(name: &str, args: &str) -> Option<RuneResult> {
    RUNES.iter()
        .find(|r| r.name() == name)
        .map(|r| r.run(args))
}
