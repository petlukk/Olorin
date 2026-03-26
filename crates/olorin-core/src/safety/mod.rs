use crate::kernels::fused_safety::FusedScanner;

pub mod leak_detector;
pub mod sanitizer;
pub mod shell_guard;
pub mod validator;

/// Result of scanning content through all safety layers.
#[derive(Debug, Clone)]
pub struct ScanResult {
    pub injection_found: bool,
    pub leaks_found: bool,
    pub details: Vec<SafetyWarning>,
}

impl ScanResult {
    /// Returns true if any safety issue was found (injection or leak).
    pub fn is_blocked(&self) -> bool {
        self.injection_found || self.leaks_found
    }

    /// Human-readable reason for blocking, or None if clean.
    pub fn block_reason(&self) -> Option<&'static str> {
        match (self.injection_found, self.leaks_found) {
            (true, true) => Some("contains prompt injection and potential secrets"),
            (true, false) => Some("contains prompt injection attempt"),
            (false, true) => Some("contains potential secrets"),
            (false, false) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafetyWarning {
    pub kind: WarningKind,
    pub pattern: String,
    pub position: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WarningKind {
    Injection,
    SecretLeak,
}

/// Pre-allocated safety scanner using Eä SIMD kernels.
///
/// Reuses SIMD mask buffers across calls — zero allocations on the hot path
/// after the first call (or after `with_capacity`). The `FusedScanner` inside
/// grows its buffers once and reuses them for every subsequent scan.
pub struct SafetyLayer {
    sanitizer: sanitizer::Sanitizer,
    leak_detector: leak_detector::LeakDetector,
    scanner: FusedScanner,
}

impl SafetyLayer {
    pub fn new() -> Self {
        Self {
            sanitizer: sanitizer::Sanitizer::new(),
            leak_detector: leak_detector::LeakDetector::new(),
            scanner: FusedScanner::new(),
        }
    }

    /// Create with pre-allocated buffers for inputs up to `max_input_len` bytes.
    pub fn with_capacity(max_input_len: usize) -> Self {
        Self {
            sanitizer: sanitizer::Sanitizer::new(),
            leak_detector: leak_detector::LeakDetector::new(),
            scanner: FusedScanner::with_capacity(max_input_len),
        }
    }

    /// Scan input content for injection attempts and secret leaks.
    ///
    /// Uses the fused safety kernel for a single memory traversal:
    /// 1. scan_safety_fused (Eä) → injection + leak bitmasks in one pass
    /// 2. Rust bit-loop: tzcnt + verify injection at each set bit
    /// 3. Rust bit-loop: tzcnt + verify leak at each set bit
    ///
    /// SIMD mask buffers are reused across calls — no heap allocation.
    pub fn scan_input(&mut self, content: &str) -> ScanResult {
        let bytes = content.as_bytes();
        let mut details = Vec::new();

        // Fused SIMD scan — reuses internal buffers
        let masks = self.scanner.scan(bytes);

        let injection_warnings =
            self.sanitizer.verify_from_masks(masks.inject_masks, bytes);
        let injection_found = !injection_warnings.is_empty();
        details.extend(injection_warnings);

        let leak_warnings = self.leak_detector.verify_from_masks(masks.leak_masks, bytes);
        let leaks_found = !leak_warnings.is_empty();
        details.extend(leak_warnings);

        ScanResult {
            injection_found,
            leaks_found,
            details,
        }
    }

    /// Scan output content (tool results, LLM responses).
    ///
    /// Checks for both injection patterns and secret leaks. Tool results
    /// (especially from HTTP fetches) can contain prompt injection attempts
    /// that must be detected before feeding content back to the LLM.
    pub fn scan_output(&mut self, content: &str) -> ScanResult {
        let bytes = content.as_bytes();
        let mut details = Vec::new();

        // Fused SIMD scan — reuses internal buffers
        let masks = self.scanner.scan(bytes);

        // Check injection patterns (defends against prompt injection via tool output)
        let injection_warnings =
            self.sanitizer.verify_from_masks(masks.inject_masks, bytes);
        let injection_found = !injection_warnings.is_empty();
        details.extend(injection_warnings);

        let leak_warnings = self.leak_detector.verify_from_masks(masks.leak_masks, bytes);
        let leaks_found = !leak_warnings.is_empty();
        details.extend(leak_warnings);

        ScanResult {
            injection_found,
            leaks_found,
            details,
        }
    }
}

impl Default for SafetyLayer {
    fn default() -> Self {
        Self::new()
    }
}
