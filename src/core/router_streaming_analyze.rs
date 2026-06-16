//! File-drop analyst — the streaming path behind the web-UI drop gesture.
//! Extracted from `router_streaming.rs` to keep both files under the 500-line
//! cap. The user dropping a file IS the "analyze this" decision, so a rune is
//! picked deterministically (no autonomous model tool-call, which is what makes
//! this reliable on the Pi) and the model only narrates the kernel output.

use crate::core::router::{DispatchContext, StreamEvent};
use crate::core::safety;

impl DispatchContext {
    /// File-drop analyst (single file). Thin wrapper over the multi-file path.
    /// The user dropped a file, so the "analyze this" decision is already made —
    /// no autonomous model tool-call, which is what makes this reliable on the
    /// Pi. `tmp_path` MUST be under the rune path allowlist (~ or /tmp).
    pub fn analyze_file_streaming(
        &mut self,
        display_name: &str,
        tmp_path: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        self.analyze_files_streaming(&[(display_name.to_string(), tmp_path.to_string())], tx);
    }

    /// File-drop analyst (one or more files): pick + run each file's rune
    /// deterministically, stream every kernel output, then narrate — a single
    /// summary for one file, or one correlation pass across the combined
    /// compact answers for several. The correlation step is reasoning over
    /// already-computed results (i.e. narration), which works on the Pi.
    pub fn analyze_files_streaming(
        &mut self,
        files: &[(String, String)],
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        if files.is_empty() {
            let _ = tx.send(StreamEvent::Done { full_text: String::new() });
            return;
        }
        let descriptor = if files.len() == 1 {
            format!("analyze {}", files[0].0)
        } else {
            format!("analyze {} files", files.len())
        };
        self.vault_save(b"user", descriptor.as_bytes());

        let mut runs: Vec<FileRun> = Vec::new();
        let mut rune_text = String::new();
        for (i, (name, path)) in files.iter().enumerate() {
            if i > 0 {
                let _ = tx.send(StreamEvent::Token("\n\n".to_string()));
                rune_text.push_str("\n\n");
            }
            if let Some(run) = self.run_file_rune(name, path, tx) {
                rune_text.push_str(&run.streamed);
                runs.push(run);
            }
        }

        // No rune produced output (all skipped/blocked) — nothing to narrate.
        if runs.is_empty() {
            let _ = tx.send(StreamEvent::Done { full_text: rune_text });
            return;
        }

        // One file → single-rune narration; several → one correlation pass.
        if runs.len() == 1 {
            let r = runs.pop().unwrap();
            let scratch = crate::runes::RuneResult {
                answer: r.answer, details: None,
                success: r.success, timing_us: r.timing_us, structured: r.structured,
            };
            match crate::runes::build_narration_prompt(r.rune, r.safety, scratch) {
                Some(prompt) => self.run_followup_streaming(&descriptor, &rune_text, &prompt, tx),
                None => { let _ = tx.send(StreamEvent::Done { full_text: rune_text }); }
            }
        } else {
            // Deterministic cross-file lag correlation (SIMD) before the
            // narration: findings stream to the user like any kernel
            // output and LEAD the narration prompt, so the model opens
            // with the conclusion instead of hunting for one across the
            // per-file summaries.
            let corr = crate::runes::eacorrelate::correlate_files(files);
            let mut findings = crate::runes::eacorrelate::findings_block(&corr);
            if let Some(block) = &findings {
                if safety::scan(block.as_bytes()).blocked {
                    findings = None;
                } else {
                    let chunk = format!(
                        "\n\n📎 ran `eacorrelate` across {} files\n\n{block}",
                        files.len(),
                    );
                    self.vault_save(b"tool", block.as_bytes());
                    let _ = tx.send(StreamEvent::Token(chunk.clone()));
                    rune_text.push_str(&chunk);
                }
            }
            // The PROMPT gets the prose rendering, not the technical
            // block the user sees — see findings_for_prompt.
            let prose = findings.and_then(|_|
                crate::runes::eacorrelate::findings_for_prompt(&corr));
            // Gemma 4 on NEON narrates 2-file prompts reliably but
            // degenerates on >= 3 full answer blocks (immediate EOS or
            // junk that slips the dump filter — Pi-verified 2026-06-11,
            // both orders, prose or block). Same playbook as the
            // aarch64 minimal chat prompt: shrink the prompt to what
            // the model handles — the prose conclusion alone — and when
            // there is no conclusion to restate, skip narration; the
            // kernel outputs stand alone.
            let digest = cfg!(target_arch = "aarch64") && runs.len() >= 3;
            if digest && prose.is_none() {
                let _ = tx.send(StreamEvent::Done { full_text: rune_text });
                return;
            }
            let prompt = if digest {
                format!(
                    "Output of analysis tools on {} files:\n\n{}\n",
                    runs.len(),
                    prose.as_deref().expect("digest narration requires findings"),
                )
            } else {
                correlation_prompt(&runs, prose.as_deref())
            };
            self.run_followup_streaming(&descriptor, &rune_text, &prompt, tx);
        }
    }

    /// Pick + run the rune for one staged file and stream its kernel output.
    /// Returns the result for the narration step, or None if no rune matched or
    /// the output was blocked (a notice is streamed in those cases).
    fn run_file_rune(
        &mut self,
        display_name: &str,
        tmp_path: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) -> Option<FileRun> {
        let prefix = read_prefix(tmp_path, 8192);
        let Some(rune) = crate::runes::select::pick_rune(display_name, &prefix) else {
            let _ = tx.send(StreamEvent::Token(format!(
                "(no rune matched {display_name} — I can analyze CSV, JSON Lines, \
                 Parquet, and log files.)"
            )));
            return None;
        };

        let name = rune.name();
        let flags = crate::runes::select::default_args(name);
        let args = if flags.is_empty() {
            tmp_path.to_string()
        } else {
            format!("{flags} {tmp_path}")
        };
        let result = rune.run(&args);
        let safety_class = rune.output_safety();
        let answer = result.answer.clone();
        let timing_us = result.timing_us;
        let structured = result.structured;

        // User-visible kernel output (mirrors handle_rune's body shape).
        let body = if structured {
            result.answer
        } else {
            let mut b = result.answer;
            if let Some(d) = result.details {
                b.push_str("\n\n---\n");
                b.push_str(&d);
            }
            b.push_str(&format!("\n[timing: {timing_us}µs]"));
            b
        };
        if safety::scan(body.as_bytes()).blocked {
            let _ = tx.send(StreamEvent::Error(
                "Analysis output blocked by safety scan.".to_string(),
            ));
            return None;
        }

        self.vault_save(b"tool", body.as_bytes());
        let header = format!("📎 ran `{name}` on {display_name}\n\n");
        let _ = tx.send(StreamEvent::Token(header.clone()));
        // A chronological series renders as a block-bar chart above the
        // text body. Presentation only — not folded into `streamed`, so it
        // never reaches the model's narration context (block glyphs would
        // just confuse it). color=false: the web chat bubble is monospace
        // but not ANSI-aware (it appends via textContent).
        if let Some(chart) = chart_for(name, &args, Some(display_name), false) {
            if !safety::scan(chart.as_bytes()).blocked {
                // Private-Use-Area sentinels (U+E000/E001) bracket the chart so
                // the web UI renders it in a dedicated line-height:1 block (block
                // bars only connect into solid columns at line-height 1; the prose
                // bubble uses 1.5). PUA chars survive JSON unescaped and never
                // occur in chart or prose text; the frontend slices them out.
                let _ = tx.send(StreamEvent::Token(format!("\u{E000}{chart}\u{E001}")));
            }
        }
        let _ = tx.send(StreamEvent::Token(body.clone()));

        Some(FileRun {
            display: display_name.to_string(),
            rune: name,
            safety: safety_class,
            answer,
            success: result.success,
            timing_us,
            structured,
            streamed: format!("{header}{body}"),
        })
    }
}

/// One file's analysis, carried from the per-file rune run to the narration
/// step. `streamed` is the header+body already sent to the client (for the
/// final `Done` full_text); `answer` is the compact summary fed to the model.
struct FileRun {
    display: String,
    rune: &'static str,
    safety: crate::runes::OutputSafety,
    answer: String,
    success: bool,
    timing_us: u64,
    structured: bool,
    streamed: String,
}

/// Build the cross-file correlation narration prompt from the compact answers.
/// DATA ONLY, no trailing instruction: the narration system prompt already asks
/// for a 1-2 sentence summary, and appending the instruction here makes Gemma 4
/// echo it back instead of answering (same trap `build_narration_prompt`
/// documents for the single-file case). `findings` must be the PROSE
/// rendering (`findings_for_prompt`), never the machine-shaped block —
/// the patterned lines bait Gemma 4 on NEON into continuing them
/// verbatim instead of summarizing (verified on the Pi 2026-06-11; x86
/// narrates either form — the same family as the grid-continuation trap).
fn correlation_prompt(runs: &[FileRun], findings: Option<&str>) -> String {
    let mut p = format!("Output of analysis tools on {} files:\n", runs.len());
    if let Some(f) = findings {
        p.push('\n');
        p.push_str(f);
        p.push('\n');
    }
    for r in runs {
        p.push_str(&format!("\n{} (via {}):\n{}\n", r.display, r.rune, r.answer));
    }
    p
}

/// Read up to `max` bytes from a file for content sniffing. Returns an empty
/// vec on any error — `pick_rune` then falls back to extension-only routing.
fn read_prefix(path: &str, max: usize) -> Vec<u8> {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return Vec::new() };
    let mut buf = vec![0u8; max];
    match f.read(&mut buf) {
        Ok(n) => { buf.truncate(n); buf }
        Err(_) => Vec::new(),
    }
}

/// Render a block-bar chart for a rune run, or `None` when it produced no
/// chronological series to plot. Re-runs the rune with `--json` (a second
/// µs-scale scan on the page-cached file) and parses the v1 `RuneOutput`
/// contract — the same JSON seam a standalone plotter consumes. `title`
/// overrides the heading (web passes the friendly filename; REPL passes
/// None and lets it fall back to the file path). `color` enables ANSI for
/// the REPL and stays off for the not-ANSI-aware web chat bubble.
///
/// Shared by the streaming (web file-drop) and non-streaming (REPL `/rune`)
/// surfaces. Only `eatime --bucket series` yields a time chart today.
pub(crate) fn chart_for(
    rune_name: &str,
    rune_args: &str,
    title: Option<&str>,
    color: bool,
) -> Option<String> {
    if rune_name != "eatime" {
        return None;
    }
    let result = crate::runes::run_rune("eatime", &format!("{rune_args} --json"))?;
    if !result.success {
        return None;
    }
    let out = crate::runes::output::RuneOutput::from_json(result.answer.as_bytes()).ok()?;
    // Chart only a real chronological series (ISO-instant labels carry a
    // 'T'), matching eatime's own series detection; skip hour/weekday
    // histograms and anything too short to read as a timeline.
    let is_series = out.categories.first().is_some_and(|c| c.name.contains('T'));
    if !is_series || out.categories.len() < 2 {
        return None;
    }
    Some(crate::runes::plot::render_series(&out, 56, 10, color, title))
}
