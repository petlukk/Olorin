use super::Channel;
use async_trait::async_trait;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

/// REPL channel using rustyline for line editing.
pub struct ReplChannel {
    rx: Mutex<mpsc::Receiver<String>>,
    shutdown: Arc<AtomicBool>,
    response_prefix: String,
    input_prompt: String,
}

impl ReplChannel {
    /// Create a new REPL channel. Spawns a blocking thread for readline.
    pub fn new(agent_name: &str) -> Self {
        // Bold green "You>" prompt where the user types
        let prompt = "\x1b[1;32mYou>\x1b[0m ".to_string();
        let response_prefix = format!("\x1b[1;36m{agent_name}>\x1b[0m");
        let (tx, rx) = mpsc::channel(1);
        let shutdown = Arc::new(AtomicBool::new(false));

        let readline_prompt = prompt.clone();
        let thread_shutdown = shutdown.clone();
        std::thread::spawn(move || {
            let mut rl = match DefaultEditor::new() {
                Ok(rl) => rl,
                Err(e) => {
                    eprintln!("Failed to initialize readline: {e}");
                    return;
                }
            };

            loop {
                match rl.readline(&readline_prompt) {
                    Ok(line) => {
                        let trimmed = line.trim().to_string();
                        if trimmed.is_empty() {
                            continue;
                        }
                        let _ = rl.add_history_entry(&trimmed);
                        if tx.blocking_send(trimmed).is_err() {
                            break;
                        }
                    }
                    Err(ReadlineError::Interrupted | ReadlineError::Eof) => {
                        let _ = tx.blocking_send("/quit".to_string());
                        break;
                    }
                    Err(e) => {
                        eprintln!("Readline error: {e}");
                        break;
                    }
                }

                if thread_shutdown.load(Ordering::Relaxed) {
                    break;
                }
            }
            // rl is dropped here, restoring terminal state
        });

        let input_prompt = "\x1b[1;32mYou>\x1b[0m ".to_string();
        Self {
            rx: Mutex::new(rx),
            shutdown,
            response_prefix,
            input_prompt,
        }
    }

    /// Signal the readline thread to stop after its current line,
    /// then restore terminal state so the parent shell works normally.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Restore cooked mode with echo. Rustyline manages termios
        // directly, so crossterm's disable_raw_mode isn't sufficient.
        // Reset via libc to guarantee echo + canonical mode are on.
        unsafe {
            let mut termios: libc::termios = std::mem::zeroed();
            if libc::tcgetattr(libc::STDIN_FILENO, &mut termios) == 0 {
                termios.c_lflag |= libc::ECHO | libc::ICANON | libc::ISIG | libc::IEXTEN;
                termios.c_iflag |= libc::ICRNL;
                termios.c_oflag |= libc::OPOST;
                libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios);
            }
        }
    }
}

#[async_trait]
impl Channel for ReplChannel {
    fn name(&self) -> &str {
        "repl"
    }

    fn response_prefix(&self) -> &str {
        &self.response_prefix
    }

    async fn recv(&self) -> Option<String> {
        let mut rx = self.rx.lock().await;
        rx.recv().await
    }

    async fn send(&self, content: &str) {
        // Clear the You> prompt that rustyline already drew, print response,
        // then reprint the input prompt since rustyline is already blocking.
        print!("\r\x1b[2K{} {content}\n\n{}", self.response_prefix, self.input_prompt);
        let _ = std::io::stdout().flush();
    }

    async fn send_chunk(&self, chunk: &str) {
        print!("{chunk}");
        let _ = std::io::stdout().flush();
    }

    async fn flush(&self) {
        print!("\n\n{}", self.input_prompt);
        let _ = std::io::stdout().flush();
    }
}
