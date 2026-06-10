//! Shell command classifier — gates destructive commands before execution.
//!
//! Three policy modes:
//! - `open`: no restrictions
//! - `safe`: block destructive, allow writes
//! - `strict`: only read-only commands

/// Policy mode for shell command execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShellPolicy {
    Open,
    Safe,
    Strict,
}

/// Classification result for a shell command.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommandRisk {
    Allow,
    Write,
    Destructive,
}

/// Classifies shell commands by risk level.
pub struct ShellGuard {
    policy: ShellPolicy,
}

impl ShellGuard {
    pub fn new(policy: ShellPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> ShellPolicy {
        self.policy
    }

    /// Returns Ok(()) if allowed, Err(reason) if blocked.
    pub fn check(&self, command: &str) -> Result<(), String> {
        // Path-textual deny — fires regardless of policy mode (even Open).
        // The destructive-command classifier lets `cat` through because cat
        // is read-only by its model; but `cat ~/.olorin/vault.bin` exfiltrates
        // the user's encrypted vault to the attacker. The path_guard pattern
        // can't be reused directly because the shell command may contain
        // multiple paths embedded in flags, redirections, pipes, etc. —
        // we conservatively textual-match the sensitive substrings instead.
        if let Some(label) = contains_sensitive_path(command) {
            return Err(format!(
                "blocked: command references sensitive path ({label}). \
                 shell tool refuses these in every policy mode."
            ));
        }
        if self.policy == ShellPolicy::Open {
            return Ok(());
        }
        let risk = classify(command);
        match (self.policy, risk) {
            (ShellPolicy::Open, _) => Ok(()),
            (_, CommandRisk::Allow) => Ok(()),
            (ShellPolicy::Safe, CommandRisk::Write) => Ok(()),
            (ShellPolicy::Strict, CommandRisk::Write) => {
                Err("blocked (strict mode): write operation. Set OLORIN_SHELL_POLICY=safe to allow.".to_string())
            }
            (_, CommandRisk::Destructive) => {
                Err("blocked: destructive command. Set OLORIN_SHELL_POLICY=open to override.".to_string())
            }
        }
    }

    pub fn classify(&self, command: &str) -> CommandRisk {
        classify(command)
    }
}

/// Substrings that disqualify any shell command from running, no matter
/// the policy mode. Mirrors `core/path_guard.rs::SENSITIVE_RELATIVE` +
/// SENSITIVE_ABSOLUTE but textual — applied to the literal command
/// string the tool receives. False positives on innocent echoes are
/// tolerable; the LLM has no reason to mention these paths during
/// legitimate work.
const SHELL_FORBIDDEN_PATHS: &[(&str, &str)] = &[
    (".olorin/",          "Olorin vault directory"),
    (".ssh/",             "SSH keys"),
    (".aws/credentials",  "AWS credentials"),
    (".aws/config",       "AWS config"),
    (".gnupg/",           "GPG keys"),
    (".config/anthropic", "Anthropic API config"),
    (".config/openai",    "OpenAI API config"),
    (".bashrc",           "shell rc file"),
    (".zshrc",            "shell rc file"),
    (".profile",          "shell profile"),
    (".bash_history",     "shell history"),
    (".zsh_history",      "shell history"),
    ("/etc/shadow",       "system password file"),
    ("/etc/sudoers",      "sudo config"),
];

fn contains_sensitive_path(cmd: &str) -> Option<&'static str> {
    // Match against the raw command AND a quote-stripped copy, so shell
    // quote-splitting (`cat ~/.s""sh/id_rsa` → `~/.ssh/id_rsa` after the
    // shell removes the empty quotes) can't slip a needle past the literal
    // substring match. False positives on innocent echoes stay tolerable.
    let unquoted: String = cmd.chars().filter(|&c| c != '"' && c != '\'').collect();
    for &(needle, label) in SHELL_FORBIDDEN_PATHS {
        if cmd.contains(needle) || unquoted.contains(needle) {
            return Some(label);
        }
    }
    None
}

const ALLOW_COMMANDS: &[&str] = &[
    "ls", "cat", "head", "tail", "less", "more", "file", "stat",
    "grep", "rg", "ag", "awk", "sed", "sort", "uniq", "wc", "tr", "cut", "tee",
    "find", "which", "whereis", "type", "locate",
    "ps", "top", "htop", "df", "du", "free", "uptime", "uname", "lscpu",
    "echo", "printf", "date", "whoami", "hostname", "id", "groups",
    "pwd", "env", "printenv", "set",
    "git", "cargo", "python", "python3", "node", "ruby", "go", "java",
    "curl", "wget", "dig", "nslookup", "ping", "traceroute",
    "jq", "yq", "xmllint", "md5sum", "sha256sum", "base64",
    "diff", "cmp", "comm",
    "man", "help", "info",
    "test", "[",
    "true", "false",
];

const WRITE_COMMANDS: &[&str] = &[
    "cp", "mv", "mkdir", "rmdir", "touch", "ln",
    "chmod", "chown", "chgrp",
    "tar", "zip", "unzip", "gzip", "gunzip", "bzip2",
    "pip", "pip3", "npm", "npx", "yarn", "pnpm",
    "apt", "apt-get", "yum", "dnf", "brew", "pacman",
    "docker", "podman",
    "kill", "pkill", "killall",
    "systemctl", "service",
    "crontab",
    "ssh", "scp", "rsync",
    "make", "cmake", "ninja",
];

const DESTRUCTIVE_COMMANDS: &[&str] = &[
    "rm", "shred", "srm",
    "mkfs", "fdisk", "parted", "gdisk",
    "dd",
    "format",
    "shutdown", "reboot", "halt", "poweroff", "init",
    "wipefs",
];

const DANGEROUS_FLAGS: &[&str] = &[
    "-rf", "-fr", "--force", "--no-preserve-root",
];

const DANGEROUS_TARGETS: &[&str] = &[
    "/dev/sda", "/dev/sdb", "/dev/nvme", "/dev/vda",
    "/dev/null",
];

fn classify(command: &str) -> CommandRisk {
    let command = command.trim();
    if command.is_empty() { return CommandRisk::Allow; }

    if command.contains("(){") || command.contains("() {") {
        return CommandRisk::Destructive;
    }

    if targets_dangerous_redirect(command) {
        return CommandRisk::Destructive;
    }

    let mut worst = CommandRisk::Allow;

    // Command substitutions run their own commands — `echo $(rm -rf ~)` would
    // otherwise classify as a harmless `echo`. Classify each substitution body
    // (recursively, so nested `$()` is covered) and fold it into the verdict.
    for sub in extract_substitutions(command) {
        worst = worst_risk(worst, classify(&sub));
        if worst == CommandRisk::Destructive { return CommandRisk::Destructive; }
    }

    for segment in split_compound(command) {
        let risk = classify_single(segment.trim());
        worst = worst_risk(worst, risk);
        if worst == CommandRisk::Destructive { return CommandRisk::Destructive; }
    }
    worst
}

/// Extract the command bodies of `$(...)` and backtick substitutions so the
/// classifier can see commands the outer command would otherwise hide. `$(` is
/// matched with paren-depth tracking (handles nesting); backticks are matched
/// to the next backtick. `${VAR}` (parameter expansion, not a command) is left
/// alone.
fn extract_substitutions(cmd: &str) -> Vec<String> {
    let bytes = cmd.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() && bytes[i + 1] == b'(' {
            let start = i + 2;
            let mut depth = 1;
            let mut j = start;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'(' => depth += 1,
                    b')' => depth -= 1,
                    _ => {}
                }
                j += 1;
            }
            let end = if depth == 0 { j - 1 } else { bytes.len() };
            if end > start {
                out.push(cmd[start..end].to_string());
            }
            i = j;
        } else if bytes[i] == b'`' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'`' { j += 1; }
            if j > start {
                out.push(cmd[start..j].to_string());
            }
            i = j + 1;
        } else {
            i += 1;
        }
    }
    out
}

fn targets_dangerous_redirect(command: &str) -> bool {
    for (i, _) in command.match_indices('>') {
        let rest = command[i + 1..].trim_start();
        for &target in DANGEROUS_TARGETS {
            if rest.starts_with(target) { return true; }
        }
    }
    false
}

fn classify_single(cmd: &str) -> CommandRisk {
    if cmd.is_empty() { return CommandRisk::Allow; }

    let base = strip_prefixes(cmd);
    let parts: Vec<&str> = base.split_whitespace().collect();
    if parts.is_empty() { return CommandRisk::Allow; }

    let binary = extract_binary_name(parts[0]);
    let has_redirect = cmd.contains('>');

    if DESTRUCTIVE_COMMANDS.iter().any(|&c| binary == c || (binary.len() > c.len() && binary.as_bytes()[c.len()] == b'.' && binary.starts_with(c))) {
        if binary == "rm" && !parts.iter().any(|p| DANGEROUS_FLAGS.iter().any(|f| p == f || p.contains(f))) {
            if parts.iter().any(|p| *p == "/" || *p == "/*" || *p == "~" || *p == "~/*") {
                return CommandRisk::Destructive;
            }
            return CommandRisk::Write;
        }
        return CommandRisk::Destructive;
    }

    if WRITE_COMMANDS.iter().any(|&c| binary == c) {
        return CommandRisk::Write;
    }

    if ALLOW_COMMANDS.iter().any(|&c| binary == c) {
        if binary == "sed" && parts.iter().any(|p| *p == "-i" || p.starts_with("-i")) {
            return CommandRisk::Write;
        }
        if binary == "git" {
            if let Some(sub) = parts.get(1) {
                match *sub {
                    "push"|"reset"|"rebase"|"merge"|"commit"|"checkout"|"switch"|
                    "pull"|"fetch"|"clone"|"init"|"stash"|"cherry-pick" => return CommandRisk::Write,
                    _ => {}
                }
            }
        }
        if binary == "cargo" {
            if let Some(sub) = parts.get(1) {
                match *sub {
                    "build"|"run"|"install"|"publish"|"clean" => return CommandRisk::Write,
                    _ => {}
                }
            }
        }
        if has_redirect { return CommandRisk::Write; }
        return CommandRisk::Allow;
    }

    if has_redirect { return CommandRisk::Write; }
    CommandRisk::Write
}

fn split_compound(cmd: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut start = 0;
    let mut in_quote = false;
    let mut quote_char = b' ';
    let bytes = cmd.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let b = bytes[i];
        if in_quote {
            if b == quote_char { in_quote = false; }
        } else {
            match b {
                b'\'' | b'"' => { in_quote = true; quote_char = b; }
                // `sh -c` terminates a command at a newline just like `;`,
                // so a newline-hidden second command must be classified too.
                b';' | b'\n' | b'\r' => { segments.push(&cmd[start..i]); start = i + 1; }
                b'&' if i + 1 < bytes.len() && bytes[i + 1] == b'&' => {
                    segments.push(&cmd[start..i]); start = i + 2; i += 1;
                }
                b'|' if i + 1 < bytes.len() && bytes[i + 1] == b'|' => {
                    segments.push(&cmd[start..i]); start = i + 2; i += 1;
                }
                b'|' => { segments.push(&cmd[start..i]); start = i + 1; }
                _ => {}
            }
        }
        i += 1;
    }
    segments.push(&cmd[start..]);
    segments
}

fn strip_prefixes(cmd: &str) -> &str {
    let mut s = cmd.trim();
    loop {
        let trimmed = s.trim_start();
        if let Some(eq_pos) = trimmed.find('=') {
            let before_eq = &trimmed[..eq_pos];
            if !before_eq.is_empty() && before_eq.chars().all(|c| c.is_alphanumeric() || c == '_') {
                let after_eq = &trimmed[eq_pos + 1..];
                if let Some(sp) = find_unquoted_space(after_eq) {
                    s = after_eq[sp..].trim_start();
                    continue;
                } else {
                    return "";
                }
            }
        }
        break;
    }
    loop {
        let parts: Vec<&str> = s.splitn(2, ' ').collect();
        let binary = extract_binary_name(parts[0]);
        match binary {
            "sudo"|"nice"|"nohup"|"env"|"timeout"|"strace"|"ltrace" => {
                if let Some(rest) = parts.get(1) {
                    s = rest.trim();
                    if binary == "sudo" {
                        while s.starts_with('-') {
                            if let Some(sp) = s.find(' ') {
                                s = s[sp..].trim_start();
                            } else {
                                return s;
                            }
                        }
                    }
                    continue;
                }
                return s;
            }
            _ => break,
        }
    }
    s
}

fn extract_binary_name(s: &str) -> &str {
    s.rsplit('/').next().unwrap_or(s)
}

fn find_unquoted_space(s: &str) -> Option<usize> {
    let mut in_quote = false;
    let mut qchar = ' ';
    for (i, c) in s.char_indices() {
        if in_quote {
            if c == qchar { in_quote = false; }
        } else {
            match c {
                '\'' | '"' => { in_quote = true; qchar = c; }
                ' ' => return Some(i),
                _ => {}
            }
        }
    }
    None
}

fn worst_risk(a: CommandRisk, b: CommandRisk) -> CommandRisk {
    match (a, b) {
        (CommandRisk::Destructive, _) | (_, CommandRisk::Destructive) => CommandRisk::Destructive,
        (CommandRisk::Write, _) | (_, CommandRisk::Write) => CommandRisk::Write,
        _ => CommandRisk::Allow,
    }
}

/// Load shell policy from env var or config file.
pub fn load_shell_policy() -> ShellPolicy {
    if let Ok(val) = std::env::var("OLORIN_SHELL_POLICY") {
        return match val.to_lowercase().as_str() {
            "open" => ShellPolicy::Open,
            "safe" => ShellPolicy::Safe,
            "strict" => ShellPolicy::Strict,
            _ => {
                eprintln!("warning: unknown OLORIN_SHELL_POLICY={val}, defaulting to 'safe'");
                ShellPolicy::Safe
            }
        };
    }
    let home = crate::home_dir().unwrap_or_default();
    let path = home.join(".olorin").join("shell_policy");
    if let Ok(content) = std::fs::read_to_string(&path) {
        return match content.trim().to_lowercase().as_str() {
            "open" => ShellPolicy::Open,
            "safe" => ShellPolicy::Safe,
            "strict" => ShellPolicy::Strict,
            _ => ShellPolicy::Safe,
        };
    }
    ShellPolicy::Safe
}
