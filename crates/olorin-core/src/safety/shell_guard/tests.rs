use super::*;

// --- Classification tests ---

#[test]
fn test_ls_is_allow() {
    assert_eq!(classify("ls -la"), CommandRisk::Allow);
}

#[test]
fn test_cat_is_allow() {
    assert_eq!(classify("cat /etc/hosts"), CommandRisk::Allow);
}

#[test]
fn test_grep_is_allow() {
    assert_eq!(classify("grep -r TODO src/"), CommandRisk::Allow);
}

#[test]
fn test_git_log_is_allow() {
    assert_eq!(classify("git log --oneline -10"), CommandRisk::Allow);
}

#[test]
fn test_git_status_is_allow() {
    assert_eq!(classify("git status"), CommandRisk::Allow);
}

#[test]
fn test_git_push_is_write() {
    assert_eq!(classify("git push origin main"), CommandRisk::Write);
}

#[test]
fn test_git_commit_is_write() {
    assert_eq!(classify("git commit -m 'test'"), CommandRisk::Write);
}

#[test]
fn test_cp_is_write() {
    assert_eq!(classify("cp file1 file2"), CommandRisk::Write);
}

#[test]
fn test_mv_is_write() {
    assert_eq!(classify("mv old new"), CommandRisk::Write);
}

#[test]
fn test_mkdir_is_write() {
    assert_eq!(classify("mkdir -p /tmp/test"), CommandRisk::Write);
}

#[test]
fn test_rm_single_file_is_write() {
    assert_eq!(classify("rm file.txt"), CommandRisk::Write);
}

#[test]
fn test_rm_rf_is_destructive() {
    assert_eq!(classify("rm -rf /tmp/something"), CommandRisk::Destructive);
}

#[test]
fn test_rm_rf_root_is_destructive() {
    assert_eq!(classify("rm -rf /"), CommandRisk::Destructive);
}

#[test]
fn test_dd_is_destructive() {
    assert_eq!(classify("dd if=/dev/zero of=/dev/sda"), CommandRisk::Destructive);
}

#[test]
fn test_mkfs_is_destructive() {
    assert_eq!(classify("mkfs.ext4 /dev/sda1"), CommandRisk::Destructive);
}

#[test]
fn test_fork_bomb_is_destructive() {
    assert_eq!(classify(":(){ :|:& };:"), CommandRisk::Destructive);
}

#[test]
fn test_redirect_to_dev_sda_is_destructive() {
    assert_eq!(classify("echo hi > /dev/sda"), CommandRisk::Destructive);
}

#[test]
fn test_sudo_rm_rf_is_destructive() {
    assert_eq!(classify("sudo rm -rf /"), CommandRisk::Destructive);
}

#[test]
fn test_pipe_with_destructive_is_destructive() {
    assert_eq!(classify("cat file | rm -rf /tmp"), CommandRisk::Destructive);
}

#[test]
fn test_and_chain_with_destructive() {
    assert_eq!(classify("ls && rm -rf /"), CommandRisk::Destructive);
}

#[test]
fn test_sed_i_is_write() {
    assert_eq!(classify("sed -i 's/old/new/g' file"), CommandRisk::Write);
}

#[test]
fn test_echo_with_redirect_is_write() {
    assert_eq!(classify("echo hello > file.txt"), CommandRisk::Write);
}

#[test]
fn test_env_var_prefix_stripped() {
    assert_eq!(classify("FOO=bar ls -la"), CommandRisk::Allow);
}

#[test]
fn test_env_var_prefix_with_destructive() {
    assert_eq!(classify("FOO=bar rm -rf /"), CommandRisk::Destructive);
}

#[test]
fn test_shutdown_is_destructive() {
    assert_eq!(classify("shutdown -h now"), CommandRisk::Destructive);
}

#[test]
fn test_empty_is_allow() {
    assert_eq!(classify(""), CommandRisk::Allow);
}

#[test]
fn test_eastat_is_allow() {
    assert_eq!(classify("eastat data.csv --json"), CommandRisk::Allow);
}

#[test]
fn test_cargo_test_is_allow() {
    assert_eq!(classify("cargo test"), CommandRisk::Allow);
}

#[test]
fn test_cargo_build_is_write() {
    assert_eq!(classify("cargo build --release"), CommandRisk::Write);
}

#[test]
fn test_unknown_command_is_write() {
    assert_eq!(classify("some_random_script"), CommandRisk::Write);
}

// --- Policy tests ---

#[test]
fn test_open_allows_everything() {
    let guard = ShellGuard::new(ShellPolicy::Open);
    assert!(guard.check("rm -rf /").is_ok());
}

#[test]
fn test_safe_blocks_destructive() {
    let guard = ShellGuard::new(ShellPolicy::Safe);
    assert!(guard.check("rm -rf /").is_err());
}

#[test]
fn test_safe_allows_write() {
    let guard = ShellGuard::new(ShellPolicy::Safe);
    assert!(guard.check("cp file1 file2").is_ok());
}

#[test]
fn test_safe_allows_read() {
    let guard = ShellGuard::new(ShellPolicy::Safe);
    assert!(guard.check("ls -la").is_ok());
}

#[test]
fn test_strict_blocks_write() {
    let guard = ShellGuard::new(ShellPolicy::Strict);
    assert!(guard.check("cp file1 file2").is_err());
}

#[test]
fn test_strict_allows_read() {
    let guard = ShellGuard::new(ShellPolicy::Strict);
    assert!(guard.check("ls -la").is_ok());
}

#[test]
fn test_strict_blocks_destructive() {
    let guard = ShellGuard::new(ShellPolicy::Strict);
    assert!(guard.check("rm -rf /").is_err());
}
