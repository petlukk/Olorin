# Olorin installer (Windows). Downloads the latest release binary, optionally
# configures the Anthropic cloud fallback key, and optionally installs the
# WhatsApp bridge.
#
#   iwr -useb https://raw.githubusercontent.com/petlukk/Olorin/main/scripts/install.ps1 | iex
#
# Honour these env vars to skip the prompts:
#   $env:OLORIN_VERSION       — release tag to install (default: latest)
#   $env:OLORIN_INSTALL_DIR   — install location (default: %LOCALAPPDATA%\Olorin\bin)
#   $env:OLORIN_WITH_BRIDGE   — "yes" / "no" to skip the bridge prompt
#   $env:OLORIN_WITH_PATH     — "yes" / "no" to skip the PATH-update prompt
#   $env:ANTHROPIC_API_KEY    — if already set, the prompt is skipped

$ErrorActionPreference = 'Stop'

$Repo = 'petlukk/Olorin'
$InstallDir = if ($env:OLORIN_INSTALL_DIR) { $env:OLORIN_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Olorin\bin' }
$OlorinHome = Join-Path $env:USERPROFILE '.olorin'
$EnvFile    = Join-Path $OlorinHome 'env'

function Step($n, $msg) { Write-Host "`n[$n] $msg" }
function Info($msg)     { Write-Host "  $msg" }
function Fail($msg)     { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

function AskYN($prompt, $default) {
    $ans = Read-Host $prompt
    if ([string]::IsNullOrWhiteSpace($ans)) { return $default }
    if ($ans -match '^(y|yes)$') { return 'yes' } else { return 'no' }
}

Write-Host 'Olorin installer'

# 1. Platform detection. Windows on x86_64 only for now.
if (-not [System.Environment]::Is64BitOperatingSystem) {
    Fail '32-bit Windows is not supported.'
}
$target = 'windows-x86_64'
Info "Platform: $target"

# 2. Resolve release tag.
if ($env:OLORIN_VERSION) {
    $tag = $env:OLORIN_VERSION
} else {
    Step 1 'Resolving latest release'
    $rel = Invoke-RestMethod -UseBasicParsing "https://api.github.com/repos/$Repo/releases/latest"
    $tag = $rel.tag_name
    if (-not $tag) { Fail 'Could not resolve latest release tag from GitHub.' }
}
Info "Version: $tag"

# 3. Download olorin.exe.
$base = "https://github.com/$Repo/releases/download/$tag"
$binaryName = "olorin-$target.exe"
Step 2 'Downloading olorin'
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("olorin-install-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$dlPath = Join-Path $tmp 'olorin.exe'
try {
    Invoke-WebRequest -UseBasicParsing -Uri "$base/$binaryName" -OutFile $dlPath
} catch {
    Fail "Failed to download $base/$binaryName : $_"
}

# 4. Verify checksum if SHA256SUMS is published.
try {
    $sumsPath = Join-Path $tmp 'SHA256SUMS'
    Invoke-WebRequest -UseBasicParsing -Uri "$base/SHA256SUMS" -OutFile $sumsPath -ErrorAction Stop
    $expectedLine = (Get-Content $sumsPath) | Where-Object { $_ -match "  $([regex]::Escape($binaryName))$" } | Select-Object -First 1
    if ($expectedLine) {
        $expected = ($expectedLine -split '\s+')[0].ToLower()
        $actual = (Get-FileHash -Algorithm SHA256 $dlPath).Hash.ToLower()
        if ($expected -ne $actual) { Fail "Checksum mismatch for $binaryName" }
        Info 'Checksum verified'
    }
} catch {
    # SHA256SUMS missing is not fatal.
}

# 5. Install.
$dest = Join-Path $InstallDir 'olorin.exe'
Move-Item -Force $dlPath $dest
Info "Installed: $dest"

# 6. Cloud fallback.
New-Item -ItemType Directory -Force -Path $OlorinHome | Out-Null
if ($env:ANTHROPIC_API_KEY) {
    Info 'ANTHROPIC_API_KEY already set in environment — skipping prompt.'
} else {
    Step 3 'Cloud fallback (optional)'
    Write-Host '  Olorin runs a local Gemma 4 model by default. If you also want'
    Write-Host '  Anthropic Claude as a cloud fallback (used when no local model'
    Write-Host '  is loaded), enter your API key. Leave blank to skip.'
    $key = Read-Host '  ANTHROPIC_API_KEY (blank to skip)'
    if (-not [string]::IsNullOrWhiteSpace($key)) {
        $alreadyHas = $false
        if (Test-Path $EnvFile) {
            $alreadyHas = (Select-String -Path $EnvFile -Pattern '^ANTHROPIC_API_KEY=' -Quiet)
        }
        if ($alreadyHas) {
            Info "ANTHROPIC_API_KEY already in $EnvFile — leaving it untouched."
        } else {
            Add-Content -Path $EnvFile -Value "ANTHROPIC_API_KEY=$key"
            Info "Wrote $EnvFile. Olorin reads this at startup."
        }
    }
}

# 7. Bridge.
Step 4 'WhatsApp gateway (optional)'
Write-Host '  /teleport launches a WhatsApp bridge subprocess (~15 MB).'
Write-Host '  Skip this if you only use the terminal REPL and web UI.'
$wantBridge = $env:OLORIN_WITH_BRIDGE
if (-not $wantBridge) { $wantBridge = AskYN '  Install bridge? [y/N]' 'no' }
if ($wantBridge -eq 'yes') {
    $bridgeName = "wa-bridge-$target.exe"
    $bridgeDest = Join-Path $InstallDir 'wa-bridge.exe'
    try {
        Invoke-WebRequest -UseBasicParsing -Uri "$base/$bridgeName" -OutFile $bridgeDest
        Info "Installed: $bridgeDest"
    } catch {
        Info "Bridge binary not in release $tag — skipping. Build from source if needed."
    }
}

# 8. PATH (User scope).
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -and ($userPath.Split(';') -contains $InstallDir)) {
    Info "PATH already includes $InstallDir."
} else {
    Step 5 'User PATH'
    $wantPath = $env:OLORIN_WITH_PATH
    if (-not $wantPath) { $wantPath = AskYN "  Add $InstallDir to User PATH? [Y/n]" 'yes' }
    if ($wantPath -eq 'yes') {
        $newPath = if ($userPath) { "$userPath;$InstallDir" } else { $InstallDir }
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Info "Updated User PATH (restart your shell to pick it up)."
    }
}

# Clean up tmp dir.
Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue

# 9. Quickstart.
Write-Host ''
Write-Host 'Done. Try one of:'
Write-Host '  olorin                 # terminal REPL'
Write-Host '  olorin --serve         # web UI on http://127.0.0.1:8080'
Write-Host '  olorin --strict        # deterministic dispatch only, no LLM (~25 ms)'
Write-Host ''
Write-Host "On first run you'll be prompted to set a vault passphrase."
Write-Host "Docs: https://github.com/$Repo"
