# Terminal Kernel Pipeline Design

SIMD-accelererad interaktiv terminal i Olorins web-UI via Eä-kärnor.

## Bakgrund

Olorin1 saknar idag interaktiv terminal. `exec.rs` kör one-shot kommandon via `fork()+pipe()`. Web-UI:t har ett Hyprland-inspirerat tile-system med chat- och repl-tiles. Denna design lägger till en tredje tile-typ (`term`) med full PTY-session, där tunga delar av terminal-pipelinen körs i SIMD-kärnor.

## Krav

- Interaktiv shell (bash) i webbläsaren via PTY
- Canvas-baserad rendering (snabbast, ingen DOM-reflow)
- SSE ner + POST upp (zero dependencies, befintligt mönster)
- En oberoende PTY-session per tile
- Dynamisk storlek baserad på tile-dimensioner
- ANSI grundnivå först (SGR, CUP, ED, EL), designat för utökning
- Två Eä-kärnor: `ansi_parser.ea` (byte-classifier), `terminal_diff.ea` (cell-grid diff)

## Arkitektur

### Pipeline

```
PTY (bash) → read(master_fd) → ansi_parser.ea → Rust state machine → terminal_diff.ea → SSE → Canvas
```

Eä-kärnorna hanterar de parallelliserbara delarna (bulk byte-scanning, grid-jämförelse). Rust äger all stateful logik (ANSI-tolkning, cursor, attribut).

### Varför inte tre kärnor?

Ursprungsplanen hade tre kärnor (ansi_parser, terminal_diff, json_envelope). `json_envelope.ea` ströks — patcharna från diff-kärnan är typiskt 50-500 bytes, och Rust kan escapa dem snabbare än FFI-overheaden kostar. ANSI-parsning är fundamentalt stateful, så kärnan gör enbart branchless byte-klassificering medan Rust driver state machine.

## PTY-lager (`interface/pty.rs`)

### PtySession

```rust
pub struct PtySession {
    master_fd: RawFd,
    child_pid: pid_t,
    cols: u16,
    rows: u16,
    cell_grid: Vec<Cell>,
    prev_grid: Vec<Cell>,
    cursor: (u16, u16),
    attrs: CellAttrs,
    parse_state: ParseState,
    scan_buf: Vec<u8>,
}
```

### Cell-representation

```rust
#[repr(C)]
struct Cell {
    ch: u32,       // Unicode codepoint
    fg: u32,       // 0x00RRGGBB (truecolor-ready)
    bg: u32,       // 0x00RRGGBB
    flags: u8,     // bold|italic|underline|inverse|dim
    _pad: [u8; 3], // alignment to 16 bytes
}
```

16 bytes per cell — exakt en `u8x16`. En 256×64 terminal = 256K, ryms i L2.

### Livscykel

1. `PtySession::new(cols, rows)` — `openpty()` → `fork()` → child: `setsid()` + `ioctl(TIOCSCTTY)` + `execvp("bash")`
2. `session.read_and_apply()` — `read(master_fd)` → `ansi_parser.ea` → Rust state update → `terminal_diff.ea` → patch
3. `session.write(bytes)` — `write(master_fd, bytes)` (tangentinput från POST)
4. `session.resize(cols, rows)` — `ioctl(master_fd, TIOCSWINSZ)` + `kill(child_pid, SIGWINCH)` + re-allokera grid
5. `Drop` — `kill(child_pid, SIGTERM)` + `close(master_fd)`

## ANSI Scanner-kärna (`kernels/ansi_parser.ea`)

Ren byte-classifier. Ingen state, inget tolkande.

### Interface

- **Input:** `data: *u8`, `len: i32`
- **Output:** `classes: *mut u8 [cap: len]`

### Klassificerings-schema

| Värde | Betydelse |
|-------|-----------|
| 0 | Printable (vanligt tecken) |
| 1 | ESC (0x1B) |
| 2 | CSI-introducer (`[` efter ESC) |
| 3 | Siffra (0x30–0x39) |
| 4 | Semikolon (0x3B) |
| 5 | Final byte (0x40–0x7E) |
| 6 | Control (0x00–0x1A exkl. ESC) |
| 7 | High byte (0x80+) |

### SIMD-strategi

Per 32-byte chunk: load `u8x32`, parallella range-checks mot varje klass, blend ihop med prioritet, store. 32 bytes klassificerade per cykel, helt branchless.

**Notering:** Klass 2 (CSI-introducer) kräver kontext — `[` är bara en CSI-introducer om den följer ESC. Kärnan klassificerar `[` som printable (klass 0). Rust state machine hanterar övergången: om `parse_state == Escape` och nästa byte är `[` → CSI-mode. Detta håller kärnan stateless.

## Diff-kärna (`kernels/terminal_diff.ea`)

Jämför gamla och nya cell-grids, producerar dirty-bitmap.

### Interface

- **Input:** `old_grid: *Cell`, `new_grid: *Cell`, `len: i32` (antal celler)
- **Output:** `dirty: *mut u8 [cap: len]` — 1 om ändrad, 0 annars

### SIMD-strategi

Varje Cell = 16 bytes = en `u8x16`. Per cell: load old, load new, XOR, movemask → dirty bit. Med `u8x32` jämförs 2 celler per op. Worst case 80×24 = 30 KB, ryms i L1d.

### Varför dirty-bitmap istället för patch-lista?

En flat array är SIMD-producerbar. Rust itererar och bygger JSON-patchen. Typisk dirty-rate: 1-5% per frame.

## Rust State Machine (`interface/ansi.rs`)

### States

```
Ground → Escape → CSI → (params) → Execute
                → OSC → (ignorera tills BEL/ST)
```

### Flöde

Itererar över `classes[]`-arrayen från `ansi_parser.ea`:
- Klass 0 (printable): skriv tecken till cell_grid, flytta cursor höger
- Klass 1 (ESC): → Escape state
- Klass 2/0 (`[` i Escape state): → CSI state, nollställ params
- Klass 3 (siffra i CSI): ackumulera parameter
- Klass 4 (`;` i CSI): avsluta param, starta nästa
- Klass 5 (final i CSI): exekvera kommando
- Klass 6 (control): `\n` → cursor ner + scroll, `\r` → kolumn 0, `\t` → tab-stop
- Klass 7 (high byte): UTF-8 ackumulering

### CSI-kommandon (grundläggande)

| Sekvens | Namn | Effekt |
|---------|------|--------|
| `CSI n m` | SGR | Färg/stil |
| `CSI r;c H` | CUP | Cursor-position |
| `CSI n J` | ED | Radera display |
| `CSI n K` | EL | Radera rad |
| `CSI n A/B/C/D` | Cursor | Flytta cursor |
| `CSI n;m r` | DECSTBM | Scroll-region (förberett) |
| `CSI ?25 h/l` | DECTCEM | Visa/dölj cursor |

### SGR-parsning

- `0` → reset, `1` → bold, `3` → italic, `4` → underline, `7` → inverse
- `30-37` → fg 8-färg, `38;5;n` → fg 256, `38;2;r;g;b` → fg truecolor
- `40-47` / `48;5;n` / `48;2;r;g;b` → bg

## Transport

### Nya endpoints i `server.rs`

| Endpoint | Metod | Funktion |
|----------|-------|----------|
| `POST /api/term/open` | Skapar PtySession, returnerar `{"id": n}` |
| `GET /api/term/{id}/stream` | SSE-ström med frame-patches |
| `POST /api/term/{id}/input` | Skriver bytes till PTY |
| `POST /api/term/{id}/resize` | Resize PTY + grid |
| `POST /api/term/{id}/close` | Dödar PTY-session |

### SSE-protokoll

```json
{"type":"frame","cursor":[12,5],"cells":[{"r":5,"c":0,"ch":"$","fg":"#a6e3a1","bg":"#1e1e2e","fl":1}]}
{"type":"bell"}
{"type":"resize","cols":120,"rows":36}
{"type":"exit","code":0}
```

### SSE-loop (egen tråd per session)

```
loop {
    poll(master_fd, POLLIN, 16ms)   // ~60 fps cap
    om data:
        read → ansi_parser.ea → state update → terminal_diff.ea → dirty
        om dirty.any(): bygg JSON-patch → SSE event
    om child död:
        exit-event → break
}
```

16ms poll-timeout ger naturlig frame-coalescing.

## Web-UI (`web/chat.html`)

### Ny tile-typ: `term`

Öppnas med `Alt+S`. `createTermTile(id)` skapar:
- `<canvas>` — renderar cell-grid med JetBrains Mono
- `keydown`/`keypress` → `POST /api/term/{id}/input`
- `EventSource(/api/term/{id}/stream)` → frame-patches
- `ResizeObserver` → räknar cols/rows från tile-dimension → `POST /api/term/{id}/resize`

### Canvas-rendering

```js
for (cell of patch.cells) {
    ctx.fillStyle = cell.bg
    ctx.fillRect(cell.c * cellW, cell.r * cellH, cellW, cellH)
    ctx.fillStyle = cell.fg
    ctx.fillText(cell.ch, cell.c * cellW, cell.r * cellH + baseline)
}
```

Cursor: blinkande block på `patch.cursor`.

## Filstruktur

### Nya filer

| Fil | ~Rader | Ansvar |
|-----|--------|--------|
| `src/interface/pty.rs` | 300 | PtySession: openpty, fork, read/write, resize, Drop, CommandBuffer + safety guard |
| `src/interface/ansi.rs` | 200 | State machine, CSI-parsning, SGR, cell-grid update |
| `kernels/ansi_parser.ea` | 60 | SIMD byte-classifier |
| `kernels/terminal_diff.ea` | 40 | SIMD cell-grid diff |

### Ändrade filer

| Fil | Ändring |
|-----|---------|
| `src/interface/server.rs` | 5 terminal-endpoints + SSE-loop + 3 config-endpoints + param-extraction i `/api/generate` |
| `src/kernels/ffi.rs` | FFI-wrappers för ansi_parser och terminal_diff |
| `src/core/router.rs` | `update_config()`, `get_config()`, runtime parameter mutation |
| `src/core/anthropic.rs` | `set_model()`, `set_max_tokens()`, pub api_key setter |
| `web/chat.html` | createTermTile(), Canvas-renderer, Alt+S, hyprbar config, olorin-knapp, config-modal, per-tile config |
| `src/lib.rs` / `mod.rs` | `mod pty; mod ansi;` |

### Oförändrade filer

- `exec.rs` — PTY:n är en ny codepath, inte en ersättning
- `terminal.rs` — REPL förblir oförändrad
- Befintliga kärnor — inga konflikter

### Testfiler

| Fil | Testar |
|-----|--------|
| `tests/pty.rs` | PTY-livscykel: open, write/read, resize, close |
| `tests/ansi.rs` | State machine: SGR, cursor, ED/EL med riktiga sekvenser |
| `tests/terminal_diff.rs` | Diff: identiska grids → tom, ändrade celler korrekt |
| `tests/pty_guard.rs` | Safety guard: blockerade destruktiva kommandon, tillåtna säkra, ctrl-C passthrough |
| `tests/config_api.rs` | Config GET/POST roundtrip, partial update, apikey store/retrieve |

## Säkerhet — PTY Command Guard

PTY-tilen ger full bash-åtkomst via webbläsaren. Utan skydd kan en angripare (eller en oförsiktig användare) köra `rm -rf /` direkt.

**Lösning:** `PtySession::write_guarded()` buffrar tangentinput tills enter. Innan raden skickas till PTY:n körs:

1. **`fused_safety.ea`** — SIMD-scan för injection-mönster och hemliga nycklar (samma kärna som skyddar chat/repl/WhatsApp)
2. **`ShellGuard::check()`** — klassificerar kommandot som Allow/Write/Destructive baserat på befintlig blocklista (rm, mkfs, shutdown, dd, etc.)

Om någon av gatarna blockerar: raden skickas aldrig till bash. Istället returneras ett error som visas som en röd flash i terminalens Canvas-ram.

**Vad som passerar direkt (utan guard):** Raw control-bytes — Ctrl-C (0x03), Ctrl-Z (0x1A), piltangenter (ESC-sekvenser), tab (0x09), backspace (0x7F). Dessa är terminal-kontroll, inte kommandon.

**Policy:** Styrs av `OLORIN_SHELL_POLICY` (env) eller `~/.olorin/shell_policy` (fil). Default: `safe` (blockerar destructive, tillåter write). `strict` blockerar även write-operationer. `open` stänger av guarden helt.

## Hyprbar Config Panel

### Bakgrund

Inference-parametrar (temp, top_k, top_p, rep_penalty, max_tokens) är idag hårdkodade. Web-UI:t skickar parametrar i `/api/generate` men servern ignorerar dem. Anthropic API key sätts via env utan runtime-ändring. Denna sektion lägger till config i hyprbaren och en modal för full konfiguration.

### Hyprbar-utökning

Befintliga element: `model | backend | tps | recall | sessions | cpu | temp | mem | os | uptime | clock`

Nya element efter `tps`:

```
temp:0.4 | k:40 | p:0.9 | rep:1.05 | max:64
```

Kompakt monospace, samma stil som befintliga metrics. Uppdateras via utökat `/api/system`-svar (piggyback på befintlig 2s poll).

### "olorin"-knapp

Klickbart element längst till vänster i hyprbaren. Text: "◆ olorin" i teal. Klick öppnar config-modalen.

### Config-modal

Fullscreen semi-transparent overlay (`rgba(17,17,27,0.85)`) med centrerad panel (max 500px bred). Catppuccin Mocha-tema. Stängs med Escape eller klick utanför.

#### Inference-sektion

| Parameter | Kontroll | Range | Default |
|-----------|----------|-------|---------|
| Model | Dropdown | bitnet / llama / llama8b / qwen | (loaded model) |
| Temperature | Slider + number | 0.0–2.0 | 0.4 |
| Top-K | Slider + number | 1–100 | 40 |
| Top-P | Slider + number | 0.0–1.0 | 0.9 |
| Repetition penalty | Slider + number | 1.0–2.0 | 1.05 |
| Max tokens | Number input | 1–4096 | 64 |

#### Cloud Fallback-sektion

| Parameter | Kontroll | Default |
|-----------|----------|---------|
| API key | Password input (masked) | (from vault or env) |
| Cloud model | Text input | claude-3-5-haiku-latest |
| Cloud max tokens | Number input (1–16384) | 4096 |

#### System-sektion

| Parameter | Kontroll | Default |
|-----------|----------|---------|
| Recall level | Slider + number (0–10) | Current level |
| System prompt | Textarea (4 rader) | Current prompt |

#### Apply-scope

"Apply to: **Global** / **This tile**" toggle längst ner. Default: Global. "This tile" syns bara om en tile är fokuserad. Per-tile override clearas vid Global-save.

#### Save/Cancel

"Apply" (teal) och "Cancel" (surface1).

### Per-tile Config Override

Varje tile får valfritt `_config`-objekt i JS:

```javascript
tile._config = null;  // null = use global
// or:
tile._config = { temperature: 0.7, max_tokens: 256 };
```

Vid `/api/generate` mergas per-tile config över global:

```javascript
const cfg = Object.assign({}, globalConfig, tile._config || {});
```

Servern behöver inte veta om per-tile — klienten skickar rätt värden per request.

### Nya Config-endpoints

| Endpoint | Metod | Funktion |
|----------|-------|----------|
| `GET /api/config` | Returnerar global config (API key masked: `sk-ant-***`) |
| `POST /api/config` | Partiell uppdatering av global config |
| `POST /api/config/apikey` | Sparar API key krypterat i vault |

#### `GET /api/config` svar

```json
{
    "model": "bitnet",
    "temperature": 0.4,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "max_tokens": 64,
    "cloud_model": "claude-3-5-haiku-latest",
    "cloud_max_tokens": 4096,
    "recall_level": 3,
    "system_prompt": "You are Olorin...",
    "has_api_key": true
}
```

#### `POST /api/config` body (partiell)

```json
{ "temperature": 0.8, "max_tokens": 256 }
```

Model-byte returnerar `{"ok":true,"reload_required":true}` — kräver Engine-reload.

#### `POST /api/config/apikey`

Raw bytes i body. Sparas via `vault.store("config:api_key", key_bytes)`. Vid startup: `vault.search("config:api_key")` → initiera AnthropicClient.

### Befintlig Endpoint-fix: `/api/generate`

Servern extraherar idag bara `prompt`. Fixas: parsa `temperature`, `repetition_penalty`, `max_tokens`, `top_k`, `top_p`, `recall_level` från JSON och skicka genom till routern.

### Config Dataflöde

```
Modal "Apply" → POST /api/config → DispatchContext: update Engine + AnthropicClient
API key "Save" → POST /api/config/apikey → vault.store() → AnthropicClient update
Hyprbar ← GET /api/system (utökat med config-fält, befintlig 2s poll)
Modal load ← GET /api/config (en gång vid öppning)
```

### Config-filer

| Fil | Config-ändring | ~Rader |
|-----|----------------|--------|
| `src/interface/server.rs` | 3 config-endpoints, extrahera params i `/api/generate`, config i `/api/system` | +120 |
| `src/core/router.rs` | `update_config()`, `get_config()`, runtime parameter mutation | +40 |
| `src/core/anthropic.rs` | `set_model()`, `set_max_tokens()`, pub api_key setter | +15 |
| `web/chat.html` | Hyprbar config-element, olorin-knapp, modal HTML/CSS/JS, per-tile config | +200 |
| `tests/config_api.rs` | GET/POST roundtrip, partial update, apikey store/retrieve | +60 |

## Hard Rules

1. **Ingen fil över 500 rader.** `pty.rs` och `ansi.rs` separata. Om `server.rs` växer förbi gränsen bryts terminal-SSE ut till `interface/term_stream.rs`.

2. **Varje feature bevisad med end-to-end test.** PTY testas genom att öppna bash, skicka `echo hello`, verifiera att cell-griden innehåller "hello". ANSI-parsning med riktiga escape-sekvenser. Diff med faktiska grids.

3. **Inga fake functions.** `openpty()` misslyckas → error. Ingen fallback till pipe. Kärna kan inte laddas → panic vid start.

4. **Inga prematura features.** Alternate screen, mouse tracking, sixel — byggs inte förrän det behövs. State machine har rum för nya CSI-grenar men de implementeras inte i förväg.

5. **Delete, don't comment.** Ingen `// TODO: add OSC support`.

6. **Cell-grid äger sina buffertar.** `PtySession::new()` allokerar grids en gång. `resize()` re-allokerar. Inga temporära allokeringar i hot path.

7. **Scan-buffert återanvänds.** `scan_buf` allokeras i `new()`, återanvänds varje read-cykel.

8. **SIMD-alignment.** Cell-grids som `Vec<Cell>` med 16-byte aligned struct. Inga casts från `Vec<u8>`.

9. **API key aldrig i plaintext.** Sparas i vault, maskas i GET-svar, skickas via POST body. Aldrig loggad.

10. **Per-tile override är klient-only.** Servern ser bara parametrar per request. Ingen server-state per tile.

11. **Model-byte är explicit.** Inte hot-swap. Kräver bekräftelse och Engine-reload.
