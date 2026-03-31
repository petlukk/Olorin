# Hyprbar Config Panel Design

Runtime-konfiguration av inference- och cloud-parametrar via Olorins web-UI.

## Bakgrund

Olorin1 har idag hårdkodade inference-parametrar (temp=0.4, top_k=40, top_p=0.9, rep_penalty=1.05, max_tokens=64). Web-UI:t skickar parametrar i `/api/generate` men servern och routern ignorerar dem. Anthropic API key sätts via env-variabel utan möjlighet att ändra runtime. Denna design lägger till en config-panel i hyprbaren och en modal för full konfiguration.

## Krav

- Inference-parametrar synliga i hyprbaren hela tiden
- Klick på "olorin" i hyprbaren öppnar config-modal
- Global config som default, per-tile override möjligt
- API key sparas krypterat i vault (zero-exposure)
- Inga nya dependencies

## Arkitektur

### Hyprbar-utökning

Befintliga element: `model | backend | tps | recall | sessions | cpu | temp | mem | os | uptime | clock`

Nya element efter `tps`:

```
temp:0.4 | k:40 | p:0.9 | rep:1.05 | max:64
```

Kompakt format, samma monospace-stil som befintliga metrics. Uppdateras via samma 2-sekunders poll som systemmetrics (`GET /api/config` piggybacked på befintlig `/api/system`-poll eller separat).

### "olorin"-knapp

Nytt klickbart element längst till vänster i hyprbaren. Text: "◆ olorin" i teal. Klick öppnar config-modalen.

## Config-modal

Fullscreen semi-transparent overlay (`rgba(17,17,27,0.85)`) med centrerad panel (max 500px bred). Catppuccin Mocha-tema. Stängs med Escape eller klick utanför.

### Sektioner

#### Inference

| Parameter | Kontroll | Range | Default |
|-----------|----------|-------|---------|
| Model | Dropdown | bitnet / llama / llama8b / qwen | (loaded model) |
| Temperature | Slider + number | 0.0–2.0 | 0.4 |
| Top-K | Slider + number | 1–100 | 40 |
| Top-P | Slider + number | 0.0–1.0 | 0.9 |
| Repetition penalty | Slider + number | 1.0–2.0 | 1.05 |
| Max tokens | Number input | 1–4096 | 64 |

#### Cloud Fallback

| Parameter | Kontroll | Default |
|-----------|----------|---------|
| API key | Password input (masked) | (from vault or env) |
| Cloud model | Text input | claude-3-5-haiku-latest |
| Cloud max tokens | Number input (1–16384) | 4096 |

#### System

| Parameter | Kontroll | Default |
|-----------|----------|---------|
| Recall level | Slider + number (0–10) | Current level |
| System prompt | Textarea (4 rader) | Current prompt |

### Apply-scope

Längst ner i modalen: "Apply to: **Global** / **This tile**" toggle. Default: Global. "This tile" syns bara om en tile är fokuserad. Per-tile override clearas om man väljer Global och sparar.

### Save/Cancel

Två knappar: "Apply" (teal) och "Cancel" (surface1). Apply skickar `POST /api/config`. Cancel stänger utan ändring.

## Per-tile Config Override

Varje tile får ett valfritt `_config`-objekt i JS:

```javascript
tile._config = null;  // null = use global
// or:
tile._config = {
    temperature: 0.7,
    max_tokens: 256,
    // only overridden fields present
};
```

När en tile skickar `/api/generate` mergas per-tile config över global:

```javascript
const cfg = Object.assign({}, globalConfig, tile._config || {});
body.temperature = cfg.temperature;
// ...
```

Servern behöver inte veta om per-tile — klienten skickar redan rätt värden per request.

## Nya Endpoints

### `GET /api/config`

Returnerar nuvarande global config. API key maskas (`sk-ant-***`).

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

### `POST /api/config`

Uppdaterar global config. Accepterar partiella uppdateringar (bara de fält som skickas). Returnerar uppdaterad config.

```json
{
    "temperature": 0.8,
    "max_tokens": 256
}
```

Fält som stöds: `temperature`, `top_k`, `top_p`, `repetition_penalty`, `max_tokens`, `cloud_model`, `cloud_max_tokens`, `recall_level`, `system_prompt`.

Model-byte (`model`-fältet) kräver omladdning av Engine — servern returnerar `{"ok":true,"reload_required":true}` och klienten visar en notis.

### `POST /api/config/apikey`

Sparar API key krypterat i vault. Request body: raw bytes (nyckeln). Servern kör `vault.store("config:api_key", key_bytes)`. Returnerar `{"ok":true}`.

Vid startup: `vault.search("config:api_key")` → om hittad, initiera AnthropicClient med den.

## Befintlig Endpoint-fix: `/api/generate`

Servern ska extrahera parametrar från request body och skicka dem till `dispatch_streaming()`. Idag extraheras bara `prompt`. Fixas genom att parsa: `temperature`, `repetition_penalty`, `max_tokens`, `recall_level` från JSON-bodyn och skicka genom till routern.

## Dataflöde

### Config-uppdatering

```
Modal "Apply" → POST /api/config (JSON)
    → server.rs: parse fields
    → DispatchContext: update Engine fields + AnthropicClient fields
    → Response: updated config JSON
```

### API key

```
Modal "Save key" → POST /api/config/apikey (raw bytes)
    → server.rs: read body
    → safety::scan(key) — skip (det är en nyckel, inte user input)
    → vault.store("config:api_key", key_bytes)
    → Om AnthropicClient saknas: skapa ny med nyckeln
    → Om finns: uppdatera api_key-fältet
```

### Hyprbar-uppdatering

```
Befintlig 2s poll → GET /api/config
    → JS uppdaterar hyprbar-element: hb-temp-cfg, hb-topk, hb-topp, hb-rep, hb-maxtok
```

Alternativt: piggybacka på `/api/system`-pollen och lägg till config-fält i svaret. Enklare, ett anrop istället för två.

**Beslut:** Utöka `/api/system`-svaret med config-fälten. Inget separat poll behövs. `/api/config` GET finns ändå för modalen att ladda initial state.

## Ändringar per fil

### Nya filer

Inga nya Rust-filer behövs — config-logiken ryms i befintliga moduler.

### Ändrade filer

| Fil | Ändring | ~Rader |
|-----|---------|--------|
| `src/interface/server.rs` | 3 nya endpoints, extrahera params i `/api/generate`, config-fält i `/api/system` | +120 |
| `src/core/router.rs` | `update_config()` metod, `get_config()` metod, runtime parameter mutation | +40 |
| `src/core/anthropic.rs` | `set_model()`, `set_max_tokens()`, pub api_key setter | +15 |
| `web/chat.html` | Hyprbar config-element, olorin-knapp, modal HTML/CSS/JS, per-tile config merge | +200 |

### Testfiler

| Fil | Testar |
|-----|--------|
| `tests/config_api.rs` | GET/POST /api/config roundtrip, partial update, apikey store/retrieve |

## Hard Rules

1. **Ingen fil över 500 rader.** Om `server.rs` växer förbi gränsen bryts config-handlers ut till `interface/config.rs`.

2. **API key aldrig i plaintext.** Sparas i vault, maskas i GET-svar, skickas via POST body (inte query param). Aldrig loggad.

3. **Config-modal är ren JS.** Ingen extern dependency. Samma mönster som befintlig chat/repl-tile.

4. **Per-tile override är klient-only.** Servern ser bara parametrar per request. Ingen server-state per tile.

5. **Model-byte är explicit.** Inte hot-swap. Kräver bekräftelse och potentiellt Engine-reload.

6. **Sliders har numeric input bredvid.** Användaren kan alltid skriva exakt värde.
