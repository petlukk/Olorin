//! Runtime config API for the Olorin Pipe.
//!
//! get/update config, API key storage/loading from vault.
//! Split from router.rs for the 500-line rule.

use crate::core::anthropic::AnthropicClient;
use crate::core::router::DispatchContext;

impl DispatchContext {
    pub fn get_config(&self) -> String {
        let (model, temp, top_k, top_p, rep_pen, max_tok) = match &self.engine {
            Some(e) => (e.quant_type_str(), e.temperature, e.top_k, e.top_p, e.repetition_penalty, e.max_tokens),
            None => ("none", 0.0, 0, 0.0, 1.0, 0),
        };
        let (cloud_model, cloud_max, has_key) = match &self.anthropic {
            Some(a) => (a.model(), a.max_tokens(), a.has_key()),
            None => ("claude-3-5-haiku-latest", 4096, false),
        };
        let system_prompt = crate::interface::server::escape_json(&self.system_prompt);
        let available = crate::inference::generate::available_models();
        let models_json: String = available.iter()
            .map(|m| format!("\"{m}\""))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"model\":\"{model}\",\"temperature\":{temp},\
             \"top_k\":{top_k},\"top_p\":{top_p},\
             \"repetition_penalty\":{rep_pen},\"max_tokens\":{max_tok},\
             \"cloud_model\":\"{cloud_model}\",\"cloud_max_tokens\":{cloud_max},\
             \"recall_level\":{},\"system_prompt\":\"{system_prompt}\",\
             \"has_api_key\":{has_key},\"available_models\":[{models_json}]}}",
            self.recall_level
        )
    }

    pub fn update_config(&mut self, json: &str) {
        use crate::interface::server::extract_json_string;
        use crate::storage::json::{extract_json_float, extract_json_int};

        // Model change requires full engine reload
        if let Some(model_name) = extract_json_string(json, "model") {
            let current = self.engine.as_ref().map(|e| e.quant_type_str()).unwrap_or("none");
            if model_name != current {
                eprintln!("[olorin] Reloading model: {model_name}");
                self.engine = Self::load_engine(Some(&model_name));
            }
        }

        if let Some(engine) = &mut self.engine {
            if let Some(v) = extract_json_float(json, "temperature") { engine.temperature = v; }
            if let Some(v) = extract_json_int(json, "top_k") { engine.top_k = v as usize; }
            if let Some(v) = extract_json_float(json, "top_p") { engine.top_p = v; }
            if let Some(v) = extract_json_float(json, "repetition_penalty") { engine.repetition_penalty = v; }
            if let Some(v) = extract_json_int(json, "max_tokens") { engine.max_tokens = v as usize; }
        }
        if let Some(anthropic) = &mut self.anthropic {
            if let Some(v) = extract_json_string(json, "cloud_model") { anthropic.set_model(v); }
            if let Some(v) = extract_json_int(json, "cloud_max_tokens") { anthropic.set_max_tokens(v as i64); }
        }
        if let Some(v) = extract_json_int(json, "recall_level") { self.recall_level = v as usize; }
        if let Some(v) = extract_json_string(json, "system_prompt") { self.system_prompt = v; }
    }

    pub fn store_api_key(&mut self, key: &str) {
        if let Some(vault) = &mut self.vault {
            let _ = vault.append(b"config:api_key", key.as_bytes());
        }
        match &mut self.anthropic {
            Some(a) => a.set_api_key(key.to_string()),
            None => self.anthropic = Some(AnthropicClient::new(key.to_string())),
        }
    }

    pub fn load_api_key_from_vault(&mut self) {
        if self.anthropic.is_some() { return; }
        if let Some(vault) = &mut self.vault {
            if let Ok(results) = vault.search("config:api_key", 1) {
                if let Some(hit) = results.first() {
                    for line in &hit.lines {
                        if let Some(key) = line.strip_prefix("config:api_key: ") {
                            let key = key.trim().to_string();
                            if !key.is_empty() {
                                self.anthropic = Some(AnthropicClient::new(key));
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}
