//! Gemma 4 model structure and weight loading from GGUF.

pub struct Gemma4Model;

impl Gemma4Model {
    pub fn from_gguf(_gguf: &crate::inference::gguf::GgufFile) -> Result<Self, String> {
        Err("Gemma 4 model loading not yet implemented".into())
    }
}
