use olorin::inference::gguf::GgufFile;
use olorin::inference::tokenizer::Tokenizer;
use std::path::Path;

#[test]
#[ignore]
fn decode_divergent_tokens() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    let gguf = GgufFile::open(&path).unwrap();
    let tok = Tokenizer::from_gguf(&gguf).unwrap();
    for id in [11733u32, 114525, 236764, 708, 3159, 495] {
        let s = tok.decode(&[id]);
        println!("token {:>6} -> {:?}", id, s);
    }
}
