fn main() {
    let mut build = cc::Build::new();
    build
        .include("csrc")
        .file("csrc/cache.c")
        .file("csrc/io.c")
        .file("csrc/attention.c")
        .file("csrc/ggml_type.c")
        .file("csrc/llama_bridge.c")
        .opt_level(3)
        .compile("eakv");
}
