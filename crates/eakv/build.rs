use std::path::Path;

fn main() {
    // Pre-built Ea SIMD kernel objects from the eakv project.
    // These are compiled from .ea kernel sources and must be linked in.
    let kernel_obj_dir = Path::new("/root/dev/eakv/build/obj");

    let kernel_objs = [
        "quantize_simd.o",
        "dequantize_sse.o",
        "dequantize_avx2.o",
        "dequantize_avx512.o",
        "validate.o",
        "fused_k_score.o",
        "fused_k_score_64.o",
        "fused_k_score_gqa.o",
        "fused_k_score_gqa_64.o",
        "fused_v_sum.o",
        "fused_v_sum_64.o",
        "fused_attention.o",
        "turbo_rotate.o",
    ];

    // Build the core C library (llama_bridge.c is excluded — it requires ggml.h
    // from llama.cpp and is only needed for optional llama.cpp integration).
    let mut build = cc::Build::new();
    build
        .include("csrc")
        .file("csrc/cache.c")
        .file("csrc/io.c")
        .file("csrc/attention.c")
        .file("csrc/ggml_type.c")
        .opt_level(3);

    // Add pre-built kernel objects directly to the compilation unit.
    for obj in &kernel_objs {
        let path = kernel_obj_dir.join(obj);
        if path.exists() {
            build.object(&path);
        }
    }

    build.compile("eakv");

    // Tell cargo to re-run if any kernel object changes.
    println!("cargo:rerun-if-changed=csrc/");
    println!("cargo:rerun-if-changed=/root/dev/eakv/build/obj/");
}
