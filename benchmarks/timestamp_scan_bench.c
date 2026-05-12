// Throughput benchmark for libtimestamp_scan.so (the kernel behind
// the eatime rune). Generates a deterministic synthetic log buffer
// in memory, times one kernel call, reports MB/s.
//
// Build:
//   x86_64:  gcc -O2 -o bench_x86 timestamp_scan_bench.c -ldl
//   aarch64: aarch64-linux-gnu-gcc -O2 -o bench_arm timestamp_scan_bench.c -ldl
//
// Run:
//   ./bench_x86 <path/to/libtimestamp_scan.so> [size_mb]
//   ./bench_arm <path/to/libtimestamp_scan.so> [size_mb]
//
// Default size: 100 MB. Default trials: 5 (reports min and mean).

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*timestamp_scan_fn)(
    const uint8_t *text, int32_t len,
    int32_t *out_positions, int32_t max_positions, int32_t *out_n_positions,
    uint8_t *scratch);

// Realistic synthetic log line. Total 95 bytes including the newline,
// so 100 MB ≈ 1.1M lines. Hour and minute vary across lines so the
// fixture exercises a range of timestamp patterns rather than a
// single repeated value.
static int fill_line(char *buf, int line_idx) {
    int h = line_idx % 24;
    int m = (line_idx * 7) % 60;
    int s = (line_idx * 13) % 60;
    return snprintf(buf, 96,
        "2026-05-%02dT%02d:%02d:%02d INFO event=heartbeat shard=%03d worker=%05d ok=true\n",
        ((line_idx / 1440) % 28) + 1, h, m, s,
        line_idx % 100, line_idx % 10000);
}

static double now_secs(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path/to/libtimestamp_scan.so> [size_mb]\n", argv[0]);
        return 2;
    }
    void *h = dlopen(argv[1], RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    timestamp_scan_fn ts = (timestamp_scan_fn) dlsym(h, "timestamp_scan");
    if (!ts) { fprintf(stderr, "dlsym failed: %s\n", dlerror()); return 1; }

    int size_mb = (argc >= 3) ? atoi(argv[2]) : 100;
    if (size_mb < 1 || size_mb > 2000) { size_mb = 100; }
    int32_t bytes_target = size_mb * 1024 * 1024;

    // Allocate fixture and fill it line by line until we hit the target.
    uint8_t *fixture = malloc((size_t) bytes_target + 256);
    if (!fixture) { fprintf(stderr, "malloc %d MB failed\n", size_mb); return 1; }
    int32_t fill = 0;
    int     line_idx = 0;
    while (fill < bytes_target) {
        char line[128];
        int n = fill_line(line, line_idx);
        if (n <= 0 || n > 128) { break; }
        if (fill + n > bytes_target) { break; }
        memcpy(fixture + fill, line, n);
        fill += n;
        line_idx++;
    }
    int32_t len = fill;

    // Output buffer: cap positions at 2M (well above realistic counts
    // for the fixture — each timestamp is one position).
    const int32_t max_positions = 2 * 1024 * 1024;
    int32_t *positions = malloc((size_t) max_positions * sizeof(int32_t));
    int32_t  n_positions = 0;
    uint8_t  scratch[16] = {0};

    // Warmup: prime caches, page-fault the fixture once.
    ts(fixture, len, positions, max_positions, &n_positions, scratch);

    const int trials = 5;
    double times[trials];
    for (int t = 0; t < trials; ++t) {
        n_positions = 0;
        double t0 = now_secs();
        ts(fixture, len, positions, max_positions, &n_positions, scratch);
        double t1 = now_secs();
        times[t] = t1 - t0;
    }

    double best = times[0];
    double sum  = 0.0;
    for (int t = 0; t < trials; ++t) {
        if (times[t] < best) best = times[t];
        sum += times[t];
    }
    double mean = sum / trials;
    double mb = (double) len / (1024.0 * 1024.0);
    double gb = mb / 1024.0;

    printf("fixture:    %.1f MB (%d bytes, %d lines)\n", mb, len, line_idx);
    printf("matches:    %d positions found\n", n_positions);
    printf("best:       %.3f ms = %.2f GB/s\n", best * 1000.0, gb / best);
    printf("mean:       %.3f ms = %.2f GB/s\n", mean * 1000.0, gb / mean);
    return 0;
}
