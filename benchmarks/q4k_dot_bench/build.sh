#!/bin/bash
# Build Q4K dot product benchmark on Pi 5 (Cortex-A76, DOTPROD)
#
# Usage: ./build.sh
# Then:  ./bench ~/.olorin/lib/*/libq4k_dot.so

set -e

gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c llama_q4k.c \
    -ldl -lm -DNDEBUG

echo "built: ./bench"
echo "run:   ./bench ~/.olorin/lib/*/libq4k_dot.so"
echo ""
echo "for cache analysis:"
echo "  perf stat -e cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses ./bench ~/.olorin/lib/*/libq4k_dot.so"
