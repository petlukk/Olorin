#!/bin/bash
# Build softmax benchmark on Pi 5 (Cortex-A76, DOTPROD)
#
# Usage: ./build.sh
# Then:  ./bench ~/.olorin/lib/*/libsoftmax.so

set -e

gcc -O3 -march=armv8.2-a+dotprod -o bench bench.c -ldl -lm -DNDEBUG

echo "built: ./bench"
echo "run:   ./bench ~/.olorin/lib/*/libsoftmax.so"
echo ""
echo "for cache analysis:"
echo "  perf stat -e cycles,instructions,cache-misses,cache-references,L1-dcache-load-misses ./bench ~/.olorin/lib/*/libsoftmax.so"
