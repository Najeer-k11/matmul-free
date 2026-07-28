#ifndef MATMUL_FREE_BENCHMARK_H
#define MATMUL_FREE_BENCHMARK_H

namespace matmul_free {

/**
 * Check if OpenMP multi-threading parallel hardware acceleration is enabled.
 */
bool is_openmp_accelerated();

/**
 * Micro-benchmark comparing traditional floating-point GEMM vs. Ternary BitLinear vs. Explicit AVX2 SIMD.
 */
void benchmark_matmul_vs_bitlinear(int num_rows = 512, int num_cols = 512, int iterations = 100);

} // namespace matmul_free

#endif // MATMUL_FREE_BENCHMARK_H
