#include "benchmark.h"
#include "../core/math_ops.h"
#include "../quantization/ternary.h"
#include "../cuda/gpu_ops.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <vector>
#include <cstdlib>

namespace matmul_free {

bool is_openmp_accelerated() {
#if defined(_OPENMP)
    return true;
#else
    return false;
#endif
}

void benchmark_matmul_vs_bitlinear(int num_rows, int num_cols, int iterations) {
    if (is_cuda_available()) {
        print_cuda_device_info();
    }
    std::cout << "Running Benchmark (Matrix Size: " << num_rows << "x" << num_cols 
              << ", Iterations: " << iterations << ", 5-Trial Median)...\n";
    
    std::vector<std::vector<float>> weight(num_rows, std::vector<float>(num_cols));
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            weight[i][j] = static_cast<float>(rand() % 100 - 50) / 100.0f;
        }
    }
    std::vector<float> input(num_cols, 0.5f);

    float scale = 1.0f;
    auto ternary_w = quantize_weights_ternary(weight, scale);
    auto packed_w  = pack_ternary_matrix(ternary_w);

    const int num_trials = 5;

    auto run_benchmark_trials = [&](auto kernel_fn) -> double {
        std::vector<double> trial_times;
        for (int trial = 0; trial < num_trials; ++trial) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it) {
                auto out = kernel_fn();
                (void)out;
            }
            auto end = std::chrono::high_resolution_clock::now();
            trial_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        std::sort(trial_times.begin(), trial_times.end());
        return trial_times[num_trials / 2];
    };

    double fp_ms   = run_benchmark_trials([&]() { return matmul_vector(weight, input); });
    double bit_ms  = run_benchmark_trials([&]() { return bitlinear_vector(ternary_w, input, scale); });
    double simd_ms = run_benchmark_trials([&]() { return bitlinear_packed_simd(packed_w, input, scale); });
    double avx_ms  = run_benchmark_trials([&]() { return bitlinear_packed_avx2(packed_w, input, scale); });
    double pop_ms  = run_benchmark_trials([&]() { return bitlinear_popcount_simd(packed_w, input, scale); });

    std::cout << "  FP32 MatMul Latency (Median):  " << std::fixed << std::setprecision(3) << fp_ms << " ms\n";
    std::cout << "  BitLinear Latency (Median):    " << std::fixed << std::setprecision(3) << bit_ms << " ms\n";
    std::cout << "  Packed 2-Bit SIMD (Median):    " << std::fixed << std::setprecision(3) << simd_ms << " ms\n";
    std::cout << "  Explicit AVX2 SIMD (Median):   " << std::fixed << std::setprecision(3) << avx_ms << " ms\n";
    std::cout << "  Bit-Parallel Popcount (Median):" << std::fixed << std::setprecision(3) << pop_ms << " ms\n";

    if (is_cuda_available()) {
        double cuda_bit_ms = run_benchmark_trials([&]() { return cuda_bitlinear_vector(ternary_w, input, scale); });
        double cuda_fp_ms  = run_benchmark_trials([&]() { return cuda_matmul_vector(weight, input); });
        std::cout << "  RTX 4060 CUDA BitLinear:      " << std::fixed << std::setprecision(3) << cuda_bit_ms << " ms\n";
        std::cout << "  RTX 4060 CUDA FP32 GEMV:       " << std::fixed << std::setprecision(3) << cuda_fp_ms << " ms\n";
    }

    size_t fp32_bytes = num_rows * num_cols * sizeof(float);
    size_t packed_bytes = (num_rows * num_cols) / 4;
    double mem_reduction = static_cast<double>(fp32_bytes) / static_cast<double>(packed_bytes);

    std::cout << "  FP32 Memory Footprint:        " << fp32_bytes / 1024 << " KB\n";
    std::cout << "  Packed Memory Footprint:      " << packed_bytes / 1024 << " KB\n";
    std::cout << "  Memory Footprint Ratio:       " << std::setprecision(2) << mem_reduction << "x smaller!\n";
}

} // namespace matmul_free
