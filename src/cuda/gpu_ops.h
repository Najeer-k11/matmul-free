#ifndef MATMUL_FREE_GPU_OPS_H
#define MATMUL_FREE_GPU_OPS_H

#include <vector>
#include <cstdint>

namespace matmul_free {

/**
 * Check if CUDA GPU acceleration is enabled and available on the system.
 */
bool is_cuda_available();

/**
 * Print CUDA device properties for the active GPU (e.g. NVIDIA GeForce RTX 4060).
 */
void print_cuda_device_info();

/**
 * GPU CUDA matrix-vector multiplication for FP32 weights.
 */
std::vector<float> cuda_matmul_vector(const std::vector<std::vector<float>>& weight,
                                      const std::vector<float>& input);

/**
 * GPU CUDA 1.58-bit ternary BitLinear matrix-vector multiplication.
 */
std::vector<float> cuda_bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                         const std::vector<float>& input,
                                         float scale);

/**
 * GPU CUDA 2-bit packed BitLinear ternary matrix-vector multiplication.
 */
std::vector<float> cuda_bitlinear_packed(const std::vector<std::vector<uint8_t>>& packed_weight,
                                         const std::vector<float>& input,
                                         float scale);

/**
 * Fast GPU CUDA BitLinear ternary sequence GEMM (MatMul-Free: zero FP32 multiplies).
 */
std::vector<std::vector<float>> cuda_bitlinear_sequence(const std::vector<std::vector<uint8_t>>& packed_weight,
                                                        const std::vector<std::vector<float>>& A,
                                                        int unpacked_cols,
                                                        float scale);

} // namespace matmul_free

#endif // MATMUL_FREE_GPU_OPS_H
