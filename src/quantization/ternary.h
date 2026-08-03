#ifndef MATMUL_FREE_TERNARY_H
#define MATMUL_FREE_TERNARY_H

#include <vector>
#include <cstdint>
#include <cstddef>

namespace matmul_free {

/**
 * Quantize floating point weight matrix into 1.58-bit ternary matrix {-1, 0, +1}.
 */
std::vector<std::vector<int8_t>> quantize_weights_ternary(const std::vector<std::vector<float>>& weight,
                                                            float& scale);

/**
 * Multiplication-free linear layer using ternary weights.
 */
std::vector<float> bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                     const std::vector<float>& input,
                                     float scale);

/**
 * Pack ternary matrix into 2-bit storage.
 */
std::vector<std::vector<uint8_t>> pack_ternary_matrix(const std::vector<std::vector<int8_t>>& ternary_weight);

/**
 * Multiplication-free linear layer using 2-bit packed ternary storage.
 */
std::vector<float> bitlinear_packed_simd(const std::vector<std::vector<uint8_t>>& packed_weight,
                                          const std::vector<float>& input,
                                          float scale);

/**
 * Packed ternary execution using explicit hardware AVX2 SIMD intrinsics.
 */
std::vector<float> bitlinear_packed_avx2(const std::vector<std::vector<uint8_t>>& packed_weight,
                                          const std::vector<float>& input,
                                          float scale);

/**
 * Multiplication-free BitLinear dot product using bitwise masks and hardware popcount instructions.
 */
std::vector<float> bitlinear_popcount_simd(const std::vector<std::vector<uint8_t>>& packed_weight,
                                           const std::vector<float>& input,
                                           float scale);

} // namespace matmul_free

#endif // MATMUL_FREE_TERNARY_H
