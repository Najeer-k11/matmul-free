#ifndef MATMUL_FREE_GPU_OPS_H
#define MATMUL_FREE_GPU_OPS_H

#include <vector>
#include <cstdint>

namespace matmul_free {

#if defined(USE_CUDA)

bool is_cuda_available();
void print_cuda_device_info();

std::vector<float> cuda_matmul_vector(const std::vector<std::vector<float>>& weight,
                                      const std::vector<float>& input);

std::vector<float> cuda_bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                         const std::vector<float>& input,
                                         float scale);

std::vector<float> cuda_bitlinear_packed(const std::vector<std::vector<uint8_t>>& packed_weight,
                                         const std::vector<float>& input,
                                         float scale);

std::vector<std::vector<float>> cuda_bitlinear_sequence(const std::vector<std::vector<uint8_t>>& packed_weight,
                                                        const std::vector<std::vector<float>>& A,
                                                        int unpacked_cols,
                                                        float scale);

#else

inline bool is_cuda_available() { return false; }
inline void print_cuda_device_info() {}
inline std::vector<float> cuda_matmul_vector(const std::vector<std::vector<float>>&, const std::vector<float>&) { return {}; }
inline std::vector<float> cuda_bitlinear_vector(const std::vector<std::vector<int8_t>>&, const std::vector<float>&, float) { return {}; }
inline std::vector<float> cuda_bitlinear_packed(const std::vector<std::vector<uint8_t>>&, const std::vector<float>&, float) { return {}; }
inline std::vector<std::vector<float>> cuda_bitlinear_sequence(const std::vector<std::vector<uint8_t>>&, const std::vector<std::vector<float>>&, int, float) { return {}; }

#endif

} // namespace matmul_free

#endif // MATMUL_FREE_GPU_OPS_H
