#include "math_ops.h"
#include "../cuda/gpu_ops.h"
#include <limits>
#include <cfloat>
#include <cmath>

namespace matmul_free {

std::vector<float> matmul_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec) {
    if (matrix.empty() || vec.empty()) return {};
    
    size_t rows = matrix.size();
#if defined(USE_CUDA)
    if (is_cuda_available() && rows >= 64) {
        return cuda_matmul_vector(matrix, vec);
    }
#endif

    size_t cols = matrix[0].size();
    std::vector<float> result(rows, 0.0f);
    
    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        size_t limit = std::min(cols, vec.size());
        for (size_t j = 0; j < limit; ++j) {
            sum += matrix[i][j] * vec[j];
        }
        result[i] = sum;
    }
    
    return result;
}

std::vector<std::vector<float>> matmul_blocks(const std::vector<std::vector<float>>& A,
                                               const std::vector<std::vector<float>>& B,
                                               int block_size) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::invalid_argument("Invalid matrix dimensions for block multiplication");
    }
    
    size_t M = A.size();
    size_t K = A[0].size();
    size_t N = B[0].size();
    
    std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));
    
    #pragma omp parallel for collapse(2) if(M > 32)
    for (size_t i0 = 0; i0 < M; i0 += block_size) {
        for (size_t j0 = 0; j0 < N; j0 += block_size) {
            for (size_t k0 = 0; k0 < K; k0 += block_size) {
                size_t i_max = std::min(i0 + block_size, M);
                size_t j_max = std::min(j0 + block_size, N);
                size_t k_max = std::min(k0 + block_size, K);
                
                for (size_t i = i0; i < i_max; ++i) {
                    for (size_t k = k0; k < k_max; ++k) {
                        float a_ik = A[i][k];
                        for (size_t j = j0; j < j_max; ++j) {
                            C[i][j] += a_ik * B[k][j];
                        }
                    }
                }
            }
        }
    }
    
    return C;
}

float gelu(float x) {
    const float SQRT_2_OVER_PI = 0.7978845608028654f;
    const float COEFF = 0.044715f;
    float x_cubed = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

float swiglu(float x, float gate) {
    float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate));
    return x * sigmoid_gate;
}

std::vector<float> rmsnorm(const std::vector<float>& x, float eps) {
    if (x.empty()) return {};
    float sum_sq = 0.0f;
    for (float val : x) {
        if (!std::isnan(val) && !std::isinf(val)) {
            sum_sq += val * val;
        }
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(x.size()) + eps);
    if (std::isnan(rms) || rms < 1e-8f) rms = 1e-4f;
    
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float val = (std::isnan(x[i]) || std::isinf(x[i])) ? 0.0f : x[i];
        out[i] = val / rms;
    }
    return out;
}

std::vector<float> rmsnorm_backward(const std::vector<float>& x, const std::vector<float>& grad_y, float eps) {
    if (x.empty()) return {};
    size_t N = x.size();
    
    float sum_sq = 0.0f;
    for (float val : x) {
        if (!std::isnan(val) && !std::isinf(val)) {
            sum_sq += val * val;
        }
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(N) + eps);
    if (std::isnan(rms) || rms < 1e-8f) rms = 1e-4f;
    float inv_rms = 1.0f / rms;

    float sum_gy_y = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float y_i = ((std::isnan(x[i]) || std::isinf(x[i])) ? 0.0f : x[i]) * inv_rms;
        float gy_i = (i < grad_y.size()) ? grad_y[i] : 0.0f;
        if (!std::isnan(gy_i) && !std::isinf(gy_i)) {
            sum_gy_y += gy_i * y_i;
        }
    }
    float mean_gy_y = sum_gy_y / static_cast<float>(N);

    std::vector<float> grad_x(N);
    for (size_t k = 0; k < N; ++k) {
        float y_k = ((std::isnan(x[k]) || std::isinf(x[k])) ? 0.0f : x[k]) * inv_rms;
        float gy_k = (k < grad_y.size()) ? grad_y[k] : 0.0f;
        if (std::isnan(gy_k) || std::isinf(gy_k)) gy_k = 0.0f;
        grad_x[k] = inv_rms * (gy_k - y_k * mean_gy_y);
    }

    return grad_x;
}

std::vector<float> apply_rope(const std::vector<float>& vec, int pos, int dim) {
    if (vec.size() < static_cast<size_t>(dim)) return vec;
    std::vector<float> out = vec;
    for (int i = 0; i < dim - 1; i += 2) {
        float theta = std::pow(10000.0f, -static_cast<float>(i) / static_cast<float>(dim));
        float m_theta = static_cast<float>(pos) * theta;
        float cos_m = std::cos(m_theta);
        float sin_m = std::sin(m_theta);

        float v0 = vec[i];
        float v1 = vec[i + 1];
        out[i]     = v0 * cos_m - v1 * sin_m;
        out[i + 1] = v0 * sin_m + v1 * cos_m;
    }
    return out;
}

} // namespace matmul_free
