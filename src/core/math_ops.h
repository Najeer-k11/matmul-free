#ifndef MATMUL_FREE_MATH_OPS_H
#define MATMUL_FREE_MATH_OPS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cfloat>

namespace matmul_free {

/**
 * Compute softmax using log-sum-exp trick for numerical stability.
 */
template<typename T>
std::vector<T> softmax(const std::vector<T>& input, bool use_log_sum_exp = true) {
    if (input.empty()) return {};
    
    T max_val = input[0];
    for (size_t i = 1; i < input.size(); ++i) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    std::vector<T> result(input.size());
    
    if (use_log_sum_exp) {
        T sum_exp = static_cast<T>(0);
        for (size_t i = 0; i < input.size(); ++i) {
            sum_exp += std::exp(input[i] - max_val);
        }
        
        T log_sum_exp = max_val + std::log(sum_exp);
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = std::exp(input[i] - log_sum_exp);
        }
    } else {
        T sum = static_cast<T>(0);
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = std::exp(input[i] - max_val);
            sum += result[i];
        }
        
        if (sum > static_cast<T>(0)) {
            for (size_t i = 0; i < input.size(); ++i) {
                result[i] /= sum;
            }
        }
    }
    
    return result;
}

/**
 * Matrix-vector product using OpenMP multi-threading.
 */
std::vector<float> matmul_vector(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec);

/**
 * Blocked matrix multiplication.
 */
std::vector<std::vector<float>> matmul_blocks(const std::vector<std::vector<float>>& A,
                                               const std::vector<std::vector<float>>& B,
                                               int block_size = 32);

/**
 * Gaussian Error Linear Unit (GELU) activation function.
 */
float gelu(float x);

/**
 * SwiGLU activation function: x * sigmoid(gate).
 */
float swiglu(float x, float gate);

/**
 * Root Mean Square Normalization (RMSNorm).
 */
std::vector<float> rmsnorm(const std::vector<float>& x, float eps = 1e-5f);

/**
 * Root Mean Square Normalization Backward Pass (gradient flow).
 */
std::vector<float> rmsnorm_backward(const std::vector<float>& x, const std::vector<float>& grad_y, float eps = 1e-5f);

/**
 * Rotary Positional Embeddings (RoPE).
 */
std::vector<float> apply_rope(const std::vector<float>& vec, int pos, int dim);

} // namespace matmul_free

#endif // MATMUL_FREE_MATH_OPS_H
