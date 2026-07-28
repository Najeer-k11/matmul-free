#include "ternary.h"
#include <cmath>
#include <algorithm>
#include <immintrin.h>

namespace matmul_free {

std::vector<std::vector<int8_t>> quantize_weights_ternary(const std::vector<std::vector<float>>& weight,
                                                            float& scale) {
    if (weight.empty() || weight[0].empty()) {
        scale = 1.0f;
        return {};
    }
    
    double abs_sum = 0.0;
    size_t count = 0;
    for (const auto& row : weight) {
        for (float val : row) {
            if (!std::isnan(val) && !std::isinf(val)) {
                abs_sum += std::abs(val);
                count++;
            }
        }
    }
    scale = count > 0 ? static_cast<float>(abs_sum / count) : 1.0f;
    if (std::isnan(scale) || std::isinf(scale) || scale < 1e-6f) scale = 1e-6f;

    std::vector<std::vector<int8_t>> ternary(weight.size(), std::vector<int8_t>(weight[0].size()));
    for (size_t i = 0; i < weight.size(); ++i) {
        for (size_t j = 0; j < weight[i].size(); ++j) {
            float w_val = (std::isnan(weight[i][j]) || std::isinf(weight[i][j])) ? 0.0f : weight[i][j];
            float val = w_val / scale;
            if (val >= 0.5f) {
                ternary[i][j] = 1;
            } else if (val <= -0.5f) {
                ternary[i][j] = -1;
            } else {
                ternary[i][j] = 0;
            }
        }
    }
    return ternary;
}

std::vector<float> bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                     const std::vector<float>& input,
                                     float scale) {
    size_t rows = weight_ternary.size();
    std::vector<float> output(rows, 0.0f);

    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const auto& w_row = weight_ternary[i];
        size_t cols = std::min(w_row.size(), input.size());
        
        const int8_t* w_ptr = w_row.data();
        const float* x_ptr = input.data();
        
        for (size_t j = 0; j < cols; ++j) {
            acc += static_cast<float>(w_ptr[j]) * x_ptr[j];
        }
        output[i] = acc * scale;
    }
    return output;
}

std::vector<std::vector<uint8_t>> pack_ternary_matrix(const std::vector<std::vector<int8_t>>& ternary) {
    if (ternary.empty()) return {};
    size_t rows = ternary.size();
    size_t cols = ternary[0].size();
    size_t packed_cols = (cols + 3) / 4;

    std::vector<std::vector<uint8_t>> packed(rows, std::vector<uint8_t>(packed_cols, 0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            int8_t val = ternary[i][j];
            uint8_t code = (val == 1) ? 0b01 : ((val == -1) ? 0b10 : 0b00);
            size_t byte_idx = j / 4;
            size_t bit_shift = (j % 4) * 2;
            packed[i][byte_idx] |= (code << bit_shift);
        }
    }
    return packed;
}

std::vector<float> bitlinear_packed_simd(const std::vector<std::vector<uint8_t>>& packed_weight,
                                           const std::vector<float>& input,
                                           float scale) {
    size_t rows = packed_weight.size();
    std::vector<float> output(rows, 0.0f);
    size_t cols = input.size();

    static const float kLut[4] = {0.0f, 1.0f, -1.0f, 0.0f};

    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const uint8_t* p_row = packed_weight[i].data();
        const float* x_ptr = input.data();

        size_t j = 0;
        size_t packed_len = packed_weight[i].size();
        for (size_t byte_i = 0; byte_i < packed_len && j < cols; ++byte_i) {
            uint8_t b = p_row[byte_i];
            
            uint8_t c0 = b & 0b11;
            acc += kLut[c0] * x_ptr[j++];
            if (j >= cols) break;

            uint8_t c1 = (b >> 2) & 0b11;
            acc += kLut[c1] * x_ptr[j++];
            if (j >= cols) break;

            uint8_t c2 = (b >> 4) & 0b11;
            acc += kLut[c2] * x_ptr[j++];
            if (j >= cols) break;

            uint8_t c3 = (b >> 6) & 0b11;
            acc += kLut[c3] * x_ptr[j++];
        }
        output[i] = acc * scale;
    }
    return output;
}

std::vector<float> bitlinear_packed_avx2(const std::vector<std::vector<uint8_t>>& packed_weight,
                                           const std::vector<float>& input,
                                           float scale) {
    size_t rows = packed_weight.size();
    std::vector<float> output(rows, 0.0f);
    size_t cols = input.size();

    static const float kLut[4] = {0.0f, 1.0f, -1.0f, 0.0f};

    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const uint8_t* p_row = packed_weight[i].data();
        const float* x_ptr = input.data();

        size_t j = 0;
        size_t packed_len = packed_weight[i].size();

#if defined(__AVX2__) || defined(__AVX__)
        __m256 v_acc = _mm256_setzero_ps();

        size_t byte_i = 0;
        for (; byte_i + 1 < packed_len && j + 7 < cols; byte_i += 2, j += 8) {
            uint8_t b0 = p_row[byte_i];
            uint8_t b1 = p_row[byte_i + 1];

            alignas(32) float multipliers[8] = {
                kLut[b0 & 0b11],
                kLut[(b0 >> 2) & 0b11],
                kLut[(b0 >> 4) & 0b11],
                kLut[(b0 >> 6) & 0b11],
                kLut[b1 & 0b11],
                kLut[(b1 >> 2) & 0b11],
                kLut[(b1 >> 4) & 0b11],
                kLut[(b1 >> 6) & 0b11]
            };

            __m256 v_mult = _mm256_load_ps(multipliers);
            __m256 v_x    = _mm256_loadu_ps(x_ptr + j);
#if defined(__FMA__)
            v_acc = _mm256_fmadd_ps(v_mult, v_x, v_acc);
#else
            v_acc = _mm256_add_ps(v_acc, _mm256_mul_ps(v_mult, v_x));
#endif
        }

        alignas(32) float acc_buf[8];
        _mm256_store_ps(acc_buf, v_acc);
        for (int k = 0; k < 8; ++k) acc += acc_buf[k];

        for (; byte_i < packed_len && j < cols; ++byte_i) {
            uint8_t b = p_row[byte_i];
            for (int shift = 0; shift < 8 && j < cols; shift += 2) {
                acc += kLut[(b >> shift) & 0b11] * x_ptr[j++];
            }
        }
#else
        for (size_t byte_i = 0; byte_i < packed_len && j < cols; ++byte_i) {
            uint8_t b = p_row[byte_i];
            for (int shift = 0; shift < 8 && j < cols; shift += 2) {
                acc += kLut[(b >> shift) & 0b11] * x_ptr[j++];
            }
        }
#endif
        output[i] = acc * scale;
    }
    return output;
}

} // namespace matmul_free
