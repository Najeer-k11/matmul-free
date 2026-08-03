#include "feed_forward.h"
#include "../core/math_ops.h"
#include "../quantization/ternary.h"
#include "../cuda/gpu_buffer.h"
#include "../cuda/gpu_ops.h"
#include <cmath>
#include <algorithm>

namespace matmul_free {

// ── GPU upload / download ────────────────────────────────────────────────────

void FFN::upload_to_gpu() {
#if defined(USE_CUDA)
    gpu_w1.to_device(weight1);
    gpu_w2.to_device(weight2);
    gpu_dirty = false;
#endif
}

void FFN::download_from_gpu() {
#if defined(USE_CUDA)
    if (gpu_w1.is_allocated()) weight1 = gpu_w1.from_device();
    if (gpu_w2.is_allocated()) weight2 = gpu_w2.from_device();
#endif
}

// ── Forward ─────────────────────────────────────────────────────────────────

std::vector<float> FFN::forward(const std::vector<float>& x) const {
#if defined(USE_CUDA)
    if (gpu_w1.is_allocated() && gpu_w2.is_allocated()) {
        // GPU-resident path: single cuBLAS SGEMV per layer
        std::vector<float> hidden = gpu_w1.gemv_cpu(x);
        std::vector<float> activated(hidden.size());
        for (size_t i = 0; i < hidden.size(); ++i)
            activated[i] = gelu(hidden[i]);
        return gpu_w2.gemv_cpu(activated);
    }
#endif
    std::vector<float> hidden = matmul_vector(weight1, x);
    std::vector<float> activated(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i)
        activated[i] = gelu(hidden[i]);
    return matmul_vector(weight2, activated);
}

/**
 * Batched forward: process entire sequence [T tokens] in one cuBLAS SGEMM.
 * Called by TransformerBlock::forward to avoid T round-trips on per-token gemv.
 */
std::vector<std::vector<float>> FFN::forward_sequence(const std::vector<std::vector<float>>& seq) const {
#if defined(USE_CUDA)
    if (gpu_w1.is_allocated() && gpu_w2.is_allocated()) {
        // W1 * seq^T  → hidden [T x hidden_dim]
        std::vector<std::vector<float>> hidden_seq = gpu_w1.gemm_sequence(seq);
        // GELU per-element
        for (auto& row : hidden_seq)
            for (auto& v : row) v = gelu(v);
        // W2 * hidden^T → output [T x input_dim]
        return gpu_w2.gemm_sequence(hidden_seq);
    }
#endif
    // CPU fallback: call per-token
    std::vector<std::vector<float>> out(seq.size());
    for (size_t t = 0; t < seq.size(); ++t)
        out[t] = forward(seq[t]);
    return out;
}

std::vector<float> FFN::forward_bitlinear(const std::vector<float>& x) const {
    std::vector<float> norm_x = rmsnorm(x);
    
    float scale1 = 1.0f;
    auto t_weight1 = quantize_weights_ternary(weight1, scale1);
    std::vector<float> hidden = bitlinear_vector(t_weight1, norm_x, scale1);
    
    std::vector<float> activated(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i)
        activated[i] = gelu(hidden[i]);
    
    std::vector<float> norm_act = rmsnorm(activated);
    float scale2 = 1.0f;
    auto t_weight2 = quantize_weights_ternary(weight2, scale2);
    return bitlinear_vector(t_weight2, norm_act, scale2);
}

std::vector<std::vector<float>> FFN::forward_bitlinear_sequence(const std::vector<std::vector<float>>& seq) const {
#if defined(USE_CUDA)
    if (is_cuda_available() && !seq.empty()) {
        std::vector<std::vector<float>> norm_seq(seq.size());
        for (size_t i = 0; i < seq.size(); ++i) norm_seq[i] = rmsnorm(seq[i]);

        float scale1 = 1.0f;
        auto t_weight1 = quantize_weights_ternary(weight1, scale1);
        auto packed1 = pack_ternary_matrix(t_weight1);
        std::vector<std::vector<float>> hidden_seq = cuda_bitlinear_sequence(packed1, norm_seq, input_dim, scale1);

        for (auto& row : hidden_seq)
            for (auto& val : row) val = gelu(val);

        std::vector<std::vector<float>> norm_act(hidden_seq.size());
        for (size_t i = 0; i < hidden_seq.size(); ++i) norm_act[i] = rmsnorm(hidden_seq[i]);

        float scale2 = 1.0f;
        auto t_weight2 = quantize_weights_ternary(weight2, scale2);
        auto packed2 = pack_ternary_matrix(t_weight2);
        return cuda_bitlinear_sequence(packed2, norm_act, hidden_dim, scale2);
    }
#endif
    std::vector<std::vector<float>> out(seq.size());
    for (size_t t = 0; t < seq.size(); ++t) {
        out[t] = forward_bitlinear(seq[t]);
    }
    return out;
}

// ── Backward ────────────────────────────────────────────────────────────────

void FFN::backward_and_update(const std::vector<float>& x, const std::vector<float>& output_grad, float lr, std::vector<float>& grad_x) {
    int hd = static_cast<int>(weight1.size());
    if (hd == 0 || x.empty()) {
        grad_x.assign(input_dim, 0.0f);
        return;
    }
    
    std::vector<float> hidden = matmul_vector(weight1, x);
    std::vector<float> activated(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i)
        activated[i] = gelu(hidden[i]);

    std::vector<float> grad_activated(hd, 0.0f);
    for (int i = 0; i < hd; ++i) {
        for (size_t j = 0; j < output_grad.size() && j < weight2.size(); ++j) {
            if (i < static_cast<int>(weight2[j].size())) {
                grad_activated[i] += weight2[j][i] * output_grad[j];
            }
        }
    }

    for (size_t i = 0; i < weight2.size() && i < output_grad.size(); ++i) {
        for (size_t j = 0; j < weight2[i].size() && j < activated.size(); ++j) {
            float grad = output_grad[i] * activated[j];
            if (std::isnan(grad) || std::isinf(grad)) continue;
            grad = std::max(-1.0f, std::min(1.0f, grad));
            weight2[i][j] -= lr * grad;
        }
    }

    static const float kSqrt2OverPi = std::sqrt(2.0f / 3.14159265f);
    std::vector<float> grad_hidden(hd);
    for (int i = 0; i < hd; ++i) {
        float x_val  = hidden[i];
        float inner  = kSqrt2OverPi * (x_val + 0.044715f * x_val * x_val * x_val);
        float tv     = std::tanh(inner);
        float sech2  = 1.0f - tv * tv;
        float gelu_deriv = 0.5f * (1.0f + tv) + 0.5f * x_val * sech2 * kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x_val * x_val);
        float gh = gelu_deriv * grad_activated[i];
        grad_hidden[i] = (std::isnan(gh) || std::isinf(gh)) ? 0.0f : gh;
    }

    grad_x.assign(input_dim, 0.0f);
    for (int j = 0; j < input_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < hd; ++i) {
            if (j < static_cast<int>(weight1[i].size())) {
                sum += weight1[i][j] * grad_hidden[i];
            }
        }
        grad_x[j] = (std::isnan(sum) || std::isinf(sum)) ? 0.0f : sum;
    }

    for (size_t i = 0; i < weight1.size() && i < grad_hidden.size(); ++i) {
        for (size_t j = 0; j < weight1[i].size() && j < x.size(); ++j) {
            float grad = grad_hidden[i] * x[j];
            if (std::isnan(grad) || std::isinf(grad)) continue;
            grad = std::max(-1.0f, std::min(1.0f, grad));
            weight1[i][j] -= lr * grad;
        }
    }

    // Mark GPU weights as stale after CPU update — will re-upload next epoch
#if defined(USE_CUDA)
    gpu_dirty = true;
#endif
}

void FFN::backward(std::vector<float>& grad_x, const std::vector<float>& output_grad) {
    backward_and_update(grad_x, output_grad, 0.0f, grad_x);
}

} // namespace matmul_free
