#include "feed_forward.h"
#include "../core/math_ops.h"
#include "../core/timing.h"
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
std::vector<std::vector<float>> FFN::forward_sequence(const std::vector<std::vector<float>>& seq, FFNActivations* act) const {
#if defined(USE_CUDA)
    if (!gpu_dirty && gpu_w1.is_allocated() && gpu_w2.is_allocated()) {
        std::vector<std::vector<float>> hidden_seq = gpu_w1.gemm_sequence(seq);
        std::vector<std::vector<float>> activated_seq = hidden_seq;
        for (auto& row : activated_seq)
            for (auto& v : row) v = gelu(v);
        auto out = gpu_w2.gemm_sequence(activated_seq);
        if (act) {
            act->hidden_seq = std::move(hidden_seq);
            act->activated_seq = std::move(activated_seq);
        }
        return out;
    }
#endif
    std::vector<std::vector<float>> out(seq.size());
    std::vector<std::vector<float>> hidden_seq(seq.size());
    std::vector<std::vector<float>> activated_seq(seq.size());
    for (size_t t = 0; t < seq.size(); ++t) {
        hidden_seq[t] = matmul_vector(weight1, seq[t]);
        activated_seq[t].resize(hidden_seq[t].size());
        for (size_t i = 0; i < hidden_seq[t].size(); ++i)
            activated_seq[t][i] = gelu(hidden_seq[t][i]);
        out[t] = matmul_vector(weight2, activated_seq[t]);
    }
    if (act) {
        act->hidden_seq = std::move(hidden_seq);
        act->activated_seq = std::move(activated_seq);
    }
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

void FFN::backward_and_update(const std::vector<float>& x,
                             const std::vector<float>& output_grad,
                             float lr,
                             std::vector<float>& grad_x,
                             const FFNActivations* act,
                             size_t token_idx) {
    timing::ScopedTimerAccumulator timer(timing::g_stats.backward_ffn_time_ms);
    int hd = static_cast<int>(weight1.size());
    if (hd == 0 || x.empty()) {
        grad_x.assign(input_dim, 0.0f);
        return;
    }
    
    std::vector<float> hidden;
    std::vector<float> activated;
    if (act && token_idx < act->hidden_seq.size()) {
        hidden = act->hidden_seq[token_idx];
        activated = act->activated_seq[token_idx];
    } else {
        hidden = matmul_vector(weight1, x);
        activated.resize(hidden.size());
        for (size_t i = 0; i < hidden.size(); ++i)
            activated[i] = gelu(hidden[i]);
    }

    std::vector<float> grad_activated(hd, 0.0f);
    #pragma omp parallel for if(hd > 32)
    for (int i = 0; i < hd; ++i) {
        float sum = 0.0f;
        int lim = std::min(static_cast<int>(output_grad.size()), static_cast<int>(weight2.size()));
        for (int j = 0; j < lim; ++j) {
            if (i < static_cast<int>(weight2[j].size())) {
                sum += weight2[j][i] * output_grad[j];
            }
        }
        grad_activated[i] = sum;
    }

    #pragma omp parallel for if(weight2.size() > 16)
    for (size_t i = 0; i < weight2.size(); ++i) {
        if (i >= output_grad.size()) continue;
        float og = output_grad[i];
        if (og == 0.0f) continue;
        auto& row = weight2[i];
        size_t lim = std::min(row.size(), activated.size());
        for (size_t j = 0; j < lim; ++j) {
            float grad = og * activated[j];
            if (std::isnan(grad) || std::isinf(grad)) continue;
            grad = std::max(-1.0f, std::min(1.0f, grad));
            row[j] -= lr * grad;
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
    #pragma omp parallel for if(input_dim > 32)
    for (int j = 0; j < input_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < hd; ++i) {
            if (j < static_cast<int>(weight1[i].size())) {
                sum += weight1[i][j] * grad_hidden[i];
            }
        }
        grad_x[j] = (std::isnan(sum) || std::isinf(sum)) ? 0.0f : sum;
    }

    #pragma omp parallel for if(weight1.size() > 16)
    for (size_t i = 0; i < weight1.size(); ++i) {
        if (i >= grad_hidden.size()) continue;
        float gh = grad_hidden[i];
        if (gh == 0.0f) continue;
        auto& row = weight1[i];
        size_t lim = std::min(row.size(), x.size());
        for (size_t j = 0; j < lim; ++j) {
            float grad = gh * x[j];
            if (std::isnan(grad) || std::isinf(grad)) continue;
            grad = std::max(-1.0f, std::min(1.0f, grad));
            row[j] -= lr * grad;
        }
    }
}

void FFN::backward(std::vector<float>& grad_x, const std::vector<float>& output_grad) {
    backward_and_update(grad_x, output_grad, 0.0f, grad_x);
}

} // namespace matmul_free
