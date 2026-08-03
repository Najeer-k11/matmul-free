#include "attention.h"
#include "../core/math_ops.h"
#include "../quantization/ternary.h"
#include "../cuda/gpu_buffer.h"
#include <cmath>
#include <algorithm>

namespace matmul_free {

// ── GPU upload / download ────────────────────────────────────────────────────

void AttentionLayer::upload_to_gpu() {
#if defined(USE_CUDA)
    int num_heads = static_cast<int>(query_weights.size());
    gpu_query_weights.resize(num_heads);
    gpu_key_weights.resize(num_heads);
    gpu_value_weights.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        gpu_query_weights[h].to_device(query_weights[h]);
        gpu_key_weights[h].to_device(key_weights[h]);
        gpu_value_weights[h].to_device(value_weights[h]);
    }
    gpu_dirty = false;
#endif
}

void AttentionLayer::download_from_gpu() {
#if defined(USE_CUDA)
    int num_heads = static_cast<int>(gpu_query_weights.size());
    query_weights.resize(num_heads);
    key_weights.resize(num_heads);
    value_weights.resize(num_heads);
    for (int h = 0; h < num_heads; ++h) {
        if (gpu_query_weights[h].is_allocated()) query_weights[h] = gpu_query_weights[h].from_device();
        if (gpu_key_weights[h].is_allocated())   key_weights[h]   = gpu_key_weights[h].from_device();
        if (gpu_value_weights[h].is_allocated())  value_weights[h]  = gpu_value_weights[h].from_device();
    }
#endif
}

std::vector<std::vector<float>> AttentionLayer::forward(const std::vector<std::vector<float>>& inputs) const {
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());
    
    std::vector<std::vector<float>> output(num_tokens,
                                           std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return output;

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        std::vector<std::vector<float>> all_k(num_tokens);
        std::vector<std::vector<float>> all_v(num_tokens);

#if defined(USE_CUDA)
        bool use_gpu = !gpu_dirty &&
                       h < static_cast<int>(gpu_key_weights.size()) &&
                       gpu_key_weights[h].is_allocated();
#else
        constexpr bool use_gpu = false;
#endif

        for (int j = 0; j < num_tokens; ++j) {
#if defined(USE_CUDA)
            if (use_gpu) {
                all_k[j] = apply_rope(gpu_key_weights[h].gemv_cpu(inputs[j]), j, head_dim);
                all_v[j] = gpu_value_weights[h].gemv_cpu(inputs[j]);
            } else {
#endif
                all_k[j] = apply_rope(matmul_vector(key_weights[h], inputs[j]), j, head_dim);
                all_v[j] = matmul_vector(value_weights[h], inputs[j]);
#if defined(USE_CUDA)
            }
#endif
        }

        for (int i = 0; i < num_tokens; ++i) {
#if defined(USE_CUDA)
            std::vector<float> q = apply_rope(
                use_gpu ? gpu_query_weights[h].gemv_cpu(inputs[i]) : matmul_vector(query_weights[h], inputs[i]),
                i, head_dim);
#else
            std::vector<float> q = apply_rope(matmul_vector(query_weights[h], inputs[i]), i, head_dim);
#endif

            std::vector<float> scores(i + 1, 0.0f);
            float max_score = -1e9f;
            for (int j = 0; j <= i; ++j) {
                float dot_val = 0.0f;
                for (int d = 0; d < head_dim && d < static_cast<int>(q.size()) && d < static_cast<int>(all_k[j].size()); ++d) {
                    dot_val += q[d] * all_k[j][d];
                }
                scores[j] = dot_val * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            float sum_exp = 0.0f;
            std::vector<float> weights(i + 1, 0.0f);
            for (int j = 0; j <= i; ++j) {
                weights[j] = std::exp(scores[j] - max_score);
                sum_exp += weights[j];
            }
            for (int j = 0; j <= i; ++j) {
                weights[j] /= (sum_exp + 1e-9f);
            }

            std::vector<float> attended_v(head_dim, 0.0f);
            for (int j = 0; j <= i; ++j) {
                for (int d = 0; d < head_dim && d < static_cast<int>(all_v[j].size()); ++d) {
                    attended_v[d] += weights[j] * all_v[j][d];
                }
            }

            for (int d = 0; d < head_dim && d < input_dim; ++d) {
                output[i][d] += attended_v[d];
            }
        }
    }

    if (num_heads > 0) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int d = 0; d < input_dim; ++d) {
                output[i][d] /= static_cast<float>(num_heads);
            }
        }
    }

    return output;
}

std::vector<std::vector<float>> AttentionLayer::forward_bitlinear(const std::vector<std::vector<float>>& inputs) const {
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());
    
    std::vector<std::vector<float>> output(num_tokens,
                                           std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return output;

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        float gamma_q = 1.0f, gamma_k = 1.0f, gamma_v = 1.0f;
        auto q_ternary = quantize_weights_ternary(query_weights[h], gamma_q);
        auto k_ternary = quantize_weights_ternary(key_weights[h], gamma_k);
        auto v_ternary = quantize_weights_ternary(value_weights[h], gamma_v);

        std::vector<std::vector<float>> all_k(num_tokens);
        std::vector<std::vector<float>> all_v(num_tokens);
        for (int j = 0; j < num_tokens; ++j) {
            all_k[j] = apply_rope(bitlinear_vector(k_ternary, inputs[j], gamma_k), j, head_dim);
            all_v[j] = bitlinear_vector(v_ternary, inputs[j], gamma_v);
        }

        for (int i = 0; i < num_tokens; ++i) {
            std::vector<float> q = apply_rope(bitlinear_vector(q_ternary, inputs[i], gamma_q), i, head_dim);

            std::vector<float> scores(i + 1, 0.0f);
            float max_score = -1e9f;
            for (int j = 0; j <= i; ++j) {
                float dot_val = 0.0f;
                for (int d = 0; d < head_dim && d < static_cast<int>(q.size()) && d < static_cast<int>(all_k[j].size()); ++d) {
                    dot_val += q[d] * all_k[j][d];
                }
                scores[j] = dot_val * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            float sum_exp = 0.0f;
            std::vector<float> weights(i + 1, 0.0f);
            for (int j = 0; j <= i; ++j) {
                weights[j] = std::exp(scores[j] - max_score);
                sum_exp += weights[j];
            }
            for (int j = 0; j <= i; ++j) {
                weights[j] /= (sum_exp + 1e-9f);
            }

            std::vector<float> attended_v(head_dim, 0.0f);
            for (int j = 0; j <= i; ++j) {
                for (int d = 0; d < head_dim && d < static_cast<int>(all_v[j].size()); ++d) {
                    attended_v[d] += weights[j] * all_v[j][d];
                }
            }

            for (int d = 0; d < head_dim && d < input_dim; ++d) {
                output[i][d] += attended_v[d];
            }
        }
    }

    if (num_heads > 0) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int d = 0; d < input_dim; ++d) {
                output[i][d] /= static_cast<float>(num_heads);
            }
        }
    }

    return output;
}

std::vector<std::vector<float>> AttentionLayer::forward_cached(
    const std::vector<std::vector<float>>& inputs,
    LayerKVCache& cache,
    int start_pos) const {
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());

    std::vector<std::vector<float>> output(num_tokens,
                                           std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return output;

    if (static_cast<int>(cache.k.size()) < num_heads) {
        cache.k.resize(num_heads);
        cache.v.resize(num_heads);
    }

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        for (int i = 0; i < num_tokens; ++i) {
            int pos = start_pos + i;
            std::vector<float> k_new = apply_rope(matmul_vector(key_weights[h], inputs[i]), pos, head_dim);
            std::vector<float> v_new = matmul_vector(value_weights[h], inputs[i]);
            cache.k[h].push_back(k_new);
            cache.v[h].push_back(v_new);
        }

        int total_cached = static_cast<int>(cache.k[h].size());

        for (int i = 0; i < num_tokens; ++i) {
            int pos = start_pos + i;
            std::vector<float> q = apply_rope(matmul_vector(query_weights[h], inputs[i]), pos, head_dim);

            int limit = std::min(pos + 1, total_cached);
            std::vector<float> scores(limit, 0.0f);
            float max_score = -1e9f;

            for (int j = 0; j < limit; ++j) {
                float dot_val = 0.0f;
                const auto& kj = cache.k[h][j];
                for (int d = 0; d < head_dim && d < static_cast<int>(q.size()) && d < static_cast<int>(kj.size()); ++d) {
                    dot_val += q[d] * kj[d];
                }
                scores[j] = dot_val * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            float sum_exp = 0.0f;
            std::vector<float> weights(limit, 0.0f);
            for (int j = 0; j < limit; ++j) {
                weights[j] = std::exp(scores[j] - max_score);
                sum_exp += weights[j];
            }
            for (int j = 0; j < limit; ++j) {
                weights[j] /= (sum_exp + 1e-9f);
            }

            std::vector<float> attended_v(head_dim, 0.0f);
            for (int j = 0; j < limit; ++j) {
                const auto& vj = cache.v[h][j];
                for (int d = 0; d < head_dim && d < static_cast<int>(vj.size()); ++d) {
                    attended_v[d] += weights[j] * vj[d];
                }
            }

            for (int d = 0; d < head_dim && d < input_dim; ++d) {
                output[i][d] += attended_v[d];
            }
        }
    }

    if (num_heads > 0) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int d = 0; d < input_dim; ++d) {
                output[i][d] /= static_cast<float>(num_heads);
            }
        }
    }

    return output;
}

std::vector<std::vector<float>> AttentionLayer::forward_bitlinear_cached(
    const std::vector<std::vector<float>>& inputs,
    LayerKVCache& cache,
    int start_pos) const {
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());

    std::vector<std::vector<float>> output(num_tokens,
                                           std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return output;

    if (static_cast<int>(cache.k.size()) < num_heads) {
        cache.k.resize(num_heads);
        cache.v.resize(num_heads);
    }

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        float gamma_q = 1.0f, gamma_k = 1.0f, gamma_v = 1.0f;
        auto q_ternary = quantize_weights_ternary(query_weights[h], gamma_q);
        auto k_ternary = quantize_weights_ternary(key_weights[h], gamma_k);
        auto v_ternary = quantize_weights_ternary(value_weights[h], gamma_v);

        for (int i = 0; i < num_tokens; ++i) {
            int pos = start_pos + i;
            std::vector<float> k_new = apply_rope(bitlinear_vector(k_ternary, inputs[i], gamma_k), pos, head_dim);
            std::vector<float> v_new = bitlinear_vector(v_ternary, inputs[i], gamma_v);
            cache.k[h].push_back(k_new);
            cache.v[h].push_back(v_new);
        }

        int total_cached = static_cast<int>(cache.k[h].size());

        for (int i = 0; i < num_tokens; ++i) {
            int pos = start_pos + i;
            std::vector<float> q = apply_rope(bitlinear_vector(q_ternary, inputs[i], gamma_q), pos, head_dim);

            int limit = std::min(pos + 1, total_cached);
            std::vector<float> scores(limit, 0.0f);
            float max_score = -1e9f;

            for (int j = 0; j < limit; ++j) {
                float dot_val = 0.0f;
                const auto& kj = cache.k[h][j];
                for (int d = 0; d < head_dim && d < static_cast<int>(q.size()) && d < static_cast<int>(kj.size()); ++d) {
                    dot_val += q[d] * kj[d];
                }
                scores[j] = dot_val * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }

            float sum_exp = 0.0f;
            std::vector<float> weights(limit, 0.0f);
            for (int j = 0; j < limit; ++j) {
                weights[j] = std::exp(scores[j] - max_score);
                sum_exp += weights[j];
            }
            for (int j = 0; j < limit; ++j) {
                weights[j] /= (sum_exp + 1e-9f);
            }

            std::vector<float> attended_v(head_dim, 0.0f);
            for (int j = 0; j < limit; ++j) {
                const auto& vj = cache.v[h][j];
                for (int d = 0; d < head_dim && d < static_cast<int>(vj.size()); ++d) {
                    attended_v[d] += weights[j] * vj[d];
                }
            }

            for (int d = 0; d < head_dim && d < input_dim; ++d) {
                output[i][d] += attended_v[d];
            }
        }
    }

    if (num_heads > 0) {
        for (int i = 0; i < num_tokens; ++i) {
            for (int d = 0; d < input_dim; ++d) {
                output[i][d] /= static_cast<float>(num_heads);
            }
        }
    }

    return output;
}

void AttentionLayer::backward_and_update(const std::vector<std::vector<float>>& inputs,
                         const std::vector<std::vector<float>>& output_grads,
                         float lr,
                         std::vector<std::vector<float>>& input_grads) {
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());
    input_grads.assign(num_tokens, std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return;

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale  = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        std::vector<std::vector<float>> all_k(num_tokens);
        std::vector<std::vector<float>> all_v(num_tokens);
        for (int j = 0; j < num_tokens; ++j) {
            all_k[j] = apply_rope(matmul_vector(key_weights[h], inputs[j]), j, head_dim);
            all_v[j] = matmul_vector(value_weights[h], inputs[j]);
        }

        std::vector<std::vector<float>> all_q(num_tokens);
        std::vector<std::vector<float>> all_weights(num_tokens);
        for (int i = 0; i < num_tokens; ++i) {
            all_q[i] = apply_rope(matmul_vector(query_weights[h], inputs[i]), i, head_dim);
            std::vector<float> scores(i + 1, 0.0f);
            float max_score = -1e9f;
            for (int j = 0; j <= i; ++j) {
                float dot_val = 0.0f;
                for (int d = 0; d < head_dim && d < static_cast<int>(all_q[i].size()) && d < static_cast<int>(all_k[j].size()); ++d) {
                    dot_val += all_q[i][d] * all_k[j][d];
                }
                scores[j] = dot_val * scale;
                if (scores[j] > max_score) max_score = scores[j];
            }
            float sum_exp = 0.0f;
            std::vector<float> weights(i + 1, 0.0f);
            for (int j = 0; j <= i; ++j) {
                weights[j] = std::exp(scores[j] - max_score);
                sum_exp += weights[j];
            }
            for (int j = 0; j <= i; ++j) {
                weights[j] /= (sum_exp + 1e-9f);
            }
            all_weights[i] = weights;
        }

        std::vector<std::vector<float>> grad_v(num_tokens, std::vector<float>(head_dim, 0.0f));
        std::vector<std::vector<float>> grad_q(num_tokens, std::vector<float>(head_dim, 0.0f));
        std::vector<std::vector<float>> grad_k(num_tokens, std::vector<float>(head_dim, 0.0f));

        for (int i = 0; i < num_tokens && i < static_cast<int>(output_grads.size()); ++i) {
            std::vector<float> grad_head_out(head_dim, 0.0f);
            for (int d = 0; d < head_dim && d < input_dim && d < static_cast<int>(output_grads[i].size()); ++d) {
                grad_head_out[d] = output_grads[i][d] / static_cast<float>(num_heads);
            }

            std::vector<float> grad_A(i + 1, 0.0f);
            for (int j = 0; j <= i; ++j) {
                for (int d = 0; d < head_dim && d < static_cast<int>(all_v[j].size()); ++d) {
                    grad_v[j][d] += all_weights[i][j] * grad_head_out[d];
                    grad_A[j]    += grad_head_out[d] * all_v[j][d];
                }
            }

            float sum_gA = 0.0f;
            for (int j = 0; j <= i; ++j) {
                sum_gA += all_weights[i][j] * grad_A[j];
            }

            for (int j = 0; j <= i; ++j) {
                float grad_score = all_weights[i][j] * (grad_A[j] - sum_gA) * scale;
                for (int d = 0; d < head_dim && d < static_cast<int>(all_q[i].size()) && d < static_cast<int>(all_k[j].size()); ++d) {
                    grad_q[i][d] += grad_score * all_k[j][d];
                    grad_k[j][d] += grad_score * all_q[i][d];
                }
            }
        }

        std::vector<std::vector<float>> grad_pre_q(num_tokens);
        std::vector<std::vector<float>> grad_pre_k(num_tokens);
        for (int i = 0; i < num_tokens; ++i) {
            grad_pre_q[i] = apply_rope(grad_q[i], -i, head_dim);
            grad_pre_k[i] = apply_rope(grad_k[i], -i, head_dim);
        }

        for (int i = 0; i < num_tokens; ++i) {
            const auto& inp = inputs[i];
            const auto& gq = grad_pre_q[i];
            const auto& gk = grad_pre_k[i];
            const auto& gv = grad_v[i];
            int inp_sz = static_cast<int>(inp.size());
            for (int r = 0; r < head_dim; ++r) {
                float q_r = std::max(-1.0f, std::min(1.0f, (std::isnan(gq[r]) || std::isinf(gq[r])) ? 0.0f : gq[r])) * lr;
                float k_r = std::max(-1.0f, std::min(1.0f, (std::isnan(gk[r]) || std::isinf(gk[r])) ? 0.0f : gk[r])) * lr;
                float v_r = std::max(-1.0f, std::min(1.0f, (std::isnan(gv[r]) || std::isinf(gv[r])) ? 0.0f : gv[r])) * lr;
                auto& qw_row = query_weights[h][r];
                auto& kw_row = key_weights[h][r];
                auto& vw_row = value_weights[h][r];
                for (int c = 0; c < input_dim && c < inp_sz; ++c) {
                    float in_c = (std::isnan(inp[c]) || std::isinf(inp[c])) ? 0.0f : inp[c];
                    float dq = q_r * in_c;
                    float dk = k_r * in_c;
                    float dv = v_r * in_c;
                    if (!std::isnan(dq) && !std::isinf(dq)) qw_row[c] -= dq;
                    if (!std::isnan(dk) && !std::isinf(dk)) kw_row[c] -= dk;
                    if (!std::isnan(dv) && !std::isinf(dv)) vw_row[c] -= dv;
                }
            }
        }

        for (int i = 0; i < num_tokens; ++i) {
            const auto& gq = grad_pre_q[i];
            const auto& gk = grad_pre_k[i];
            const auto& gv = grad_v[i];
            auto& in_g = input_grads[i];
            for (int r = 0; r < head_dim; ++r) {
                float q_r = (std::isnan(gq[r]) || std::isinf(gq[r])) ? 0.0f : std::max(-1.0f, std::min(1.0f, gq[r]));
                float k_r = (std::isnan(gk[r]) || std::isinf(gk[r])) ? 0.0f : std::max(-1.0f, std::min(1.0f, gk[r]));
                float v_r = (std::isnan(gv[r]) || std::isinf(gv[r])) ? 0.0f : std::max(-1.0f, std::min(1.0f, gv[r]));
                const auto& qw_row = query_weights[h][r];
                const auto& kw_row = key_weights[h][r];
                const auto& vw_row = value_weights[h][r];
                for (int c = 0; c < input_dim && c < static_cast<int>(qw_row.size()); ++c) {
                    float acc = qw_row[c] * q_r + kw_row[c] * k_r + vw_row[c] * v_r;
                    if (!std::isnan(acc) && !std::isinf(acc)) {
                        in_g[c] += acc;
                    }
                }
            }
        }
    }
#if defined(USE_CUDA)
    gpu_dirty = true; // weights updated on CPU; GPU mirrors need refresh
#endif
}

void AttentionLayer::backward(std::vector<std::vector<float>>& input_grads, 
              const std::vector<std::vector<float>>& output_grads) {
    backward_and_update(input_grads, output_grads, 0.0f, input_grads);
}

} // namespace matmul_free

