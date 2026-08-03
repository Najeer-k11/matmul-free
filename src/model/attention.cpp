#include "attention.h"
#include "../core/math_ops.h"
#include "../core/timing.h"
#include "../quantization/ternary.h"
#include "../cuda/gpu_buffer.h"
#include "../cuda/gpu_ops.h"
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

std::vector<std::vector<float>> AttentionLayer::forward(const std::vector<std::vector<float>>& inputs,
                                       AttentionActivations* act) const {
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());
    
    std::vector<std::vector<float>> output(num_tokens,
                                           std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return output;

    if (act) {
        act->heads.resize(num_heads);
    }

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        std::vector<std::vector<float>> all_k(num_tokens);
        std::vector<std::vector<float>> all_v(num_tokens);
        std::vector<std::vector<float>> all_q(num_tokens);
        std::vector<std::vector<float>> all_weights(num_tokens);

#if defined(USE_CUDA)
        bool use_gpu = !gpu_dirty &&
                       h < static_cast<int>(gpu_key_weights.size()) &&
                       gpu_key_weights[h].is_allocated();
        if (use_gpu) {
            std::vector<std::vector<float>> k_seq = gpu_key_weights[h].gemm_sequence(inputs);
            std::vector<std::vector<float>> v_seq = gpu_value_weights[h].gemm_sequence(inputs);
            std::vector<std::vector<float>> q_seq = gpu_query_weights[h].gemm_sequence(inputs);
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(k_seq[j], j, head_dim);
                all_v[j] = v_seq[j];
                all_q[j] = apply_rope(q_seq[j], j, head_dim);
            }
        } else {
#endif
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(matmul_vector(key_weights[h], inputs[j]), j, head_dim);
                all_v[j] = matmul_vector(value_weights[h], inputs[j]);
                all_q[j] = apply_rope(matmul_vector(query_weights[h], inputs[j]), j, head_dim);
            }
#if defined(USE_CUDA)
        }
#endif

        for (int i = 0; i < num_tokens; ++i) {
            const std::vector<float>& q = all_q[i];

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
            all_weights[i] = weights;

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

        if (act) {
            act->heads[h].all_k = std::move(all_k);
            act->heads[h].all_v = std::move(all_v);
            act->heads[h].all_q = std::move(all_q);
            act->heads[h].all_weights = std::move(all_weights);
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

std::vector<std::vector<float>> AttentionLayer::forward_bitlinear(const std::vector<std::vector<float>>& inputs,
                                       AttentionActivations* act) const {
    (void)act;
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
        std::vector<std::vector<float>> all_q(num_tokens);

#if defined(USE_CUDA)
        if (is_cuda_available() && !inputs.empty()) {
            auto packed_q = pack_ternary_matrix(q_ternary);
            auto packed_k = pack_ternary_matrix(k_ternary);
            auto packed_v = pack_ternary_matrix(v_ternary);
            auto q_seq = cuda_bitlinear_sequence(packed_q, inputs, input_dim, gamma_q);
            auto k_seq = cuda_bitlinear_sequence(packed_k, inputs, input_dim, gamma_k);
            auto v_seq = cuda_bitlinear_sequence(packed_v, inputs, input_dim, gamma_v);
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(k_seq[j], j, head_dim);
                all_v[j] = v_seq[j];
                all_q[j] = apply_rope(q_seq[j], j, head_dim);
            }
        } else {
#endif
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(bitlinear_vector(k_ternary, inputs[j], gamma_k), j, head_dim);
                all_v[j] = bitlinear_vector(v_ternary, inputs[j], gamma_v);
                all_q[j] = apply_rope(bitlinear_vector(q_ternary, inputs[j], gamma_q), j, head_dim);
            }
#if defined(USE_CUDA)
        }
#endif

        for (int i = 0; i < num_tokens; ++i) {
            const std::vector<float>& q = all_q[i];

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
                         std::vector<std::vector<float>>& input_grads,
                         const AttentionActivations* act) {
    timing::ScopedTimerAccumulator timer(timing::g_stats.backward_attn_time_ms);
    int num_tokens = static_cast<int>(inputs.size());
    int num_heads  = static_cast<int>(query_weights.size());
    input_grads.assign(num_tokens, std::vector<float>(input_dim, 0.0f));
    if (num_tokens == 0 || num_heads == 0) return;

    std::vector<std::vector<float>> output_grads_dropped = output_grads;
    if (act != nullptr && !act->dropout_mask.empty()) {
        for (size_t i = 0; i < output_grads_dropped.size(); ++i) {
            for (size_t d = 0; d < output_grads_dropped[i].size(); ++d) {
                if (i < act->dropout_mask.size() && d < act->dropout_mask[i].size()) {
                    output_grads_dropped[i][d] *= act->dropout_mask[i][d];
                }
            }
        }
    }

    for (int h = 0; h < num_heads; ++h) {
        int head_dim = static_cast<int>(query_weights[h].size());
        float scale  = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

        std::vector<std::vector<float>> all_k;
        std::vector<std::vector<float>> all_v;
        std::vector<std::vector<float>> all_q;
        std::vector<std::vector<float>> all_weights;

        if (act && h < static_cast<int>(act->heads.size())) {
            all_k = act->heads[h].all_k;
            all_v = act->heads[h].all_v;
            all_q = act->heads[h].all_q;
            all_weights = act->heads[h].all_weights;
        } else {
            all_k.resize(num_tokens);
            all_v.resize(num_tokens);
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(matmul_vector(key_weights[h], inputs[j]), j, head_dim);
                all_v[j] = matmul_vector(value_weights[h], inputs[j]);
            }
            all_q.resize(num_tokens);
            all_weights.resize(num_tokens);
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
        }

        std::vector<std::vector<float>> grad_v(num_tokens, std::vector<float>(head_dim, 0.0f));
        std::vector<std::vector<float>> grad_q(num_tokens, std::vector<float>(head_dim, 0.0f));
        std::vector<std::vector<float>> grad_k(num_tokens, std::vector<float>(head_dim, 0.0f));

        for (int i = 0; i < num_tokens && i < static_cast<int>(output_grads_dropped.size()); ++i) {
            std::vector<float> grad_head_out(head_dim, 0.0f);
            for (int d = 0; d < head_dim && d < input_dim && d < static_cast<int>(output_grads_dropped[i].size()); ++d) {
                grad_head_out[d] = output_grads_dropped[i][d] / static_cast<float>(num_heads);
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

#if defined(USE_CUDA)
        if (is_cuda_available() && h < static_cast<int>(gpu_query_weights.size()) && gpu_query_weights[h].is_allocated()) {
            if (lr > 0.0f) {
                auto dW_q = gpu_query_weights[h].gemm_sequence_weight_grad(grad_pre_q, inputs);
                auto dW_k = gpu_key_weights[h].gemm_sequence_weight_grad(grad_pre_k, inputs);
                auto dW_v = gpu_value_weights[h].gemm_sequence_weight_grad(grad_v, inputs);

                for (int r = 0; r < head_dim && r < static_cast<int>(dW_q.size()); ++r) {
                    for (int c = 0; c < input_dim && c < static_cast<int>(dW_q[r].size()); ++c) {
                        float dq = std::max(-1.0f, std::min(1.0f, dW_q[r][c]));
                        float dk = std::max(-1.0f, std::min(1.0f, dW_k[r][c]));
                        float dv = std::max(-1.0f, std::min(1.0f, dW_v[r][c]));
                        query_weights[h][r][c] -= lr * dq;
                        key_weights[h][r][c]   -= lr * dk;
                        value_weights[h][r][c] -= lr * dv;
                    }
                }
            }

            auto in_g_q = gpu_query_weights[h].gemm_sequence_transpose(grad_pre_q);
            auto in_g_k = gpu_key_weights[h].gemm_sequence_transpose(grad_pre_k);
            auto in_g_v = gpu_value_weights[h].gemm_sequence_transpose(grad_v);

            for (int i = 0; i < num_tokens; ++i) {
                auto& in_g = input_grads[i];
                for (int c = 0; c < input_dim && c < static_cast<int>(in_g_q[i].size()); ++c) {
                    float acc = in_g_q[i][c] + in_g_k[i][c] + in_g_v[i][c];
                    if (!std::isnan(acc) && !std::isinf(acc)) {
                        in_g[c] += acc;
                    }
                }
            }
            continue;
        }
#endif

        #pragma omp parallel for if(num_tokens > 4)
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

        #pragma omp parallel for if(num_tokens > 4)
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
}

void AttentionLayer::backward(std::vector<std::vector<float>>& input_grads, 
              const std::vector<std::vector<float>>& output_grads) {
    backward_and_update(input_grads, output_grads, 0.0f, input_grads);
}

} // namespace matmul_free

