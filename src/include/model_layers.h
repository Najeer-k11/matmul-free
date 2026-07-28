/**
 * Model Layers - MatMul-Free Neural Network Components
 * 
 * Implements neural network layers without traditional matrix multiplication.
 */

#ifndef MODEL_LAYERS_H
#define MODEL_LAYERS_H

#include "math_utils.h"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace matmul_free {

// ============================================================================
// Feed-Forward Network (FFN) - MatMul-Free Implementation
// ============================================================================

/**
 * Two-layer feed-forward network with GELU activations.
 * Implemented using element-wise operations and dot products.
 */
struct FFN {
    std::vector<std::vector<float>> weight1; // Input -> hidden weights
    std::vector<std::vector<float>> weight2; // Hidden -> output weights
    
    int input_dim;      // Input dimension
    int hidden_dim;     // Hidden layer dimension
    
    /**
     * Forward pass through FFN.
     * x -> GELU(x @ W1) @ W2
     */
    std::vector<float> forward(const std::vector<float>& x) const {
        // Step 1: x @ W1 -> hidden layer
        std::vector<float> hidden = matmul_vector(weight1, x);
        
        // Step 2: Apply GELU activation (element-wise)
        std::vector<float> activated(hidden.size());
        for (size_t i = 0; i < hidden.size(); ++i) {
            activated[i] = gelu(hidden[i]);
        }
        
        // Step 3: Activated @ W2 -> output
        return matmul_vector(weight2, activated);
    }

    /**
     * BitLinear 1.58-bit Forward Pass.
     * Applies RMSNorm, quantizes weight matrices to ternary {-1, 0, +1},
     * and executes multiplication-free linear layers.
     */
    std::vector<float> forward_bitlinear(const std::vector<float>& x) const {
        std::vector<float> norm_x = rmsnorm(x);
        
        float scale1 = 1.0f;
        auto t_weight1 = quantize_weights_ternary(weight1, scale1);
        std::vector<float> hidden = bitlinear_vector(t_weight1, norm_x, scale1);
        
        std::vector<float> activated(hidden.size());
        for (size_t i = 0; i < hidden.size(); ++i) {
            activated[i] = gelu(hidden[i]);
        }
        
        std::vector<float> norm_act = rmsnorm(activated);
        float scale2 = 1.0f;
        auto t_weight2 = quantize_weights_ternary(weight2, scale2);
        return bitlinear_vector(t_weight2, norm_act, scale2);
    }
    /**
     * Backward pass with weight gradient calculation & update.
     * Computes grad_x w.r.t input vector x.
     */
    void backward_and_update(const std::vector<float>& x, const std::vector<float>& output_grad, float lr, std::vector<float>& grad_x) {
        int hd = static_cast<int>(weight1.size());
        if (hd == 0 || x.empty()) {
            grad_x.assign(input_dim, 0.0f);
            return;
        }
        
        // Forward intermediate activations for the exact x provided
        std::vector<float> hidden = matmul_vector(weight1, x);
        std::vector<float> activated(hidden.size());
        for (size_t i = 0; i < hidden.size(); ++i) {
            activated[i] = gelu(hidden[i]);
        }

        // Grad wrt activated = output_grad @ weight2
        std::vector<float> grad_activated(hd, 0.0f);
        for (int i = 0; i < hd; ++i) {
            for (size_t j = 0; j < output_grad.size() && j < weight2.size(); ++j) {
                if (i < static_cast<int>(weight2[j].size())) {
                    grad_activated[i] += weight2[j][i] * output_grad[j];
                }
            }
        }

        // Update weight2: dL/dW2[i][j] = output_grad[i] * activated[j]
        for (size_t i = 0; i < weight2.size() && i < output_grad.size(); ++i) {
            for (size_t j = 0; j < weight2[i].size() && j < activated.size(); ++j) {
                float grad = output_grad[i] * activated[j];
                if (std::isnan(grad) || std::isinf(grad)) continue;
                grad = std::max(-1.0f, std::min(1.0f, grad));
                weight2[i][j] -= lr * grad;
            }
        }

        // Grad wrt hidden = grad_activated * GELU'(hidden)
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

        // Grad wrt input x: grad_x = grad_hidden @ W1
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

        // Update weight1: dL/dW1[i][j] = grad_hidden[i] * x[j]
        for (size_t i = 0; i < weight1.size() && i < grad_hidden.size(); ++i) {
            for (size_t j = 0; j < weight1[i].size() && j < x.size(); ++j) {
                float grad = grad_hidden[i] * x[j];
                if (std::isnan(grad) || std::isinf(grad)) continue;
                grad = std::max(-1.0f, std::min(1.0f, grad));
                weight1[i][j] -= lr * grad;
            }
        }
    }

    /**
     * Legacy backward pass overload.
     */
    void backward(std::vector<float>& grad_x, const std::vector<float>& output_grad) {
        backward_and_update(grad_x, output_grad, 0.0f, grad_x);
    }
};

// ============================================================================
// Attention Layer - MatMul-Free Implementation
// ============================================================================

/**
 * Multi-head attention layer without matrix multiplication.
 * Uses element-wise operations and on-the-fly dot products.
 *
 * query_weights / key_weights / value_weights:
 *   One projection matrix per head: [num_heads][head_dim][input_dim]
 */
struct AttentionLayer {
    // Per-head projection matrices: [num_heads][head_dim][input_dim]
    std::vector<std::vector<std::vector<float>>> query_weights;
    std::vector<std::vector<std::vector<float>>> key_weights;
    std::vector<std::vector<std::vector<float>>> value_weights;
    
    int input_dim;      // Input dimension per token
    int hidden_dim;     // Hidden dimension (FFN width)
    
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) const {
        int num_tokens = static_cast<int>(inputs.size());
        int num_heads  = static_cast<int>(query_weights.size());
        
        std::vector<std::vector<float>> output(num_tokens,
                                               std::vector<float>(input_dim, 0.0f));
        if (num_tokens == 0 || num_heads == 0) return output;

        for (int h = 0; h < num_heads; ++h) {
            int head_dim = static_cast<int>(query_weights[h].size());
            float scale = head_dim > 0 ? 1.0f / std::sqrt(static_cast<float>(head_dim)) : 1.0f;

            // Precompute K and V vectors for all sequence positions j = 0..num_tokens-1
            std::vector<std::vector<float>> all_k(num_tokens);
            std::vector<std::vector<float>> all_v(num_tokens);
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(matmul_vector(key_weights[h], inputs[j]), j);
                all_v[j] = matmul_vector(value_weights[h], inputs[j]);
            }

            // For each query position i, compute causal attention over j <= i
            for (int i = 0; i < num_tokens; ++i) {
                std::vector<float> q = apply_rope(matmul_vector(query_weights[h], inputs[i]), i);

                // Dot product scores S_{i, j} = dot(q_i, k_j) * scale for j <= i
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

                // Causal Softmax across positions j <= i
                float sum_exp = 0.0f;
                std::vector<float> weights(i + 1, 0.0f);
                for (int j = 0; j <= i; ++j) {
                    weights[j] = std::exp(scores[j] - max_score);
                    sum_exp += weights[j];
                }
                for (int j = 0; j <= i; ++j) {
                    weights[j] /= (sum_exp + 1e-9f);
                }

                // Weighted sum over V_j for j <= i
                std::vector<float> attended_v(head_dim, 0.0f);
                for (int j = 0; j <= i; ++j) {
                    for (int d = 0; d < head_dim && d < static_cast<int>(all_v[j].size()); ++d) {
                        attended_v[d] += weights[j] * all_v[j][d];
                    }
                }

                // Accumulate into head outputs
                for (int d = 0; d < head_dim && d < input_dim; ++d) {
                    output[i][d] += attended_v[d];
                }
            }
        }

        // Average output over heads
        if (num_heads > 0) {
            for (int i = 0; i < num_tokens; ++i) {
                for (int d = 0; d < input_dim; ++d) {
                    output[i][d] /= static_cast<float>(num_heads);
                }
            }
        }

        return output;
    }

    /**
     * Analytical backward pass with weight gradient calculation & update.
     * Computes input_grads wrt layer inputs.
     */
    void backward_and_update(const std::vector<std::vector<float>>& inputs,
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

            // Recompute K and V for all sequence positions j
            std::vector<std::vector<float>> all_k(num_tokens);
            std::vector<std::vector<float>> all_v(num_tokens);
            for (int j = 0; j < num_tokens; ++j) {
                all_k[j] = apply_rope(matmul_vector(key_weights[h], inputs[j]), j);
                all_v[j] = matmul_vector(value_weights[h], inputs[j]);
            }

            // Forward Q, scores, and attention weights
            std::vector<std::vector<float>> all_q(num_tokens);
            std::vector<std::vector<float>> all_weights(num_tokens);
            for (int i = 0; i < num_tokens; ++i) {
                all_q[i] = apply_rope(matmul_vector(query_weights[h], inputs[i]), i);
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

            // Initialize per-head gradients
            std::vector<std::vector<float>> grad_v(num_tokens, std::vector<float>(head_dim, 0.0f));
            std::vector<std::vector<float>> grad_q(num_tokens, std::vector<float>(head_dim, 0.0f));
            std::vector<std::vector<float>> grad_k(num_tokens, std::vector<float>(head_dim, 0.0f));

            // Backprop through attention output calculation per query position i
            for (int i = 0; i < num_tokens && i < static_cast<int>(output_grads.size()); ++i) {
                // Output gradient for head h at position i (averaged over num_heads in forward pass)
                std::vector<float> grad_head_out(head_dim, 0.0f);
                for (int d = 0; d < head_dim && d < input_dim && d < static_cast<int>(output_grads[i].size()); ++d) {
                    grad_head_out[d] = output_grads[i][d] / static_cast<float>(num_heads);
                }

                // Gradient wrt V_j and weights A_{i, j}
                std::vector<float> grad_A(i + 1, 0.0f);
                for (int j = 0; j <= i; ++j) {
                    for (int d = 0; d < head_dim && d < static_cast<int>(all_v[j].size()); ++d) {
                        grad_v[j][d] += all_weights[i][j] * grad_head_out[d];
                        grad_A[j]    += grad_head_out[d] * all_v[j][d];
                    }
                }

                // Softmax backward for position i
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

            // Apply inverse RoPE (-pos) to Query and Key gradients
            std::vector<std::vector<float>> grad_pre_q(num_tokens);
            std::vector<std::vector<float>> grad_pre_k(num_tokens);
            for (int i = 0; i < num_tokens; ++i) {
                grad_pre_q[i] = apply_rope(grad_q[i], -i);
                grad_pre_k[i] = apply_rope(grad_k[i], -i);
            }

            // Update Projection Weights
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

            // Accumulate input gradients from W_q, W_k, W_v
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

    /**
     * Backward pass overload.
     */
    void backward(std::vector<std::vector<float>>& input_grads, 
                  const std::vector<std::vector<float>>& output_grads) {
        backward_and_update(input_grads, output_grads, 0.0f, input_grads);
    }
};

// ============================================================================
// Transformer Encoder Block - MatMul-Free Implementation
// ============================================================================

/**
 * Single transformer encoder block combining self-attention and FFN.
 */
struct TransformerBlock {
    AttentionLayer attention;  // Self-attention mechanism
    FFN ffn;                   // Feed-forward network
    
    int input_dim;             // Input dimension (token embeddings)
    
    /**
     * Forward pass through transformer block (Pre-LN architecture).
     */
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) {
        std::vector<std::vector<float>> norm_inputs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            norm_inputs[i] = rmsnorm(inputs[i]);
        }
        std::vector<std::vector<float>> after_attention = attention.forward(norm_inputs);
        
        std::vector<std::vector<float>> after_residual = inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                after_residual[i][j] += after_attention[i][j];
            }
        }
        
        std::vector<std::vector<float>> ff_output(after_residual.size());
        for (size_t i = 0; i < after_residual.size(); ++i) {
            std::vector<float> norm_res = rmsnorm(after_residual[i]);
            std::vector<float> ffn_out = ffn.forward(norm_res);
            ff_output[i].resize(after_residual[i].size());
            for (size_t j = 0; j < after_residual[i].size(); ++j) {
                ff_output[i][j] = after_residual[i][j] + ffn_out[j];
            }
        }
        
        return ff_output;
    }

    /**
     * BitLinear 1.58-bit Forward Pass.
     */
    std::vector<std::vector<float>> forward_bitlinear(const std::vector<std::vector<float>>& inputs) {
        std::vector<std::vector<float>> norm_inputs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            norm_inputs[i] = rmsnorm(inputs[i]);
        }
        std::vector<std::vector<float>> after_attention = attention.forward(norm_inputs);
        
        std::vector<std::vector<float>> after_residual = inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                after_residual[i][j] += after_attention[i][j];
            }
        }
        
        std::vector<std::vector<float>> ff_output(after_residual.size());
        for (size_t i = 0; i < after_residual.size(); ++i) {
            std::vector<float> norm_res = rmsnorm(after_residual[i]);
            std::vector<float> ffn_out = ffn.forward_bitlinear(norm_res);
            ff_output[i].resize(after_residual[i].size());
            for (size_t j = 0; j < after_residual[i].size(); ++j) {
                ff_output[i][j] = after_residual[i][j] + ffn_out[j];
            }
        }
        
        return ff_output;
    }

    /**
     * Backward pass & weight update with input gradient chaining.
     */
    void backward_and_update(const std::vector<std::vector<float>>& inputs,
                             const std::vector<std::vector<float>>& output_grads,
                             float lr,
                             std::vector<std::vector<float>>& input_grads) {
        std::vector<std::vector<float>> norm_inputs(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            norm_inputs[i] = rmsnorm(inputs[i]);
        }
        std::vector<std::vector<float>> after_attention = attention.forward(norm_inputs);
        std::vector<std::vector<float>> after_residual = inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                after_residual[i][j] += after_attention[i][j];
            }
        }

        size_t num_tokens = inputs.size();
        std::vector<std::vector<float>> grad_after_residual(num_tokens, std::vector<float>(input_dim, 0.0f));

        for (size_t i = 0; i < num_tokens && i < output_grads.size(); ++i) {
            std::vector<float> norm_res = rmsnorm(after_residual[i]);
            std::vector<float> ffn_grad_in;
            ffn.backward_and_update(norm_res, output_grads[i], lr, ffn_grad_in);
            
            // Route gradient through RMSNorm backward pass!
            std::vector<float> ffn_grad_presub = rmsnorm_backward(after_residual[i], ffn_grad_in);

            for (int d = 0; d < input_dim && d < static_cast<int>(output_grads[i].size()); ++d) {
                grad_after_residual[i][d] = output_grads[i][d] + (d < static_cast<int>(ffn_grad_presub.size()) ? ffn_grad_presub[d] : 0.0f);
            }
        }

        std::vector<std::vector<float>> attn_grad_in;
        attention.backward_and_update(norm_inputs, grad_after_residual, lr, attn_grad_in);

        // Route gradient through RMSNorm backward pass per token!
        input_grads.assign(num_tokens, std::vector<float>(input_dim, 0.0f));
        for (size_t i = 0; i < num_tokens; ++i) {
            std::vector<float> attn_grad_presub = rmsnorm_backward(inputs[i], attn_grad_in[i]);
            for (int d = 0; d < input_dim; ++d) {
                input_grads[i][d] = grad_after_residual[i][d] + (d < static_cast<int>(attn_grad_presub.size()) ? attn_grad_presub[d] : 0.0f);
            }
        }
    }
};

// Helper for Xavier (Glorot) uniform weight initialization
inline float xavier_init(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return -limit + r * (2.0f * limit);
}

// ============================================================================
// RNN Cell - MatMul-Free Implementation
// ============================================================================

/**
 * Simple LSTM cell without matrix multiplication (uses dot products).
 *
 * weight_ih layout: [4*hidden_dim][input_dim + hidden_dim]
 *   Rows 0..hd-1          -> forget gate
 *   Rows hd..2hd-1        -> input gate
 *   Rows 2hd..3hd-1       -> candidate cell
 *   Rows 3hd..4hd-1       -> output gate
 */
struct LSTMCell {
    std::vector<std::vector<float>> weight_ih; // [4*H x (I+H)]
    std::vector<std::vector<float>> weight_hh; // reserved / not used in simplified version
    
    int input_dim;      // Input dimension  (I)
    int hidden_dim;     // Hidden state dim (H)
    
    struct CellState {
        std::vector<float> h;   // Hidden state
        std::vector<float> c;   // Cell state
    };
    
    /**
     * Forward pass through LSTM cell.
     */
    CellState forward(const std::vector<float>& input,
                      const CellState& prev_state) const {
        int hd = hidden_dim;
        int id = input_dim;
        
        // Concatenate input and previous hidden state: [input | prev_h]
        std::vector<float> combined(id + hd);
        for (int i = 0; i < id; ++i)  combined[i]      = input[i];
        for (int i = 0; i < hd; ++i)  combined[id + i] = prev_state.h[i];
        
        // Gate pre-activations: weight_ih @ combined  [4*hd]
        std::vector<float> gates(4 * hd, 0.0f);
        for (int g = 0; g < 4 * hd; ++g) {
            float sum = 0.0f;
            for (int j = 0; j < id + hd; ++j) {
                sum += weight_ih[g][j] * combined[j];
            }
            gates[g] = sum;
        }
        
        // Apply sigmoid / tanh gate activations
        std::vector<float> forget_gate(hd);
        std::vector<float> input_gate(hd);
        std::vector<float> candidate_cell(hd);
        std::vector<float> output_gate(hd);
        
        for (int i = 0; i < hd; ++i) {
            forget_gate[i]    = 1.0f / (1.0f + std::exp(-gates[i]));
            input_gate[i]     = 1.0f / (1.0f + std::exp(-gates[hd + i]));
            candidate_cell[i] = std::tanh(gates[2 * hd + i]);
            output_gate[i]    = 1.0f / (1.0f + std::exp(-gates[3 * hd + i]));
        }
        
        // c_t = f_t * c_{t-1} + i_t * ~c_t  (element-wise)
        CellState new_state;
        new_state.c.resize(hd);
        new_state.h.resize(hd);
        for (int i = 0; i < hd; ++i) {
            new_state.c[i] = forget_gate[i] * prev_state.c[i]
                           + input_gate[i]  * candidate_cell[i];
            // h_t = o_t * tanh(c_t)
            new_state.h[i] = output_gate[i] * std::tanh(new_state.c[i]);
        }
        
        return new_state;
    }
};

// ============================================================================
// Language Model - MatMul-Free Implementation
// ============================================================================

/**
 * Simple language model using transformer architecture.
 */
class LanguageModel {
public:
    std::vector<std::vector<float>> token_embeddings;   // Token embeddings
    std::vector<TransformerBlock>   transformer_blocks; // Transformer encoder blocks
    LSTMCell                        output_lstm;        // Output recurrent layer
    std::vector<std::vector<float>> vocab_projection;   // Linear output layer
    ModelConfig                     config_;

    LanguageModel(const ModelConfig& config) : config_(config) {
        // Initialize token embeddings (Xavier initialization)
        int vocab_size = 256; // Simple byte-level vocabulary
        for (int i = 0; i < vocab_size; ++i) {
            std::vector<float> embedding(config.hidden_dim);
            for (int j = 0; j < config.hidden_dim; ++j) {
                embedding[j] = xavier_init(1, config.hidden_dim);
            }
            token_embeddings.push_back(embedding);
        }
        
        // Initialize transformer blocks (using num_layers, not max_seq_len)
        int head_dim = config.hidden_dim / config.num_heads;
        for (int b = 0; b < config.num_layers; ++b) {
            TransformerBlock block;
            block.input_dim = config.hidden_dim;
            
            // Initialize attention layer
            AttentionLayer attn;
            attn.input_dim  = config.hidden_dim;
            attn.hidden_dim = config.hidden_dim * 4;
            
            // One projection matrix per head: [head_dim x hidden_dim]
            for (int h = 0; h < config.num_heads; ++h) {
                std::vector<std::vector<float>> q_w(head_dim,
                                                    std::vector<float>(config.hidden_dim));
                std::vector<std::vector<float>> k_w(head_dim,
                                                    std::vector<float>(config.hidden_dim));
                std::vector<std::vector<float>> v_w(head_dim,
                                                    std::vector<float>(config.hidden_dim));
                for (int r = 0; r < head_dim; ++r) {
                    for (int c = 0; c < config.hidden_dim; ++c) {
                        q_w[r][c] = xavier_init(config.hidden_dim, head_dim);
                        k_w[r][c] = xavier_init(config.hidden_dim, head_dim);
                        v_w[r][c] = xavier_init(config.hidden_dim, head_dim);
                    }
                }
                attn.query_weights.push_back(q_w);
                attn.key_weights.push_back(k_w);
                attn.value_weights.push_back(v_w);
            }
            
            block.attention = attn;
            
            // Initialize FFN: weight1 [hidden_dim*4 x hidden_dim]
            //                 weight2 [hidden_dim   x hidden_dim*4]
            FFN ffn;
            ffn.input_dim  = config.hidden_dim;
            ffn.hidden_dim = config.hidden_dim * 4;
            
            ffn.weight1.resize(ffn.hidden_dim, std::vector<float>(ffn.input_dim));
            for (int i = 0; i < ffn.hidden_dim; ++i)
                for (int j = 0; j < ffn.input_dim; ++j)
                    ffn.weight1[i][j] = xavier_init(ffn.input_dim, ffn.hidden_dim);
            
            ffn.weight2.resize(ffn.input_dim, std::vector<float>(ffn.hidden_dim));
            for (int i = 0; i < ffn.input_dim; ++i)
                for (int j = 0; j < ffn.hidden_dim; ++j)
                    ffn.weight2[i][j] = xavier_init(ffn.hidden_dim, ffn.input_dim);
            
            block.ffn = ffn;
            transformer_blocks.push_back(block);
        }
        
        // Initialize output LSTM
        output_lstm.input_dim  = config.hidden_dim;
        output_lstm.hidden_dim = config.hidden_dim;
        
        int lstm_id  = output_lstm.input_dim;
        int lstm_hd  = output_lstm.hidden_dim;
        int combined = lstm_id + lstm_hd;
        // weight_ih: [4*H x (I+H)]
        output_lstm.weight_ih.resize(4 * lstm_hd, std::vector<float>(combined));
        for (int i = 0; i < 4 * lstm_hd; ++i)
            for (int j = 0; j < combined; ++j)
                output_lstm.weight_ih[i][j] = xavier_init(combined, 4 * lstm_hd);

        // Initialize Vocabulary Projection Layer (256 x hidden_dim)
        vocab_projection.resize(256, std::vector<float>(config.hidden_dim));
        for (int i = 0; i < 256; ++i) {
            for (int j = 0; j < config.hidden_dim; ++j) {
                vocab_projection[i][j] = xavier_init(config.hidden_dim, 256);
            }
        }
    }

    /**
     * Dynamically resize vocabulary for BPE tokenization.
     */
    void resize_vocab(int new_vocab_size) {
        if (new_vocab_size <= 0) return;
        int old_emb_size = static_cast<int>(token_embeddings.size());
        token_embeddings.resize(new_vocab_size, std::vector<float>(config_.hidden_dim));
        for (int i = old_emb_size; i < new_vocab_size; ++i) {
            for (int j = 0; j < config_.hidden_dim; ++j) {
                token_embeddings[i][j] = xavier_init(config_.hidden_dim, new_vocab_size);
            }
        }
        
        int old_proj_size = static_cast<int>(vocab_projection.size());
        vocab_projection.resize(new_vocab_size, std::vector<float>(config_.hidden_dim));
        for (int i = old_proj_size; i < new_vocab_size; ++i) {
            for (int j = 0; j < config_.hidden_dim; ++j) {
                vocab_projection[i][j] = xavier_init(config_.hidden_dim, new_vocab_size);
            }
        }
    }
    
    /**
     * Compute logits over 256-vocabulary size from hidden vector.
     */
    std::vector<float> get_logits(const std::vector<float>& hidden) const {
        return matmul_vector(vocab_projection, hidden);
    }

    /**
     * Clip gradients to prevent gradient explosion and overshooting.
     */
    void clip_grad_norm(std::vector<std::vector<float>>& grads, float max_norm = 1.0f) const {
        float sum_sq = 0.0f;
        for (const auto& row : grads) {
            for (float g : row) sum_sq += g * g;
        }
        float norm = std::sqrt(sum_sq + 1e-9f);
        if (norm > max_norm) {
            float scale = max_norm / norm;
            for (auto& row : grads) {
                for (float& g : row) g *= scale;
            }
        }
    }

    // Training methods
    void train(const std::vector<std::string>& training_data, 
               int epochs = 50, float initial_learning_rate = 0.025f, bool use_qat = false,
               const BPETokenizer* bpe = nullptr) {
        if (is_gpu_accelerated()) {
            std::cout << "  [GPU Acceleration: ENABLED]\n";
        } else {
            std::cout << "  [GPU Acceleration: DISABLED (Using CPU)]\n";
        }

        // 90% train / 10% validation split
        size_t val_size = std::max(size_t(1), training_data.size() / 10);
        size_t train_size = training_data.size() - val_size;
        std::vector<std::string> train_set(training_data.begin(), training_data.begin() + train_size);
        std::vector<std::string> val_set(training_data.begin() + train_size, training_data.end());

        float best_train_loss = 1e9f;
        int best_epoch = 0;
        auto best_embeddings = token_embeddings;
        auto best_blocks = transformer_blocks;
        auto best_vocab = vocab_projection;

        float prev_epoch_loss = 1e9f;
        int worsening_streak = 0;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Linear learning rate decay schedule
            float current_lr = initial_learning_rate * (1.0f - static_cast<float>(epoch) / static_cast<float>(epochs));
            if (current_lr < 0.0001f) current_lr = 0.0001f;

            float total_train_loss = 0.0f;
            int train_samples = 0;
            
            for (const auto& text : train_set) {
                std::vector<int> tokens;
                if (bpe != nullptr) {
                    tokens = bpe->encode(text);
                } else {
                    tokens = tokenize_text(text, static_cast<int>(token_embeddings.size()));
                }
                if (tokens.size() <= 1) continue;
                
                std::vector<std::vector<float>> input_seq(tokens.size());
                for (size_t i = 0; i < tokens.size(); ++i) {
                    int id = tokens[i];
                    if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
                        input_seq[i] = token_embeddings[id];
                    } else {
                        input_seq[i].assign(config_.hidden_dim, 0.0f);
                    }
                }

                // Forward pass through transformer blocks
                std::vector<std::vector<std::vector<float>>> layer_inputs(transformer_blocks.size());
                std::vector<std::vector<float>> seq = input_seq;
                for (size_t b = 0; b < transformer_blocks.size(); ++b) {
                    layer_inputs[b] = seq;
                    if (use_qat) {
                        seq = transformer_blocks[b].forward_bitlinear(seq);
                    } else {
                        seq = transformer_blocks[b].forward(seq);
                    }
                }
                
                // Compute cross-entropy loss
                float loss = compute_loss(seq, tokens);
                if (std::isnan(loss) || std::isinf(loss)) {
                    std::cout << "  [WARNING] NaN/Inf loss detected at Epoch " << (epoch + 1) << "! Aborting and restoring best checkpoint.\n";
                    token_embeddings = best_embeddings;
                    transformer_blocks = best_blocks;
                    vocab_projection = best_vocab;
                    return;
                }
                total_train_loss += loss;
                train_samples++;
                
                // Backpropagation through Vocabulary Projection & Transformer Blocks
                std::vector<std::vector<float>> hidden_grads(seq.size(), std::vector<float>(config_.hidden_dim, 0.0f));
                int vocab_sz = static_cast<int>(vocab_projection.size());

                for (size_t t = 0; t + 1 < tokens.size(); ++t) {
                    int target_token = tokens[t + 1];
                    std::vector<float> logits = get_logits(seq[t]);
                    std::vector<float> probs = softmax(logits);
                    
                    std::vector<float> d_logits = probs;
                    if (target_token >= 0 && target_token < vocab_sz) {
                        d_logits[target_token] -= 1.0f;
                    }
                    
                    for (int d = 0; d < config_.hidden_dim; ++d) {
                        float sum = 0.0f;
                        for (int k = 0; k < vocab_sz; ++k) {
                            sum += vocab_projection[k][d] * d_logits[k];
                        }
                        hidden_grads[t][d] = sum;
                    }
                    
                    // Update vocab_projection with L2 weight decay
                    for (int k = 0; k < vocab_sz; ++k) {
                        float dl_k = d_logits[k];
                        if (std::isnan(dl_k) || std::isinf(dl_k)) continue;
                        dl_k = std::max(-1.0f, std::min(1.0f, dl_k));
                        for (int d = 0; d < config_.hidden_dim; ++d) {
                            float s_td = (std::isnan(seq[t][d]) || std::isinf(seq[t][d])) ? 0.0f : seq[t][d];
                            float grad_v = current_lr * (dl_k * s_td + 0.0001f * vocab_projection[k][d]);
                            if (!std::isnan(grad_v) && !std::isinf(grad_v)) {
                                vocab_projection[k][d] -= std::max(-0.1f, std::min(0.1f, grad_v));
                            }
                        }
                    }
                }
                
                clip_grad_norm(hidden_grads, 1.0f);

                // Backprop through transformer blocks
                std::vector<std::vector<float>> curr_grads = hidden_grads;
                for (int b = static_cast<int>(transformer_blocks.size()) - 1; b >= 0; --b) {
                    std::vector<std::vector<float>> next_grads;
                    transformer_blocks[b].backward_and_update(layer_inputs[b], curr_grads, current_lr, next_grads);
                    curr_grads = next_grads;
                }

                // Update token embeddings with L2 weight decay
                for (size_t t = 0; t < tokens.size(); ++t) {
                    int token_idx = tokens[t];
                    if (token_idx >= 0 && token_idx < static_cast<int>(token_embeddings.size())) {
                        for (int d = 0; d < config_.hidden_dim; ++d) {
                            float cg = curr_grads[t][d];
                            if (!std::isnan(cg) && !std::isinf(cg)) {
                                float emb_update = current_lr * (std::max(-1.0f, std::min(1.0f, cg)) + 0.0001f * token_embeddings[token_idx][d]);
                                token_embeddings[token_idx][d] -= emb_update;
                            }
                        }
                    }
                }
            }

            // Compute Validation Loss
            float total_val_loss = 0.0f;
            int val_samples = 0;
            for (const auto& text : val_set) {
                std::vector<int> tokens = bpe != nullptr ? bpe->encode(text) : tokenize_text(text, static_cast<int>(token_embeddings.size()));
                if (tokens.size() <= 1) continue;
                std::vector<std::vector<float>> input_seq(tokens.size());
                for (size_t i = 0; i < tokens.size(); ++i) {
                    int id = tokens[i];
                    if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
                        input_seq[i] = token_embeddings[id];
                    } else {
                        input_seq[i].assign(config_.hidden_dim, 0.0f);
                    }
                }
                std::vector<std::vector<float>> seq = input_seq;
                for (size_t b = 0; b < transformer_blocks.size(); ++b) {
                    seq = use_qat ? transformer_blocks[b].forward_bitlinear(seq) : transformer_blocks[b].forward(seq);
                }
                total_val_loss += compute_loss(seq, tokens);
                val_samples++;
            }

            float avg_train_loss = train_samples > 0 ? total_train_loss / train_samples : 0.0f;
            float avg_val_loss = val_samples > 0 ? total_val_loss / val_samples : avg_train_loss;
            float epoch_pct = ((epoch + 1) * 100.0f) / static_cast<float>(epochs);

            // Track & Save Best Checkpoint by lowest Train Loss
            if (avg_train_loss < best_train_loss) {
                best_train_loss = avg_train_loss;
                best_epoch = epoch + 1;
                best_embeddings = token_embeddings;
                best_blocks = transformer_blocks;
                best_vocab = vocab_projection;
            }

            std::cout << "  [Epoch " << std::setw(3) << (epoch + 1) << "/" << epochs 
                      << " | " << std::setw(5) << std::fixed << std::setprecision(1) << epoch_pct << "%] "
                      << "Train Loss: " << std::setprecision(4) << avg_train_loss 
                      << " | Val Loss: " << avg_val_loss << "\n";

            // Live Generation Diagnostic every 10 epochs
            if ((epoch + 1) % 10 == 0) {
                std::string sample = generate("the ", 12, 0.5f, 3, 0.85f, use_qat, bpe);
                std::cout << "     >> Sample Gen [Epoch " << (epoch + 1) << "]: \"" << sample << "\"\n";
            }

            // Extended Abort Criterion: loss > 15.0 or 4 consecutive worsening epochs
            if (avg_train_loss > prev_epoch_loss && epoch > 5) {
                worsening_streak++;
            } else {
                worsening_streak = 0;
            }

            if (avg_train_loss > 15.0f || worsening_streak >= 4) {
                std::cout << "  [WARNING] Loss explosion/flatline detected (Train Loss: " << avg_train_loss 
                          << ") at Epoch " << (epoch + 1) << "! Aborting and restoring best checkpoint.\n";
                break;
            }

            // Early stopping check if loss improvement < 1e-5
            if (std::abs(prev_epoch_loss - avg_train_loss) < 1e-5f && epoch > 20) {
                std::cout << "  >> Loss converged (delta < 1e-5). Stopping early at epoch " << (epoch + 1) << ".\n";
                break;
            }
            prev_epoch_loss = avg_train_loss;
        }

        // Restore best model checkpoint by train loss
        token_embeddings = best_embeddings;
        transformer_blocks = best_blocks;
        vocab_projection = best_vocab;
        std::cout << "  >> Restored best model checkpoint from Epoch " << best_epoch 
                  << " (Train Loss: " << std::setprecision(4) << best_train_loss << ")!\n";
    }
    
    /**
     * Autoregressive Text Generation Loop.
     * Iteratively predicts and appends next tokens up to max_length.
     * Option use_bitlinear toggles 1.58-bit ternary BitLinear inference.
     */
    std::string generate(const std::string& input_text, int max_length = 20,
                         float temperature = 1.0f, int top_k = 0, float top_p = 1.0f,
                         bool use_bitlinear = false, const BPETokenizer* bpe = nullptr) {
        std::vector<int> tokens;
        if (bpe != nullptr) {
            tokens = bpe->encode(input_text);
        } else {
            tokens = tokenize_text(input_text, static_cast<int>(token_embeddings.size()));
        }
        if (tokens.empty()) return input_text;

        for (int step = 0; step < max_length; ++step) {
            if (static_cast<int>(tokens.size()) >= config_.max_seq_len) break;

            std::vector<std::vector<float>> input_seq(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                int id = tokens[i];
                if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
                    input_seq[i] = token_embeddings[id];
                } else {
                    input_seq[i].assign(config_.hidden_dim, 0.0f);
                }
            }

            std::vector<std::vector<float>> seq = input_seq;
            for (auto& block : transformer_blocks) {
                seq = use_bitlinear ? block.forward_bitlinear(seq) : block.forward(seq);
            }

            if (seq.empty()) break;

            std::vector<float> last_hidden = seq.back();
            std::vector<float> logits = get_logits(last_hidden);
            int vocab_sz = static_cast<int>(logits.size());

            // Mask invalid tokens outside vocabulary
            for (int i = 0; i < vocab_sz; ++i) {
                if (bpe == nullptr && (i < 32 || i > 126)) {
                    logits[i] = -1e9f;
                }
            }

            // Hard rule: candidate exclusion if same token appeared 3+ times consecutively
            int last_tok = -1;
            int streak = 0;
            for (int t : tokens) {
                if (t == last_tok) streak++;
                else { last_tok = t; streak = 1; }
            }
            if (streak >= 3 && last_tok >= 0 && last_tok < vocab_sz) {
                logits[last_tok] = -1e9f; // Hard exclude candidate!
            }

            // Repetition penalty once per distinct token in last 16 tokens (excluding space & pad)
            std::unordered_set<int> distinct_tokens;
            size_t start_r = tokens.size() > 16 ? tokens.size() - 16 : 0;
            for (size_t r = start_r; r < tokens.size(); ++r) {
                distinct_tokens.insert(tokens[r]);
            }
            for (int tok : distinct_tokens) {
                if (tok >= 0 && tok < vocab_sz) {
                    if (tok != 32 && tok != ' ' && tok != 1) {
                        logits[tok] -= 3.0f;
                    }
                }
            }

            int next_token = 0;
            if (temperature <= 0.01f) {
                next_token = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
            } else {
                next_token = sample_logits(logits, temperature, top_k, top_p);
            }

            tokens.push_back(next_token);

            if (next_token == 0 || next_token == '\n') break;
        }

        if (bpe != nullptr) {
            return bpe->decode(tokens);
        } else {
            return detokenize_tokens(tokens);
        }
    }

    /**
     * Compute cross-entropy loss over token sequences.
     */
    float compute_loss(const std::vector<std::vector<float>>& seq, 
                       const std::vector<int>& target_tokens) {
        float total_loss = 0.0f;
        int count = 0;
        
        for (size_t t = 1; t < target_tokens.size() && t - 1 < seq.size(); ++t) {
            std::vector<float> logits = get_logits(seq[t - 1]);
            std::vector<float> probs = softmax(logits);
            
            int target_idx = target_tokens[t];
            if (target_idx >= 0 && target_idx < static_cast<int>(probs.size())) {
                total_loss -= std::log(probs[target_idx] + 1e-9f);
                count++;
            }
        }
        
        return count > 0 ? total_loss / count : 0.0f;
    }

    /**
     * Legacy compute_loss overload for backwards compatibility.
     */
    float compute_loss(const std::vector<std::vector<float>>& input_seq, 
                       const std::vector<std::vector<float>>& target_seq) {
        std::vector<int> dummy_targets(target_seq.size(), 32);
        for (size_t i = 0; i < target_seq.size(); ++i) {
            dummy_targets[i] = target_seq[i].empty() ? 32 : static_cast<int>(target_seq[i][0]);
        }
        return compute_loss(input_seq, dummy_targets);
    }

    /**
     * Encode text to sequence of embedding vectors.
     */
    std::vector<std::vector<float>> encode_text(const std::string& text) {
        std::vector<int> tokens = tokenize_text(text,
                                                static_cast<int>(token_embeddings.size()));
        std::vector<std::vector<float>> sequences(tokens.size());
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            int token_idx = tokens[i];
            if (token_idx >= 0 &&
                token_idx < static_cast<int>(token_embeddings.size())) {
                sequences[i] = token_embeddings[token_idx];
            } else {
                // Unknown token - zero vector
                sequences[i].assign(config_.hidden_dim, 0.0f);
            }
        }
        
        return sequences;
    }

    /**
     * Save model weights and configuration to file checkpoint.
     */
    bool save_model(const std::string& filepath) const {
        std::ofstream file(filepath, std::ios::out);
        if (!file.is_open()) return false;

        file << config_.hidden_dim << " " << config_.num_heads << " "
             << config_.num_layers << " " << config_.max_seq_len << "\n";

        file << token_embeddings.size() << " " << config_.hidden_dim << "\n";
        for (const auto& vec : token_embeddings) {
            for (float v : vec) file << v << " ";
            file << "\n";
        }

        return true;
    }

    /**
     * Load model weights and configuration from file checkpoint.
     */
    bool load_model(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::in);
        if (!file.is_open()) return false;

        file >> config_.hidden_dim >> config_.num_heads >> config_.num_layers >> config_.max_seq_len;

        size_t vocab_sz, emb_dim;
        file >> vocab_sz >> emb_dim;
        token_embeddings.resize(vocab_sz, std::vector<float>(emb_dim));
        for (size_t i = 0; i < vocab_sz; ++i) {
            for (size_t j = 0; j < emb_dim; ++j) {
                file >> token_embeddings[i][j];
            }
        }

        return true;
    }

    /**
     * Decode vector sequence back to text.
     */
    std::string decode_sequence(const std::vector<std::vector<float>>& sequences) {
        std::vector<int> decoded_tokens;
        
        for (const auto& seq : sequences) {
            // Find token with maximum dot-product activation (nearest-neighbour)
            int   best_idx = 0;
            float max_val  = -1e9f;
            
            for (int i = 0; i < static_cast<int>(token_embeddings.size()); ++i) {
                float dot_product = 0.0f;
                for (int j = 0; j < config_.hidden_dim; ++j) {
                    dot_product += seq[j] * token_embeddings[i][j];
                }
                if (dot_product > max_val) {
                    max_val  = dot_product;
                    best_idx = i;
                }
            }
            decoded_tokens.push_back(best_idx);
        }
        
        return detokenize_tokens(decoded_tokens);
    }

};

} // namespace matmul_free

#endif // MODEL_LAYERS_H
