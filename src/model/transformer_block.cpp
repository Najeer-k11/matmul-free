#include "transformer_block.h"
#include "../core/math_ops.h"

namespace matmul_free {

std::vector<std::vector<float>> TransformerBlock::forward(const std::vector<std::vector<float>>& inputs) {
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

std::vector<std::vector<float>> TransformerBlock::forward_bitlinear(const std::vector<std::vector<float>>& inputs) {
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

std::vector<std::vector<float>> TransformerBlock::forward_cached(
    const std::vector<std::vector<float>>& inputs,
    LayerKVCache& cache,
    int start_pos,
    bool use_bitlinear) {
    std::vector<std::vector<float>> norm_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        norm_inputs[i] = rmsnorm(inputs[i]);
    }
    std::vector<std::vector<float>> after_attention = attention.forward_cached(norm_inputs, cache, start_pos);

    std::vector<std::vector<float>> after_residual = inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            after_residual[i][j] += after_attention[i][j];
        }
    }

    std::vector<std::vector<float>> ff_output(after_residual.size());
    for (size_t i = 0; i < after_residual.size(); ++i) {
        std::vector<float> norm_res = rmsnorm(after_residual[i]);
        std::vector<float> ffn_out = use_bitlinear ? ffn.forward_bitlinear(norm_res) : ffn.forward(norm_res);
        ff_output[i].resize(after_residual[i].size());
        for (size_t j = 0; j < after_residual[i].size(); ++j) {
            ff_output[i][j] = after_residual[i][j] + ffn_out[j];
        }
    }

    return ff_output;
}

void TransformerBlock::backward_and_update(const std::vector<std::vector<float>>& inputs,
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
        
        std::vector<float> ffn_grad_presub = rmsnorm_backward(after_residual[i], ffn_grad_in);

        for (int d = 0; d < input_dim && d < static_cast<int>(output_grads[i].size()); ++d) {
            grad_after_residual[i][d] = output_grads[i][d] + (d < static_cast<int>(ffn_grad_presub.size()) ? ffn_grad_presub[d] : 0.0f);
        }
    }

    std::vector<std::vector<float>> attn_grad_in;
    attention.backward_and_update(norm_inputs, grad_after_residual, lr, attn_grad_in);

    input_grads.assign(num_tokens, std::vector<float>(input_dim, 0.0f));
    for (size_t i = 0; i < num_tokens; ++i) {
        std::vector<float> attn_grad_presub = rmsnorm_backward(inputs[i], attn_grad_in[i]);
        for (int d = 0; d < input_dim; ++d) {
            input_grads[i][d] = grad_after_residual[i][d] + (d < static_cast<int>(attn_grad_presub.size()) ? attn_grad_presub[d] : 0.0f);
        }
    }
}

void TransformerBlock::backward(std::vector<std::vector<float>>& input_grads, 
              const std::vector<std::vector<float>>& output_grads) {
    backward_and_update(input_grads, output_grads, 0.0f, input_grads);
}

} // namespace matmul_free
