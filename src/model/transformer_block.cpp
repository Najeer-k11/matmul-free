#include "transformer_block.h"
#include "../core/math_ops.h"

namespace matmul_free {

void TransformerBlock::upload_to_gpu() {
    attention.upload_to_gpu();
    ffn.upload_to_gpu();
}

void TransformerBlock::download_from_gpu() {
    attention.download_from_gpu();
    ffn.download_from_gpu();
}

std::vector<std::vector<float>> TransformerBlock::forward(const std::vector<std::vector<float>>& inputs, BlockActivations* act) {
    std::vector<std::vector<float>> norm_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        norm_inputs[i] = rmsnorm(inputs[i]);
    }
    std::vector<std::vector<float>> after_attention = attention.forward(norm_inputs, act ? &act->attn_act : nullptr);
    
    std::vector<std::vector<float>> after_residual = inputs;
    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t j = 0; j < inputs[i].size(); ++j) {
            after_residual[i][j] += after_attention[i][j];
        }
    }

    // RMSNorm all tokens before FFN
    std::vector<std::vector<float>> norm_residual(after_residual.size());
    for (size_t i = 0; i < after_residual.size(); ++i)
        norm_residual[i] = rmsnorm(after_residual[i]);

    // Batched FFN: entire sequence in one GPU SGEMM call
    std::vector<std::vector<float>> ffn_out_seq = ffn.forward_sequence(norm_residual, act ? &act->ffn_act : nullptr);

    std::vector<std::vector<float>> ff_output(after_residual.size());
    for (size_t i = 0; i < after_residual.size(); ++i) {
        ff_output[i].resize(after_residual[i].size());
        const auto& ffn_out = (i < ffn_out_seq.size()) ? ffn_out_seq[i] : norm_residual[i];
        for (size_t j = 0; j < after_residual[i].size(); ++j) {
            ff_output[i][j] = after_residual[i][j] + (j < ffn_out.size() ? ffn_out[j] : 0.0f);
        }
    }

    if (act) {
        act->norm_inputs = std::move(norm_inputs);
        act->after_attention = std::move(after_attention);
        act->after_residual = std::move(after_residual);
        act->norm_residual = std::move(norm_residual);
    }
    
    return ff_output;
}

std::vector<std::vector<float>> TransformerBlock::forward_bitlinear(const std::vector<std::vector<float>>& inputs, BlockActivations* act) {
    (void)act;
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
    
    // RMSNorm all tokens before FFN
    std::vector<std::vector<float>> norm_residual(after_residual.size());
    for (size_t i = 0; i < after_residual.size(); ++i)
        norm_residual[i] = rmsnorm(after_residual[i]);

    // Batched 1.58-bit BitLinear FFN: zero FP32 multiplies on CUDA GPU!
    std::vector<std::vector<float>> ffn_out_seq = ffn.forward_bitlinear_sequence(norm_residual);

    std::vector<std::vector<float>> ff_output(after_residual.size());
    for (size_t i = 0; i < after_residual.size(); ++i) {
        ff_output[i].resize(after_residual[i].size());
        const auto& ffn_out = (i < ffn_out_seq.size()) ? ffn_out_seq[i] : norm_residual[i];
        for (size_t j = 0; j < after_residual[i].size(); ++j) {
            ff_output[i][j] = after_residual[i][j] + (j < ffn_out.size() ? ffn_out[j] : 0.0f);
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
                         std::vector<std::vector<float>>& input_grads,
                         const BlockActivations* act) {
    const std::vector<std::vector<float>>* norm_inputs_ptr = nullptr;
    const std::vector<std::vector<float>>* after_residual_ptr = nullptr;
    const std::vector<std::vector<float>>* norm_residual_ptr = nullptr;

    std::vector<std::vector<float>> norm_inputs_local;
    std::vector<std::vector<float>> after_residual_local;
    std::vector<std::vector<float>> norm_residual_local;

    if (act) {
        norm_inputs_ptr = &act->norm_inputs;
        after_residual_ptr = &act->after_residual;
        norm_residual_ptr = &act->norm_residual;
    } else {
        norm_inputs_local.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) norm_inputs_local[i] = rmsnorm(inputs[i]);
        auto after_attention = attention.forward(norm_inputs_local);
        after_residual_local = inputs;
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) after_residual_local[i][j] += after_attention[i][j];
        }
        norm_residual_local.resize(after_residual_local.size());
        for (size_t i = 0; i < after_residual_local.size(); ++i) norm_residual_local[i] = rmsnorm(after_residual_local[i]);

        norm_inputs_ptr = &norm_inputs_local;
        after_residual_ptr = &after_residual_local;
        norm_residual_ptr = &norm_residual_local;
    }

    const auto& norm_inputs = *norm_inputs_ptr;
    const auto& after_residual = *after_residual_ptr;
    const auto& norm_residual = *norm_residual_ptr;

    size_t num_tokens = inputs.size();
    std::vector<std::vector<float>> ffn_grads_in;
    ffn.backward_and_update_sequence(norm_residual, output_grads, lr, ffn_grads_in, act ? &act->ffn_act : nullptr);

    std::vector<std::vector<float>> grad_after_residual(num_tokens, std::vector<float>(input_dim, 0.0f));
    for (size_t i = 0; i < num_tokens && i < output_grads.size(); ++i) {
        std::vector<float> ffn_grad_presub = rmsnorm_backward(after_residual[i], i < ffn_grads_in.size() ? ffn_grads_in[i] : std::vector<float>(input_dim, 0.0f));

        for (int d = 0; d < input_dim && d < static_cast<int>(output_grads[i].size()); ++d) {
            grad_after_residual[i][d] = output_grads[i][d] + (d < static_cast<int>(ffn_grad_presub.size()) ? ffn_grad_presub[d] : 0.0f);
        }
    }

    std::vector<std::vector<float>> attn_grad_in;
    attention.backward_and_update(norm_inputs, grad_after_residual, lr, attn_grad_in, act ? &act->attn_act : nullptr);

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
