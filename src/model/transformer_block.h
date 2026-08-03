#ifndef MATMUL_FREE_TRANSFORMER_BLOCK_H
#define MATMUL_FREE_TRANSFORMER_BLOCK_H

#include "attention.h"
#include "feed_forward.h"
#include <vector>

namespace matmul_free {

struct BlockActivations {
    std::vector<std::vector<float>> norm_inputs;
    std::vector<std::vector<float>> after_attention;
    std::vector<std::vector<float>> after_residual;
    std::vector<std::vector<float>> norm_residual;
    AttentionActivations attn_act;
    FFNActivations ffn_act;
};

/**
 * Single transformer encoder block combining self-attention and FFN with Pre-LN RMSNorm.
 */
struct TransformerBlock {
    AttentionLayer attention;  // Self-attention mechanism
    FFN ffn;                   // Feed-forward network
    
    int input_dim = 0;         // Input dimension (token embeddings)

    /** Upload all weights (attention Q/K/V and FFN W1/W2) to GPU VRAM. */
    void upload_to_gpu();

    /** Download GPU weights back to CPU (for save/load). */
    void download_from_gpu();

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs,
                                           BlockActivations* act = nullptr);
    std::vector<std::vector<float>> forward_bitlinear(const std::vector<std::vector<float>>& inputs,
                                                     BlockActivations* act = nullptr);
    std::vector<std::vector<float>> forward_cached(const std::vector<std::vector<float>>& inputs,
                                                   LayerKVCache& cache,
                                                   int start_pos = 0,
                                                   bool use_bitlinear = false);
    void backward_and_update(const std::vector<std::vector<float>>& inputs,
                             const std::vector<std::vector<float>>& output_grads,
                             float lr,
                             std::vector<std::vector<float>>& input_grads,
                             const BlockActivations* act = nullptr);
    void backward(std::vector<std::vector<float>>& input_grads, 
                  const std::vector<std::vector<float>>& output_grads);
};

} // namespace matmul_free

#endif // MATMUL_FREE_TRANSFORMER_BLOCK_H
