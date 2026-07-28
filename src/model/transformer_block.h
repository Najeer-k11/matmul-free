#ifndef MATMUL_FREE_TRANSFORMER_BLOCK_H
#define MATMUL_FREE_TRANSFORMER_BLOCK_H

#include "attention.h"
#include "feed_forward.h"
#include <vector>

namespace matmul_free {

/**
 * Single transformer encoder block combining self-attention and FFN with Pre-LN RMSNorm.
 */
struct TransformerBlock {
    AttentionLayer attention;  // Self-attention mechanism
    FFN ffn;                   // Feed-forward network
    
    int input_dim = 0;         // Input dimension (token embeddings)
    
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs);
    std::vector<std::vector<float>> forward_bitlinear(const std::vector<std::vector<float>>& inputs);
    void backward_and_update(const std::vector<std::vector<float>>& inputs,
                             const std::vector<std::vector<float>>& output_grads,
                             float lr,
                             std::vector<std::vector<float>>& input_grads);
    void backward(std::vector<std::vector<float>>& input_grads, 
                  const std::vector<std::vector<float>>& output_grads);
};

} // namespace matmul_free

#endif // MATMUL_FREE_TRANSFORMER_BLOCK_H
