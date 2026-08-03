#ifndef MATMUL_FREE_ATTENTION_H
#define MATMUL_FREE_ATTENTION_H

#include "kv_cache.h"
#include <vector>

namespace matmul_free {

/**
 * Multi-head causal self-attention layer.
 */
struct AttentionLayer {
    // Per-head projection matrices: [num_heads][head_dim][input_dim]
    std::vector<std::vector<std::vector<float>>> query_weights;
    std::vector<std::vector<std::vector<float>>> key_weights;
    std::vector<std::vector<std::vector<float>>> value_weights;
    
    int input_dim = 0;      // Input dimension per token
    int hidden_dim = 0;     // Hidden dimension (FFN width)
    
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& inputs) const;
    std::vector<std::vector<float>> forward_bitlinear(const std::vector<std::vector<float>>& inputs) const;
    std::vector<std::vector<float>> forward_cached(const std::vector<std::vector<float>>& inputs,
                                                  LayerKVCache& cache,
                                                  int start_pos = 0) const;
    std::vector<std::vector<float>> forward_bitlinear_cached(const std::vector<std::vector<float>>& inputs,
                                                            LayerKVCache& cache,
                                                            int start_pos = 0) const;
    void backward_and_update(const std::vector<std::vector<float>>& inputs,
                             const std::vector<std::vector<float>>& output_grads,
                             float lr,
                             std::vector<std::vector<float>>& input_grads);
    void backward(std::vector<std::vector<float>>& input_grads, 
                  const std::vector<std::vector<float>>& output_grads);
};

} // namespace matmul_free

#endif // MATMUL_FREE_ATTENTION_H
