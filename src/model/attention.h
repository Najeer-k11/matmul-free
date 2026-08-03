#ifndef MATMUL_FREE_ATTENTION_H
#define MATMUL_FREE_ATTENTION_H

#include "kv_cache.h"
#include "../cuda/gpu_buffer.h"
#include <vector>

namespace matmul_free {

/**
 * Multi-head causal self-attention layer.
 * When USE_CUDA is defined, Q/K/V weight matrices are stored GPU-resident
 * in gpu_query/key/value_weights for zero-copy cuBLAS SGEMV.
 */
struct AttentionLayer {
    // Per-head projection matrices (CPU mirror): [num_heads][head_dim][input_dim]
    std::vector<std::vector<std::vector<float>>> query_weights;
    std::vector<std::vector<std::vector<float>>> key_weights;
    std::vector<std::vector<std::vector<float>>> value_weights;

#if defined(USE_CUDA)
    mutable std::vector<GpuMatrix> gpu_query_weights;
    mutable std::vector<GpuMatrix> gpu_key_weights;
    mutable std::vector<GpuMatrix> gpu_value_weights;
    mutable bool gpu_dirty = true;
#endif

    int input_dim = 0;
    int hidden_dim = 0;

    // Explicit copy: copies CPU weights only, resets GPU state
    AttentionLayer() = default;
    AttentionLayer(const AttentionLayer& other)
        : query_weights(other.query_weights), key_weights(other.key_weights),
          value_weights(other.value_weights),
          input_dim(other.input_dim), hidden_dim(other.hidden_dim)
#if defined(USE_CUDA)
          , gpu_dirty(true)
#endif
    {}

    AttentionLayer& operator=(const AttentionLayer& other) {
        if (this != &other) {
            query_weights = other.query_weights;
            key_weights   = other.key_weights;
            value_weights = other.value_weights;
            input_dim     = other.input_dim;
            hidden_dim    = other.hidden_dim;
#if defined(USE_CUDA)
            gpu_dirty = true;
#endif
        }
        return *this;
    }

    AttentionLayer(AttentionLayer&&) = default;
    AttentionLayer& operator=(AttentionLayer&&) = default;

    /** Upload all Q/K/V weight heads to GPU VRAM. */
    void upload_to_gpu();

    /** Download GPU weights back to CPU (for save/load). */
    void download_from_gpu();

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
