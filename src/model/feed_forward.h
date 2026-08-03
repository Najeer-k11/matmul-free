#ifndef MATMUL_FREE_FEED_FORWARD_H
#define MATMUL_FREE_FEED_FORWARD_H

#include "../cuda/gpu_buffer.h"
#include <vector>

namespace matmul_free {

struct FFNActivations {
    std::vector<std::vector<float>> hidden_seq;
    std::vector<std::vector<float>> activated_seq;
    std::vector<std::vector<float>> dropout_mask;
};

/**
 * Two-layer feed-forward network with GELU activations.
 * When USE_CUDA is defined, weight1 and weight2 are also stored
 * GPU-resident in gpu_w1/gpu_w2 for zero-copy cuBLAS inference.
 */
struct FFN {
    std::vector<std::vector<float>> weight1; // Input -> hidden weights (CPU mirror)
    std::vector<std::vector<float>> weight2; // Hidden -> output weights (CPU mirror)

#if defined(USE_CUDA)
    mutable GpuMatrix gpu_w1;  // GPU-resident copy of weight1
    mutable GpuMatrix gpu_w2;  // GPU-resident copy of weight2
    mutable bool gpu_dirty = true; // true = GPU copy needs refresh
#endif

    int input_dim = 0;      // Input dimension
    int hidden_dim = 0;     // Hidden layer dimension

    // Explicit copy: copies CPU weights only, resets GPU state
    FFN(const FFN& other)
        : weight1(other.weight1), weight2(other.weight2),
          input_dim(other.input_dim), hidden_dim(other.hidden_dim)
#if defined(USE_CUDA)
          , gpu_dirty(true)
#endif
    {}

    FFN& operator=(const FFN& other) {
        if (this != &other) {
            weight1 = other.weight1;
            weight2 = other.weight2;
            input_dim = other.input_dim;
            hidden_dim = other.hidden_dim;
#if defined(USE_CUDA)
            gpu_dirty = true;
#endif
        }
        return *this;
    }

    FFN() = default;
    FFN(FFN&&) = default;
    FFN& operator=(FFN&&) = default;

    /** Upload weight1/weight2 to GPU VRAM. Call after weight updates. */
    void upload_to_gpu();

    /** Download GPU weights back to CPU (for save/load). */
    void download_from_gpu();

    std::vector<float> forward(const std::vector<float>& x) const;
    /** Batched GPU sequence forward — processes all T tokens in one SGEMM. */
    std::vector<std::vector<float>> forward_sequence(const std::vector<std::vector<float>>& seq,
                                                    FFNActivations* act = nullptr) const;
    std::vector<float> forward_bitlinear(const std::vector<float>& x) const;
    /** Batched GPU sequence 1.58-bit BitLinear ternary forward (MatMul-Free: zero FP32 multiplies). */
    std::vector<std::vector<float>> forward_bitlinear_sequence(const std::vector<std::vector<float>>& seq) const;
    void backward_and_update(const std::vector<float>& x,
                             const std::vector<float>& output_grad,
                             float lr,
                             std::vector<float>& grad_x,
                             const FFNActivations* act = nullptr,
                             size_t token_idx = 0);
    void backward_and_update_sequence(const std::vector<std::vector<float>>& seq_x,
                                      const std::vector<std::vector<float>>& seq_output_grad,
                                      float lr,
                                      std::vector<std::vector<float>>& seq_grad_x,
                                      const FFNActivations* act = nullptr);
    void backward(std::vector<float>& grad_x, const std::vector<float>& output_grad);
};

} // namespace matmul_free

#endif // MATMUL_FREE_FEED_FORWARD_H
