#ifndef MATMUL_FREE_FEED_FORWARD_H
#define MATMUL_FREE_FEED_FORWARD_H

#include <vector>

namespace matmul_free {

/**
 * Two-layer feed-forward network with GELU activations.
 */
struct FFN {
    std::vector<std::vector<float>> weight1; // Input -> hidden weights
    std::vector<std::vector<float>> weight2; // Hidden -> output weights
    
    int input_dim = 0;      // Input dimension
    int hidden_dim = 0;     // Hidden layer dimension
    
    std::vector<float> forward(const std::vector<float>& x) const;
    std::vector<float> forward_bitlinear(const std::vector<float>& x) const;
    void backward_and_update(const std::vector<float>& x,
                             const std::vector<float>& output_grad,
                             float lr,
                             std::vector<float>& grad_x);
    void backward(std::vector<float>& grad_x, const std::vector<float>& output_grad);
};

} // namespace matmul_free

#endif // MATMUL_FREE_FEED_FORWARD_H
