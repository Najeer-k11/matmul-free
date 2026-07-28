#ifndef MATMUL_FREE_MODEL_CONFIG_H
#define MATMUL_FREE_MODEL_CONFIG_H

#include <vector>

namespace matmul_free {

struct ModelConfig {
    int hidden_dim = 128;           // Hidden dimension size
    int num_heads = 4;              // Number of attention heads
    int num_layers = 3;             // Number of transformer layers
    int max_seq_len = 128;          // Maximum sequence length
    float dropout_prob = 0.1f;      // Dropout probability
    bool use_log_sum_exp = true;    // Use log-sum-exp for softmax stability
    
    std::vector<int> vocab_size;    // Vocabulary size per layer
};

} // namespace matmul_free

#endif // MATMUL_FREE_MODEL_CONFIG_H
