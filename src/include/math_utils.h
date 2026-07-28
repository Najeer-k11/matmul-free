/**
 * Math Utilities - MatMul-Free Operations
 * 
 * This header provides mathematical operations optimized to avoid
 * traditional matrix multiplication (O(n³)) by using:
 * - Element-wise operations where possible
 * - Block-wise processing for large matrices
 * - Numerical stability techniques
 */

#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace matmul_free {

// ============================================================================
// Softmax without Division (using Log-Sum-Exp trick for numerical stability)
// ============================================================================

/**
 * Compute softmax using log-sum-exp trick to avoid division and overflow.
 * 
 * Traditional softmax: exp(x_i) / sum(exp(x_j))
 * Using log-sum-exp: exp(x_i - max_x + log(sum(exp(x_j - max_x))))
 * 
 * @param input Vector of floating-point values
 * @return Vector of softmax probabilities (sum to 1.0)
 */
template<typename T>
std::vector<T> softmax(const std::vector<T>& input, bool use_log_sum_exp = true);

// ============================================================================
// Attention without Full Matrix Multiplication
// ============================================================================

/**
 * Compute attention scores using element-wise operations instead of matmul.
 * 
 * Standard attention: Q @ K^T / sqrt(d_k) + bias
 * Matmul-free version uses element-wise dot products computed on-the-fly.
 * 
 * @param query Query vector (or matrix)
 * @param key Key vector (or matrix)
 * @param value Value vector (or matrix)
 * @param head_dim Dimension per attention head
 * @return Attention weights and transformed values
 */
struct MatMulFreeAttention {
    std::vector<float> query;      // Query vector(s)
    std::vector<float> key;        // Key vector(s)
    std::vector<float> value;      // Value vector(s)
    int head_dim;                  // Dimension per attention head
    
    /**
     * Compute attention weights without explicit matrix multiplication.
     * Uses element-wise operations and on-the-fly dot products.
     */
    std::vector<float> compute_attention_weights();
    
    /**
     * Apply attention to values using computed weights.
     */
    std::vector<float> apply_attention(const std::vector<float>& weights);
};

// ============================================================================
// Linear Layer without MatMul (Dot Product Implementation)
// ============================================================================

/**
 * Matrix-vector multiplication implemented as dot products,
 * avoiding explicit matrix multiplication operations.
 * 
 * @param weight Weight matrix (row-major order)
 * @param input Input vector
 * @return Output vector = weight @ input
 */
std::vector<float> matmul_vector(const std::vector<std::vector<float>>& weight, 
                                  const std::vector<float>& input);

/**
 * Matrix-matrix multiplication using block-wise processing.
 * Splits large matrices into blocks to process element-wise.
 * 
 * @param A First matrix (row-major)
 * @param B Second matrix (row-major)
 * @return C = A @ B
 */
std::vector<std::vector<float>> matmul_blocks(const std::vector<std::vector<float>>& A,
                                               const std::vector<std::vector<float>>& B);

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * GELU activation function (approximated without lookup tables)
 */
float gelu(float x);

/**
 * SwiGLU activation function (used in Transformer-XL and some LLMs)
 */
float swiglu(const std::vector<float>& x);

// ============================================================================
// BitLinear 1.58-bit Ternary Quantization & RMSNorm
// ============================================================================

/**
 * Root Mean Square Normalization (RMSNorm).
 */
std::vector<float> rmsnorm(const std::vector<float>& x, float eps = 1e-5f);

/**
 * Quantize floating point weight matrix into 1.58-bit ternary matrix {-1, 0, +1}.
 * Returns scale factor gamma via reference parameter.
 */
std::vector<std::vector<int8_t>> quantize_weights_ternary(const std::vector<std::vector<float>>& weight,
                                                            float& scale);

/**
 * Perform multiplication-free linear layer using ternary weights and additions.
 */
std::vector<float> bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                     const std::vector<float>& input,
                                     float scale);

// ============================================================================
// Tokenization Utilities
// ============================================================================

/**
 * Simple byte-level tokenization for text processing.
 */
std::vector<int> tokenize_text(const std::string& text, int vocab_size);

/**
 * Detokenize tokens back to text.
 */
std::string detokenize_tokens(const std::vector<int>& tokens);

// ============================================================================
// Autoregressive Sampling (Temperature, Top-K, Top-P)
// ============================================================================

/**
 * Sample a token index from raw logits using Temperature, Top-K, and Top-P (Nucleus) filtering.
 */
int sample_logits(const std::vector<float>& logits, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);

// ============================================================================
// Subword Byte-Pair Encoding (BPE) Tokenizer
// ============================================================================

class BPETokenizer {
public:
    BPETokenizer();
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    std::vector<std::string> vocab_;
};

/**
 * Apply Rotary Positional Embeddings (RoPE) to vector at sequence position.
 */
std::vector<float> apply_rope(const std::vector<float>& vec, int position);

/**
 * Pack 4 ternary values {-1, 0, +1} into 1 byte (uint8_t).
 */
std::vector<std::vector<uint8_t>> pack_ternary_matrix(const std::vector<std::vector<int8_t>>& ternary);

/**
 * Packed ternary execution using unrolled table lookup.
 */
std::vector<float> bitlinear_packed_simd(const std::vector<std::vector<uint8_t>>& packed_weight,
                                          const std::vector<float>& input,
                                          float scale);

/**
 * Packed ternary execution using explicit hardware AVX2 SIMD intrinsics.
 */
std::vector<float> bitlinear_packed_avx2(const std::vector<std::vector<uint8_t>>& packed_weight,
                                          const std::vector<float>& input,
                                          float scale);

/**
 * Check if GPU or parallel hardware acceleration is available and enabled.
 */
bool is_gpu_accelerated();

/**
 * Micro-benchmark comparing traditional floating-point GEMM vs. Ternary BitLinear vs. Explicit AVX2 SIMD.
 */
void benchmark_matmul_vs_bitlinear(int num_rows = 512, int num_cols = 512, int iterations = 100);

// ============================================================================
// Model Configuration
// ============================================================================

struct ModelConfig {
    int hidden_dim = 512;           // Hidden dimension size
    int num_heads = 8;              // Number of attention heads
    int num_layers = 2;             // Number of transformer layers
    int max_seq_len = 512;          // Maximum sequence length
    float dropout_prob = 0.1f;      // Dropout probability
    bool use_log_sum_exp = true;    // Use log-sum-exp for softmax stability
    
    std::vector<int> vocab_size;    // Vocabulary size per layer (for RNN) or single value
};

} // namespace matmul_free

#endif // MATH_UTILS_H
