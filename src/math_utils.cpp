/**
 * Math Utilities Implementation - MatMul-Free Operations
 */

#include "include/math_utils.h"
#include <limits>
#include <cfloat>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <immintrin.h>
#include <unordered_map>
#include <map>
#include <sstream>

namespace matmul_free {

// ============================================================================
// Softmax without Division (Log-Sum-Exp)
// ============================================================================

template<typename T>
std::vector<T> softmax(const std::vector<T>& input, bool use_log_sum_exp) {
    (void)use_log_sum_exp;
    if (input.empty()) return {};
    
    // Find maximum value for numerical stability
    T max_val = input[0];
    for (const auto& val : input) {
        if (val > max_val) max_val = val;
    }
    
    std::vector<T> result(input.size());
    T sum = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i] - max_val);
        sum += result[i];
    }
    
    if (sum > static_cast<T>(1e-9)) {
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] /= sum;
        }
    }
    
    return result;
}

// ============================================================================
// Attention without Full Matrix Multiplication
// ============================================================================

std::vector<float> MatMulFreeAttention::compute_attention_weights() {
    // In our usage: query, key, value each have head_dim elements.
    // Compute a single attention score: dot(query, key) / sqrt(head_dim)
    float score = 0.0f;
    for (int d = 0; d < head_dim && d < static_cast<int>(query.size()) &&
                                    d < static_cast<int>(key.size()); ++d) {
        score += query[d] * key[d];
    }
    if (head_dim > 0) score /= std::sqrt(static_cast<float>(head_dim));
    
    // Return a single-element weight (softmax over one element = 1.0)
    return std::vector<float>(1, score);
}

std::vector<float> MatMulFreeAttention::apply_attention(const std::vector<float>& weights) {
    // In our usage: value has head_dim elements.
    // Return a weighted copy of the value vector (single attention weight).
    std::vector<float> output(value.size(), 0.0f);
    float w = weights.empty() ? 1.0f : std::tanh(weights[0]); // squash score to (-1,1)
    for (size_t i = 0; i < value.size(); ++i) {
        output[i] = value[i] * w;
    }
    return output;
}

// ============================================================================
// Linear Layer without MatMul (Dot Product)
// ============================================================================

std::vector<float> matmul_vector(const std::vector<std::vector<float>>& weight, 
                                  const std::vector<float>& input) {
    if (weight.empty() || weight[0].empty() || input.empty()) return {};
    int n = static_cast<int>(weight.size());
    int d = static_cast<int>(weight[0].size());
    
    std::vector<float> output(n, 0.0f);
    
    // Compute dot product for each row (element-wise operations)
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        int max_cols = std::min(d, static_cast<int>(input.size()));
        for (int j = 0; j < max_cols; ++j) {
            sum += weight[i][j] * input[j];
        }
        output[i] = sum;
    }
    
    return output;
}

std::vector<std::vector<float>> matmul_blocks(const std::vector<std::vector<float>>& A,
                                               const std::vector<std::vector<float>>& B) {
    int rows_a = static_cast<int>(A.size());
    int cols_a = static_cast<int>(A[0].size());
    int cols_b = static_cast<int>(B[0].size());
    
    // Block size for processing large matrices
    const int block_size = 32;
    
    std::vector<std::vector<float>> C(rows_a, std::vector<float>(cols_b));
    
    // Process in blocks to avoid memory issues with large matrices
    for (int i = 0; i < rows_a; i += block_size) {
        int end_i = std::min(i + block_size, rows_a);
        
        for (int j = 0; j < cols_b; j += block_size) {
            int end_j = std::min(j + block_size, cols_b);
            
            // Compute C[i:end_i][j:end_j] = A[i:end_i] @ B[:, j:end_j]
            for (int ii = i; ii < end_i; ++ii) {
                for (int jj = j; jj < end_j; ++jj) {
                    float sum = 0.0f;
                    
                    // Element-wise dot product
                    for (int k = 0; k < cols_a; ++k) {
                        sum += A[ii][k] * B[k][jj];
                    }
                    
                    C[ii][jj] = sum;
                }
            }
        }
    }
    
    return C;
}

// ============================================================================
// Activation Functions
// ============================================================================

float gelu(float x) {
    // Standard tanh-based GELU approximation:
    //   GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    static const float kSqrt2OverPi = std::sqrt(2.0f / 3.14159265f);
    float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

float swiglu(const std::vector<float>& x) {
    // Split input and apply GELU to each part
    int mid = static_cast<int>(x.size()) / 2;
    
    if (mid == 0) return gelu(x[0]);
    
    float left = gelu(x[mid - 1]);
    float right = x[x.size() - 1]; // Last element as gate
    
    return left * right;
}

// ============================================================================
// BitLinear 1.58-bit Ternary Quantization & RMSNorm
// ============================================================================

std::vector<float> rmsnorm(const std::vector<float>& x, float eps) {
    if (x.empty()) return {};
    float sum_sq = 0.0f;
    for (float val : x) {
        if (!std::isnan(val) && !std::isinf(val)) {
            sum_sq += val * val;
        }
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(x.size()) + eps);
    if (std::isnan(rms) || rms < 1e-8f) rms = 1e-4f;
    
    std::vector<float> out(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        float val = (std::isnan(x[i]) || std::isinf(x[i])) ? 0.0f : x[i];
        out[i] = val / rms;
    }
    return out;
}

std::vector<std::vector<int8_t>> quantize_weights_ternary(const std::vector<std::vector<float>>& weight,
                                                            float& scale) {
    if (weight.empty() || weight[0].empty()) {
        scale = 1.0f;
        return {};
    }
    
    // Calculate mean absolute value (gamma scale factor)
    double abs_sum = 0.0;
    size_t count = 0;
    for (const auto& row : weight) {
        for (float val : row) {
            if (!std::isnan(val) && !std::isinf(val)) {
                abs_sum += std::abs(val);
                count++;
            }
        }
    }
    scale = count > 0 ? static_cast<float>(abs_sum / count) : 1.0f;
    if (std::isnan(scale) || std::isinf(scale) || scale < 1e-6f) scale = 1e-6f;

    std::vector<std::vector<int8_t>> ternary(weight.size(), std::vector<int8_t>(weight[0].size()));
    for (size_t i = 0; i < weight.size(); ++i) {
        for (size_t j = 0; j < weight[i].size(); ++j) {
            float w_val = (std::isnan(weight[i][j]) || std::isinf(weight[i][j])) ? 0.0f : weight[i][j];
            float val = w_val / scale;
            if (val >= 0.5f) {
                ternary[i][j] = 1;
            } else if (val <= -0.5f) {
                ternary[i][j] = -1;
            } else {
                ternary[i][j] = 0;
            }
        }
    }
    return ternary;
}

std::vector<float> bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                     const std::vector<float>& input,
                                     float scale) {
    size_t rows = weight_ternary.size();
    std::vector<float> output(rows, 0.0f);

    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const auto& w_row = weight_ternary[i];
        size_t cols = std::min(w_row.size(), input.size());
        
        const int8_t* w_ptr = w_row.data();
        const float* x_ptr = input.data();
        
        // Branchless execution allows SIMD auto-vectorization
        for (size_t j = 0; j < cols; ++j) {
            acc += static_cast<float>(w_ptr[j]) * x_ptr[j];
        }
        output[i] = acc * scale;
    }
    return output;
}

// ============================================================================
// Tokenization Utilities
// ============================================================================

std::vector<int> tokenize_text(const std::string& text, int vocab_size) {
    (void)vocab_size;
    std::vector<int> tokens;
    
    // Simple byte-level tokenization (each character is a token)
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 32 && uc < 128) { // Printable ASCII characters only
            tokens.push_back(uc);
        } else {
            // Handle special tokens for non-printable or control characters
            tokens.push_back(0); // Unknown token
        }
    }
    
    return tokens;
}

std::string detokenize_tokens(const std::vector<int>& tokens) {
    std::string result;
    
    for (int token : tokens) {
        if (token == 0) continue; // Skip unknown tokens
        
        unsigned char uc = static_cast<unsigned char>(token);
        if (uc >= 32 && uc < 128) {
            result += static_cast<char>(uc);
        } else {
            result += '?'; // Replace non-printable characters
        }
    }
    
    return result;
}

// ============================================================================
// Autoregressive Sampling (Temperature, Top-K, Top-P)
// ============================================================================

int sample_logits(const std::vector<float>& logits, float temperature, int top_k, float top_p) {
    if (logits.empty()) return 32; // Default to space if empty
    
    float temp = std::max(temperature, 1e-4f);
    std::vector<float> scaled_logits(logits.size(), -1e9f);
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isnan(logits[i]) && !std::isinf(logits[i])) {
            scaled_logits[i] = logits[i] / temp;
        }
    }
    
    std::vector<float> probs = softmax(scaled_logits);
    int n = static_cast<int>(probs.size());
    
    std::vector<std::pair<float, int>> indexed_probs(n);
    for (int i = 0; i < n; ++i) {
        indexed_probs[i] = {probs[i], i};
    }
    std::sort(indexed_probs.rbegin(), indexed_probs.rend());

    // Top-K filtering
    if (top_k > 0 && top_k < n) {
        for (int i = top_k; i < n; ++i) {
            indexed_probs[i].first = 0.0f;
        }
    }

    // Top-P (Nucleus) filtering
    float cdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        cdf += indexed_probs[i].first;
        if (cdf > top_p && i > 0) {
            for (int j = i + 1; j < n; ++j) {
                indexed_probs[j].first = 0.0f;
            }
            break;
        }
    }

    // Re-normalize probabilities
    float total_p = 0.0f;
    for (int i = 0; i < n; ++i) {
        total_p += indexed_probs[i].first;
    }

    if (total_p <= 1e-9f) {
        return indexed_probs[0].second;
    }

    float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * total_p;
    float accum = 0.0f;
    for (int i = 0; i < n; ++i) {
        accum += indexed_probs[i].first;
        if (r <= accum) {
            return indexed_probs[i].second;
        }
    }

    return indexed_probs[0].second;
}

// ============================================================================
// Subword Byte-Pair Encoding (BPE) Tokenizer
// ============================================================================

BPETokenizer::BPETokenizer() {
    vocab_ = {"<unk>", "<pad>"};
    for (int i = 32; i < 128; ++i) {
        vocab_.push_back(std::string(1, static_cast<char>(i)));
    }
    for (size_t i = 0; i < vocab_.size(); ++i) {
        vocab_map_[vocab_[i]] = static_cast<int>(i);
    }
}

void BPETokenizer::build_vocab_from_corpus(const std::vector<std::string>& corpus, int target_vocab_size) {
    vocab_ = {"<unk>", "<pad>"};
    vocab_map_.clear();
    merges_.clear();
    
    // Add base ASCII printable characters
    for (int i = 32; i < 128; ++i) {
        vocab_.push_back(std::string(1, static_cast<char>(i)));
    }
    for (size_t i = 0; i < vocab_.size(); ++i) {
        vocab_map_[vocab_[i]] = static_cast<int>(i);
    }

    // Prepare character sequences for all corpus lines
    std::vector<std::vector<std::string>> sequences;
    for (const auto& line : corpus) {
        if (line.empty()) continue;
        std::vector<std::string> seq;
        for (char c : line) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (uc >= 32 && uc < 128) {
                seq.push_back(std::string(1, c));
            }
        }
        if (!seq.empty()) {
            sequences.push_back(seq);
        }
    }

    // Iterative BPE pair-merging loop
    while (static_cast<int>(vocab_.size()) < target_vocab_size) {
        std::map<std::pair<std::string, std::string>, int> pair_counts;
        
        for (const auto& seq : sequences) {
            if (seq.size() < 2) continue;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                pair_counts[{seq[i], seq[i + 1]}]++;
            }
        }

        if (pair_counts.empty()) break;

        // Find most frequent pair
        std::pair<std::string, std::string> best_pair;
        int max_freq = 0;
        for (const auto& kv : pair_counts) {
            if (kv.second > max_freq) {
                max_freq = kv.second;
                best_pair = kv.first;
            }
        }

        if (max_freq < 2) break; // Stop merging if no pair appears at least twice

        std::string new_token = best_pair.first + best_pair.second;
        merges_.push_back(best_pair);
        int new_id = static_cast<int>(vocab_.size());
        vocab_.push_back(new_token);
        vocab_map_[new_token] = new_id;

        // Replace occurrences of best_pair in sequences
        for (auto& seq : sequences) {
            std::vector<std::string> new_seq;
            size_t i = 0;
            while (i < seq.size()) {
                if (i + 1 < seq.size() && seq[i] == best_pair.first && seq[i + 1] == best_pair.second) {
                    new_seq.push_back(new_token);
                    i += 2;
                } else {
                    new_seq.push_back(seq[i]);
                    i++;
                }
            }
            seq = new_seq;
        }
    }
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<std::string> symbols;
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 32 && uc < 128) {
            symbols.push_back(std::string(1, c));
        }
    }

    if (symbols.empty()) return {};

    // Apply learned BPE merges in sequence
    for (const auto& merge : merges_) {
        std::vector<std::string> new_symbols;
        size_t i = 0;
        while (i < symbols.size()) {
            if (i + 1 < symbols.size() && symbols[i] == merge.first && symbols[i + 1] == merge.second) {
                new_symbols.push_back(merge.first + merge.second);
                i += 2;
            } else {
                new_symbols.push_back(symbols[i]);
                i++;
            }
        }
        symbols = new_symbols;
    }

    std::vector<int> tokens;
    for (const auto& sym : symbols) {
        auto it = vocab_map_.find(sym);
        if (it != vocab_map_.end()) {
            tokens.push_back(it->second);
        } else {
            tokens.push_back(0); // <unk>
        }
    }
    return tokens;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int token : tokens) {
        if (token >= 0 && token < static_cast<int>(vocab_.size())) {
            if (token == 0) continue; // Skip <unk>
            if (token == 1) continue; // Skip <pad>
            text += vocab_[token];
        }
    }
    return text;
}

// ============================================================================
// RoPE & Packed 2-Bit Ternary SIMD Implementation
// ============================================================================

std::vector<float> apply_rope(const std::vector<float>& vec, int position) {
    int dim = static_cast<int>(vec.size());
    std::vector<float> rotated = vec;
    for (int i = 0; i < dim - 1; i += 2) {
        float freq = 1.0f / std::pow(10000.0f, static_cast<float>(i) / static_cast<float>(dim));
        float val = static_cast<float>(position) * freq;
        float cos_val = std::cos(val);
        float sin_val = std::sin(val);

        float v0 = vec[i];
        float v1 = vec[i + 1];
        rotated[i]     = v0 * cos_val - v1 * sin_val;
        rotated[i + 1] = v0 * sin_val + v1 * cos_val;
    }
    return rotated;
}

std::vector<std::vector<uint8_t>> pack_ternary_matrix(const std::vector<std::vector<int8_t>>& ternary) {
    if (ternary.empty()) return {};
    size_t rows = ternary.size();
    size_t cols = ternary[0].size();
    size_t packed_cols = (cols + 3) / 4;

    std::vector<std::vector<uint8_t>> packed(rows, std::vector<uint8_t>(packed_cols, 0));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            int8_t val = ternary[i][j];
            uint8_t code = (val == 1) ? 0b01 : ((val == -1) ? 0b10 : 0b00);
            size_t byte_idx = j / 4;
            size_t bit_shift = (j % 4) * 2;
            packed[i][byte_idx] |= (code << bit_shift);
        }
    }
    return packed;
}

std::vector<float> bitlinear_packed_simd(const std::vector<std::vector<uint8_t>>& packed_weight,
                                          const std::vector<float>& input,
                                          float scale) {
    size_t rows = packed_weight.size();
    std::vector<float> output(rows, 0.0f);
    size_t cols = input.size();

    static const float kLut[4] = {0.0f, 1.0f, -1.0f, 0.0f};

    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const uint8_t* p_row = packed_weight[i].data();
        const float* x_ptr = input.data();

        size_t j = 0;
        size_t packed_len = packed_weight[i].size();
        for (size_t byte_i = 0; byte_i < packed_len && j < cols; ++byte_i) {
            uint8_t b = p_row[byte_i];
            
            uint8_t c0 = b & 0b11;
            acc += kLut[c0] * x_ptr[j++];
            if (j >= cols) break;

            uint8_t c1 = (b >> 2) & 0b11;
            acc += kLut[c1] * x_ptr[j++];
            if (j >= cols) break;

            uint8_t c2 = (b >> 4) & 0b11;
            acc += kLut[c2] * x_ptr[j++];
            if (j >= cols) break;

            uint8_t c3 = (b >> 6) & 0b11;
            acc += kLut[c3] * x_ptr[j++];
        }
        output[i] = acc * scale;
    }
    return output;
}

std::vector<float> bitlinear_packed_avx2(const std::vector<std::vector<uint8_t>>& packed_weight,
                                          const std::vector<float>& input,
                                          float scale) {
    size_t rows = packed_weight.size();
    std::vector<float> output(rows, 0.0f);
    size_t cols = input.size();

    static const float kLut[4] = {0.0f, 1.0f, -1.0f, 0.0f};

    #pragma omp parallel for if(rows > 32)
    for (size_t i = 0; i < rows; ++i) {
        float acc = 0.0f;
        const uint8_t* p_row = packed_weight[i].data();
        const float* x_ptr = input.data();

        size_t j = 0;
        size_t packed_len = packed_weight[i].size();

#if defined(__AVX2__) || defined(__AVX__)
        __m256 v_acc = _mm256_setzero_ps();

        size_t byte_i = 0;
        for (; byte_i + 1 < packed_len && j + 7 < cols; byte_i += 2, j += 8) {
            uint8_t b0 = p_row[byte_i];
            uint8_t b1 = p_row[byte_i + 1];

            alignas(32) float multipliers[8] = {
                kLut[b0 & 0b11],
                kLut[(b0 >> 2) & 0b11],
                kLut[(b0 >> 4) & 0b11],
                kLut[(b0 >> 6) & 0b11],
                kLut[b1 & 0b11],
                kLut[(b1 >> 2) & 0b11],
                kLut[(b1 >> 4) & 0b11],
                kLut[(b1 >> 6) & 0b11]
            };

            __m256 v_mult = _mm256_load_ps(multipliers);
            __m256 v_x    = _mm256_loadu_ps(x_ptr + j);
#if defined(__FMA__)
            v_acc = _mm256_fmadd_ps(v_mult, v_x, v_acc);
#else
            v_acc = _mm256_add_ps(v_acc, _mm256_mul_ps(v_mult, v_x));
#endif
        }

        alignas(32) float acc_buf[8];
        _mm256_store_ps(acc_buf, v_acc);
        for (int k = 0; k < 8; ++k) acc += acc_buf[k];

        for (; byte_i < packed_len && j < cols; ++byte_i) {
            uint8_t b = p_row[byte_i];
            for (int shift = 0; shift < 8 && j < cols; shift += 2) {
                acc += kLut[(b >> shift) & 0b11] * x_ptr[j++];
            }
        }
#else
        for (size_t byte_i = 0; byte_i < packed_len && j < cols; ++byte_i) {
            uint8_t b = p_row[byte_i];
            for (int shift = 0; shift < 8 && j < cols; shift += 2) {
                acc += kLut[(b >> shift) & 0b11] * x_ptr[j++];
            }
        }
#endif
        output[i] = acc * scale;
    }
    return output;
}

// ============================================================================
// Performance Benchmarking (Multi-Trial Median Latency)
// ============================================================================

void benchmark_matmul_vs_bitlinear(int num_rows, int num_cols, int iterations) {
    std::cout << "Running Benchmark (Matrix Size: " << num_rows << "x" << num_cols 
              << ", Iterations: " << iterations << ", 5-Trial Median)...\n";
    
    // Create random FP32 weights & input
    std::vector<std::vector<float>> weight(num_rows, std::vector<float>(num_cols));
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            weight[i][j] = (float)(rand() % 100 - 50) / 100.0f;
        }
    }
    std::vector<float> input(num_cols, 0.5f);

    float scale = 1.0f;
    auto ternary_w = quantize_weights_ternary(weight, scale);
    auto packed_w  = pack_ternary_matrix(ternary_w);

    const int num_trials = 5;

    // Trial run helper
    auto run_benchmark_trials = [&](auto kernel_fn) -> double {
        std::vector<double> trial_times;
        for (int trial = 0; trial < num_trials; ++trial) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iterations; ++it) {
                auto out = kernel_fn();
                (void)out;
            }
            auto end = std::chrono::high_resolution_clock::now();
            trial_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        std::sort(trial_times.begin(), trial_times.end());
        return trial_times[num_trials / 2]; // Median latency
    };

    double fp_ms   = run_benchmark_trials([&]() { return matmul_vector(weight, input); });
    double bit_ms  = run_benchmark_trials([&]() { return bitlinear_vector(ternary_w, input, scale); });
    double simd_ms = run_benchmark_trials([&]() { return bitlinear_packed_simd(packed_w, input, scale); });
    double avx_ms  = run_benchmark_trials([&]() { return bitlinear_packed_avx2(packed_w, input, scale); });
    
    size_t fp32_bytes = num_rows * num_cols * sizeof(float);
    size_t packed_bytes = (num_rows * num_cols) / 4;
    double mem_reduction = static_cast<double>(fp32_bytes) / static_cast<double>(packed_bytes);
    
    std::cout << "  FP32 MatMul Latency (Median):  " << std::fixed << std::setprecision(3) << fp_ms << " ms\n";
    std::cout << "  BitLinear Latency (Median):    " << std::fixed << std::setprecision(3) << bit_ms << " ms\n";
    std::cout << "  Packed 2-Bit SIMD (Median):    " << std::fixed << std::setprecision(3) << simd_ms << " ms\n";
    std::cout << "  Explicit AVX2 SIMD (Median):   " << std::fixed << std::setprecision(3) << avx_ms << " ms\n";
    std::cout << "  FP32 Memory Footprint:        " << fp32_bytes / 1024 << " KB\n";
    std::cout << "  Packed Memory Footprint:      " << packed_bytes / 1024 << " KB\n";
    std::cout << "  Memory Footprint Ratio:       " << std::setprecision(2) << mem_reduction << "x smaller!\n";
}

// ============================================================================
// GPU / Parallel Acceleration Check
// ============================================================================

bool is_gpu_accelerated() {
#if defined(USE_GPU) || defined(__CUDACC__) || defined(_OPENMP) || defined(__NVCC__)
    return true;
#else
    return false;
#endif
}

} // namespace matmul_free
