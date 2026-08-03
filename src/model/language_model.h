#ifndef MATMUL_FREE_LANGUAGE_MODEL_H
#define MATMUL_FREE_LANGUAGE_MODEL_H

#include "config.h"
#include "transformer_block.h"
#include "../tokenizer/bpe_tokenizer.h"
#include "../tokenizer/byte_tokenizer.h"
#include <vector>
#include <string>

namespace matmul_free {

class LanguageModel {
public:
    std::vector<std::vector<float>> token_embeddings;   // Token embeddings
    std::vector<TransformerBlock>   transformer_blocks; // Transformer encoder blocks
    std::vector<std::vector<float>> vocab_projection;   // Linear output layer
    ModelConfig                     config_;

    explicit LanguageModel(const ModelConfig& config);

    void resize_vocab(int new_vocab_size);
    std::vector<float> get_logits(const std::vector<float>& hidden) const;
    std::vector<float> get_logits_bitlinear(const std::vector<float>& hidden) const;
    void clip_grad_norm(std::vector<std::vector<float>>& grads, float max_norm = 1.0f) const;

    void train(const std::vector<std::string>& training_data, 
               int epochs = 50, float initial_learning_rate = 0.025f, bool use_qat = false,
               const BPETokenizer* bpe = nullptr, int patience = 8);

    std::string generate(const std::string& input_text, int max_length = 20,
                         float temperature = 1.0f, int top_k = 0, float top_p = 1.0f,
                         bool use_bitlinear = false, const BPETokenizer* bpe = nullptr);

    std::string generate_fast(const std::string& input_text, int max_length = 20,
                              float temperature = 1.0f, int top_k = 0, float top_p = 1.0f,
                              bool use_bitlinear = false, const BPETokenizer* bpe = nullptr);

    float compute_loss(const std::vector<std::vector<float>>& seq, 
                       const std::vector<int>& target_tokens);
    float compute_loss(const std::vector<std::vector<float>>& input_seq, 
                       const std::vector<std::vector<float>>& target_seq);

    std::vector<std::vector<float>> encode_text(const std::string& text);
    std::string decode_sequence(const std::vector<std::vector<float>>& sequences);

    bool save_model(const std::string& filepath) const;
    bool load_model(const std::string& filepath);
};

} // namespace matmul_free

#endif // MATMUL_FREE_LANGUAGE_MODEL_H
