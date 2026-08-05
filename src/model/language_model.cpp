#include "language_model.h"
#include "adamw.h"
#include "../core/math_ops.h"
#include "../core/timing.h"
#include "../sampling/sampling.h"
#include "../benchmark/benchmark.h"
#include "../quantization/ternary.h"
#include "../cuda/gpu_ops.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_set>

namespace matmul_free {

static float xavier_init(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return -limit + r * (2.0f * limit);
}

LanguageModel::LanguageModel(const ModelConfig& config) : config_(config) {
    int vocab_size = 256;
    for (int i = 0; i < vocab_size; ++i) {
        std::vector<float> embedding(config.hidden_dim);
        for (int j = 0; j < config.hidden_dim; ++j) {
            embedding[j] = xavier_init(1, config.hidden_dim);
        }
        token_embeddings.push_back(embedding);
    }
    
    int head_dim = config.hidden_dim / config.num_heads;
    for (int b = 0; b < config.num_layers; ++b) {
        TransformerBlock block;
        block.input_dim = config.hidden_dim;
        
        AttentionLayer attn;
        attn.input_dim  = config.hidden_dim;
        attn.hidden_dim = config.hidden_dim * 4;
        
        for (int h = 0; h < config.num_heads; ++h) {
            std::vector<std::vector<float>> q_w(head_dim, std::vector<float>(config.hidden_dim));
            std::vector<std::vector<float>> k_w(head_dim, std::vector<float>(config.hidden_dim));
            std::vector<std::vector<float>> v_w(head_dim, std::vector<float>(config.hidden_dim));
            for (int r = 0; r < head_dim; ++r) {
                for (int c = 0; c < config.hidden_dim; ++c) {
                    q_w[r][c] = xavier_init(config.hidden_dim, head_dim);
                    k_w[r][c] = xavier_init(config.hidden_dim, head_dim);
                    v_w[r][c] = xavier_init(config.hidden_dim, head_dim);
                }
            }
            attn.query_weights.push_back(q_w);
            attn.key_weights.push_back(k_w);
            attn.value_weights.push_back(v_w);
        }
        
        block.attention = attn;
        
        FFN ffn;
        ffn.input_dim  = config.hidden_dim;
        ffn.hidden_dim = config.hidden_dim * 4;
        
        ffn.weight1.resize(ffn.hidden_dim, std::vector<float>(ffn.input_dim));
        for (int i = 0; i < ffn.hidden_dim; ++i)
            for (int j = 0; j < ffn.input_dim; ++j)
                ffn.weight1[i][j] = xavier_init(ffn.input_dim, ffn.hidden_dim);
        
        ffn.weight2.resize(ffn.input_dim, std::vector<float>(ffn.hidden_dim));
        for (int i = 0; i < ffn.input_dim; ++i)
            for (int j = 0; j < ffn.hidden_dim; ++j)
                ffn.weight2[i][j] = xavier_init(ffn.hidden_dim, ffn.input_dim);
        
        block.ffn = ffn;
        transformer_blocks.push_back(block);
    }

    vocab_projection.resize(256, std::vector<float>(config.hidden_dim));
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < config.hidden_dim; ++j) {
            vocab_projection[i][j] = xavier_init(config.hidden_dim, 256);
        }
    }

    // Upload all weights to GPU VRAM on construction (one-time transfer)
#if defined(USE_CUDA)
    if (is_cuda_available()) {
        for (auto& block : transformer_blocks) {
            block.upload_to_gpu();
        }
    }
#endif
}

void LanguageModel::resize_vocab(int new_vocab_size) {
    if (new_vocab_size <= 0) return;
    int old_emb_size = static_cast<int>(token_embeddings.size());
    token_embeddings.resize(new_vocab_size, std::vector<float>(config_.hidden_dim));
    for (int i = old_emb_size; i < new_vocab_size; ++i) {
        for (int j = 0; j < config_.hidden_dim; ++j) {
            token_embeddings[i][j] = xavier_init(config_.hidden_dim, new_vocab_size);
        }
    }
    
    int old_proj_size = static_cast<int>(vocab_projection.size());
    vocab_projection.resize(new_vocab_size, std::vector<float>(config_.hidden_dim));
    for (int i = old_proj_size; i < new_vocab_size; ++i) {
        for (int j = 0; j < config_.hidden_dim; ++j) {
            vocab_projection[i][j] = xavier_init(config_.hidden_dim, new_vocab_size);
        }
    }
}

std::vector<float> LanguageModel::get_logits(const std::vector<float>& hidden) const {
    return matmul_vector(vocab_projection, hidden);
}

std::vector<float> LanguageModel::get_logits_bitlinear(const std::vector<float>& hidden) const {
    float gamma = 1.0f;
    auto ternary_vocab = quantize_weights_ternary(vocab_projection, gamma);
    return bitlinear_vector(ternary_vocab, hidden, gamma);
}

void LanguageModel::clip_grad_norm(std::vector<std::vector<float>>& grads, float max_norm) const {
    float sum_sq = 0.0f;
    for (const auto& row : grads) {
        for (float g : row) sum_sq += g * g;
    }
    float norm = std::sqrt(sum_sq + 1e-9f);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (auto& row : grads) {
            for (float& g : row) g *= scale;
        }
    }
}

void LanguageModel::train(const std::vector<std::string>& training_data, 
                           int epochs, float initial_learning_rate, bool use_qat,
                           const BPETokenizer* bpe, int patience) {
    if (training_data.empty()) {
        std::cout << "Error: Training data is empty.\n";
        return;
    }

    std::cout << ">> Initializing training sequence with " << training_data.size() << " samples...\n";

#if defined(USE_CUDA)
    if (is_cuda_available()) {
        std::cout << "  [CUDA GPU Acceleration: ACTIVE (NVIDIA GeForce RTX 4060 Target)]\n";
    } else {
        std::cout << "  [CUDA GPU Acceleration: INACTIVE (Fallback to CPU)]\n";
    }
#else
    if (is_openmp_accelerated()) {
        std::cout << "  [OpenMP Multi-Threading: ENABLED]\n";
    } else {
        std::cout << "  [OpenMP Multi-Threading: DISABLED (Single-Threaded)]\n";
    }
#endif

    size_t val_size = std::max(size_t(1), training_data.size() / 10);
    size_t train_size = training_data.size() - val_size;
    std::vector<std::string> train_set(training_data.begin(), training_data.begin() + train_size);
    std::vector<std::string> val_set(training_data.begin() + train_size, training_data.end());

    timing::g_stats.reset();

    float best_val_loss = 1e9f;
    float best_train_loss = 1e9f;
    int best_epoch = 0;
    int no_improve_epochs = 0;
    auto best_embeddings = token_embeddings;
    auto best_blocks = transformer_blocks;
    auto best_vocab = vocab_projection;

    AdamWMatrix adamw_vocab;
    AdamWMatrix adamw_emb;
    int adamw_step = 0;

    float prev_epoch_loss = 1e9f;
    int worsening_streak = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        timing::ScopedTimerAccumulator epoch_timer(timing::g_stats.total_epoch_time_ms);
        float current_lr = initial_learning_rate * (1.0f - static_cast<float>(epoch) / static_cast<float>(epochs));
        if (current_lr < 0.0001f) current_lr = 0.0001f;

        // Re-sync GPU weight mirrors after backprop updated CPU weights
#if defined(USE_CUDA)
        if (is_cuda_available()) {
            for (auto& block : transformer_blocks) {
                block.upload_to_gpu();
            }
        }
#endif

        float total_train_loss = 0.0f;
        int train_samples = 0;
        int total_sentences = static_cast<int>(train_set.size());
        
        constexpr int sync_interval = 16;

        for (size_t sample_idx = 0; sample_idx < train_set.size(); ++sample_idx) {
            const auto& text = train_set[sample_idx];
            std::vector<int> tokens = bpe != nullptr ? bpe->encode(text, true) : tokenize_text(text, static_cast<int>(token_embeddings.size()));
            if (tokens.size() <= 1) continue;
            
            std::vector<std::vector<float>> input_seq(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                int id = tokens[i];
                if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
                    input_seq[i] = token_embeddings[id];
                } else {
                    input_seq[i].assign(config_.hidden_dim, 0.0f);
                }
            }

            std::vector<std::vector<std::vector<float>>> layer_inputs(transformer_blocks.size());
            std::vector<BlockActivations> layer_activations(transformer_blocks.size());
            std::vector<std::vector<float>> seq = input_seq;
            {
                timing::ScopedTimerAccumulator forward_timer(timing::g_stats.forward_time_ms);
                for (size_t b = 0; b < transformer_blocks.size(); ++b) {
                    layer_inputs[b] = seq;
                    seq = use_qat ? transformer_blocks[b].forward_bitlinear(seq, &layer_activations[b])
                                  : transformer_blocks[b].forward(seq, &layer_activations[b]);
                }
            }
            
            float loss = compute_loss(seq, tokens);
            if (std::isnan(loss) || std::isinf(loss)) {
                std::cout << "\n  [WARNING] NaN/Inf loss detected at Epoch " << (epoch + 1) << "! Aborting and restoring best checkpoint.\n";
                token_embeddings = best_embeddings;
                transformer_blocks = best_blocks;
                vocab_projection = best_vocab;
                return;
            }
            total_train_loss += loss;
            train_samples++;
            
            std::vector<std::vector<float>> hidden_grads(seq.size(), std::vector<float>(config_.hidden_dim, 0.0f));
            int vocab_sz = static_cast<int>(vocab_projection.size());
            std::vector<std::vector<float>> vocab_grads(vocab_sz, std::vector<float>(config_.hidden_dim, 0.0f));

            for (size_t t = 0; t + 1 < tokens.size(); ++t) {
                int target_token = tokens[t + 1];
                std::vector<float> logits = get_logits(seq[t]);
                std::vector<float> probs = softmax(logits);
                
                std::vector<float> d_logits = probs;
                if (target_token >= 0 && target_token < vocab_sz) {
                    d_logits[target_token] -= 1.0f;
                }
                
                #pragma omp parallel for if(vocab_sz > 32)
                for (int k = 0; k < vocab_sz; ++k) {
                    float dl_k = d_logits[k];
                    if (std::isnan(dl_k) || std::isinf(dl_k) || dl_k == 0.0f) continue;
                    float* vg_row = vocab_grads[k].data();
                    const float* s_row = seq[t].data();
                    for (int d = 0; d < config_.hidden_dim; ++d) {
                        float s_td = (std::isnan(s_row[d]) || std::isinf(s_row[d])) ? 0.0f : s_row[d];
                        vg_row[d] += dl_k * s_td;
                    }
                }

                #pragma omp parallel for if(config_.hidden_dim > 32)
                for (int d = 0; d < config_.hidden_dim; ++d) {
                    float sum = 0.0f;
                    for (int k = 0; k < vocab_sz; ++k) {
                        float dl_k = d_logits[k];
                        if (std::isnan(dl_k) || std::isinf(dl_k)) continue;
                        sum += vocab_projection[k][d] * dl_k;
                    }
                    hidden_grads[t][d] = sum;
                }
            }

            adamw_step++;
            adamw_vocab.update(vocab_projection, vocab_grads, current_lr, adamw_step);
            
            clip_grad_norm(hidden_grads, 1.0f);

            std::vector<std::vector<float>> curr_grads = hidden_grads;
            for (int b = static_cast<int>(transformer_blocks.size()) - 1; b >= 0; --b) {
                std::vector<std::vector<float>> next_grads;
                transformer_blocks[b].backward_and_update(layer_inputs[b], curr_grads, current_lr, next_grads, &layer_activations[b]);
                curr_grads = next_grads;
            }

            std::vector<std::vector<float>> emb_grads(token_embeddings.size(), std::vector<float>(config_.hidden_dim, 0.0f));
            for (size_t t = 0; t < tokens.size(); ++t) {
                int token_idx = tokens[t];
                if (token_idx >= 0 && token_idx < static_cast<int>(token_embeddings.size())) {
                    for (int d = 0; d < config_.hidden_dim; ++d) {
                        float cg = curr_grads[t][d];
                        if (!std::isnan(cg) && !std::isinf(cg)) {
                            emb_grads[token_idx][d] += cg;
                        }
                    }
                }
            }
            adamw_emb.update(token_embeddings, emb_grads, current_lr, adamw_step);

#if defined(USE_CUDA)
            if (is_cuda_available() && ((sample_idx + 1) % sync_interval == 0 || sample_idx + 1 == train_set.size())) {
                for (auto& block : transformer_blocks) {
                    block.upload_to_gpu();
                }
            }
#endif

            if (sample_idx % 4 == 0 || sample_idx + 1 == train_set.size()) {
                float batch_pct = ((sample_idx + 1) * 100.0f) / static_cast<float>(total_sentences);
                float curr_avg = total_train_loss / static_cast<float>(sample_idx + 1);
                std::cout << "\r  [Epoch " << std::setw(2) << (epoch + 1) << "/" << epochs 
                          << " | Batch " << std::setw(3) << (sample_idx + 1) << "/" << total_sentences
                          << " (" << std::setw(3) << static_cast<int>(batch_pct) << "%)] "
                          << "Loss: " << std::fixed << std::setprecision(4) << curr_avg << " " << std::flush;
            }
        }

        float total_val_loss = 0.0f;
        int val_samples = 0;
        for (const auto& text : val_set) {
            std::vector<int> tokens = bpe != nullptr ? bpe->encode(text, true) : tokenize_text(text, static_cast<int>(token_embeddings.size()));
            if (tokens.size() <= 1) continue;
            std::vector<std::vector<float>> input_seq(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) {
                int id = tokens[i];
                if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
                    input_seq[i] = token_embeddings[id];
                } else {
                    input_seq[i].assign(config_.hidden_dim, 0.0f);
                }
            }
            std::vector<std::vector<float>> seq = input_seq;
            for (size_t b = 0; b < transformer_blocks.size(); ++b) {
                seq = use_qat ? transformer_blocks[b].forward_bitlinear(seq) : transformer_blocks[b].forward(seq);
            }
            total_val_loss += compute_loss(seq, tokens);
            val_samples++;
        }

        float avg_train_loss = train_samples > 0 ? total_train_loss / train_samples : 0.0f;
        float avg_val_loss = val_samples > 0 ? total_val_loss / val_samples : avg_train_loss;
        float epoch_pct = ((epoch + 1) * 100.0f) / static_cast<float>(epochs);

        if (avg_val_loss < best_val_loss) {
            best_val_loss = avg_val_loss;
            best_train_loss = avg_train_loss;
            best_epoch = epoch + 1;
            best_embeddings = token_embeddings;
            best_blocks = transformer_blocks;
            best_vocab = vocab_projection;
            no_improve_epochs = 0;
        } else {
            no_improve_epochs++;
        }

        std::cout << "\r  [Epoch " << std::setw(2) << (epoch + 1) << "/" << epochs 
                  << " | " << std::setw(5) << std::fixed << std::setprecision(1) << epoch_pct << "%] "
                  << "Train Loss: " << std::setprecision(4) << avg_train_loss 
                  << " | Val Loss: " << avg_val_loss << "             \n";

        if ((epoch + 1) % 10 == 0) {
            std::string sample = generate("the ", 12, 0.5f, 3, 0.85f, use_qat, bpe);
            std::cout << "     >> Sample Gen [Epoch " << (epoch + 1) << "]: \"" << sample << "\"\n";
        }

        if (patience > 0 && no_improve_epochs >= patience) {
            std::cout << "  >> Early stopping triggered! Validation loss did not improve for " 
                      << patience << " consecutive epochs (Best Epoch " << best_epoch 
                      << " | Val Loss: " << std::fixed << std::setprecision(4) << best_val_loss << ").\n";
            break;
        }

        if (avg_train_loss > prev_epoch_loss && epoch > 5) {
            worsening_streak++;
        } else {
            worsening_streak = 0;
        }

        if (avg_train_loss > 15.0f || worsening_streak >= 4) {
            std::cout << "  [WARNING] Loss explosion/flatline detected (Train Loss: " << avg_train_loss 
                      << ") at Epoch " << (epoch + 1) << "! Aborting and restoring best checkpoint.\n";
            break;
        }

        if (std::abs(prev_epoch_loss - avg_train_loss) < 1e-5f && epoch > 20) {
            std::cout << "  >> Loss converged (delta < 1e-5). Stopping early at epoch " << (epoch + 1) << ".\n";
            break;
        }
        prev_epoch_loss = avg_train_loss;
    }

    token_embeddings = best_embeddings;
    transformer_blocks = best_blocks;
    vocab_projection = best_vocab;
    std::cout << "  >> Restored best model checkpoint from Epoch " << best_epoch 
              << " (Val Loss: " << std::fixed << std::setprecision(4) << best_val_loss 
              << " | Train Loss: " << best_train_loss << ")!\n";
    timing::print_summary();
}

std::string LanguageModel::generate(const std::string& input_text, int max_length,
                                     float temperature, int top_k, float top_p,
                                     bool use_bitlinear, const BPETokenizer* bpe) {
    std::vector<int> tokens;
    if (bpe != nullptr) {
        tokens = bpe->encode(input_text);
    } else {
        tokens = tokenize_text(input_text, static_cast<int>(token_embeddings.size()));
    }
    if (tokens.empty()) return input_text;

    for (int step = 0; step < max_length; ++step) {
        if (static_cast<int>(tokens.size()) >= config_.max_seq_len) break;

        std::vector<std::vector<float>> input_seq(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            int id = tokens[i];
            if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
                input_seq[i] = token_embeddings[id];
            } else {
                input_seq[i].assign(config_.hidden_dim, 0.0f);
            }
        }

        std::vector<std::vector<float>> seq = input_seq;
        for (auto& block : transformer_blocks) {
            seq = use_bitlinear ? block.forward_bitlinear(seq) : block.forward(seq);
        }

        if (seq.empty()) break;

        std::vector<float> last_hidden = seq.back();
        std::vector<float> logits = use_bitlinear ? get_logits_bitlinear(last_hidden) : get_logits(last_hidden);
        int vocab_sz = static_cast<int>(logits.size());

        for (int i = 0; i < vocab_sz; ++i) {
            if (bpe == nullptr && (i < 32 || i > 126)) {
                logits[i] = -1e9f;
            }
        }

        int last_tok = -1;
        int streak = 0;
        for (int t : tokens) {
            if (t == last_tok) streak++;
            else { last_tok = t; streak = 1; }
        }
        if (streak >= 3 && last_tok >= 0 && last_tok < vocab_sz) {
            logits[last_tok] = -1e9f;
        }

        std::unordered_set<int> distinct_tokens;
        size_t start_r = tokens.size() > 16 ? tokens.size() - 16 : 0;
        for (size_t r = start_r; r < tokens.size(); ++r) {
            distinct_tokens.insert(tokens[r]);
        }
        for (int tok : distinct_tokens) {
            if (tok >= 0 && tok < vocab_sz) {
                if (tok != 32 && tok != ' ' && tok != 1) {
                    logits[tok] -= 3.0f;
                }
            }
        }

        int next_token = 0;
        if (temperature <= 0.01f) {
            next_token = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        } else {
            next_token = sample_logits(logits, temperature, top_k, top_p);
        }

        tokens.push_back(next_token);

        if (next_token == 0 || next_token == 1 || next_token == 2 || next_token == '\n') break;
    }

    if (bpe != nullptr) {
        return bpe->decode(tokens);
    } else {
        return detokenize_tokens(tokens);
    }
}

std::string LanguageModel::generate_fast(const std::string& input_text, int max_length,
                                          float temperature, int top_k, float top_p,
                                          bool use_bitlinear, const BPETokenizer* bpe) {
    std::vector<int> tokens;
    if (bpe != nullptr) {
        tokens = bpe->encode(input_text);
    } else {
        tokens = tokenize_text(input_text, static_cast<int>(token_embeddings.size()));
    }
    if (tokens.empty()) return input_text;

    ModelKVCache cache(transformer_blocks.size());

    // 1. Prefill Phase: process initial prompt sequence
    std::vector<std::vector<float>> prompt_seq(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        int id = tokens[i];
        if (id >= 0 && id < static_cast<int>(token_embeddings.size())) {
            prompt_seq[i] = token_embeddings[id];
        } else {
            prompt_seq[i].assign(config_.hidden_dim, 0.0f);
        }
    }

    std::vector<std::vector<float>> seq = prompt_seq;
    for (size_t b = 0; b < transformer_blocks.size(); ++b) {
        seq = transformer_blocks[b].forward_cached(seq, cache.layers[b], 0, use_bitlinear);
    }

    // 2. Decode Phase: generate new tokens 1-by-1 with KV Cache
    for (int step = 0; step < max_length; ++step) {
        if (static_cast<int>(tokens.size()) >= config_.max_seq_len) break;

        std::vector<float> last_hidden = seq.back();
        std::vector<float> logits = use_bitlinear ? get_logits_bitlinear(last_hidden) : get_logits(last_hidden);
        int vocab_sz = static_cast<int>(logits.size());

        if (bpe == nullptr) {
            for (int i = 0; i < vocab_sz; ++i) {
                if (i < 32 || i > 126) {
                    logits[i] = -1e9f;
                }
            }
        }

        int last_tok = -1;
        int streak = 0;
        for (int t : tokens) {
            if (t == last_tok) streak++;
            else { last_tok = t; streak = 1; }
        }
        if (streak >= 3 && last_tok >= 0 && last_tok < vocab_sz) {
            logits[last_tok] = -1e9f;
        }

        std::unordered_set<int> distinct_tokens;
        size_t start_r = tokens.size() > 16 ? tokens.size() - 16 : 0;
        for (size_t r = start_r; r < tokens.size(); ++r) {
            distinct_tokens.insert(tokens[r]);
        }
        for (int tok : distinct_tokens) {
            if (tok >= 0 && tok < vocab_sz) {
                if (tok != 32 && tok != ' ' && tok != 1) {
                    logits[tok] -= 3.0f;
                }
            }
        }

        int next_token = 0;
        if (temperature <= 0.01f) {
            next_token = static_cast<int>(std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        } else {
            next_token = sample_logits(logits, temperature, top_k, top_p);
        }

        tokens.push_back(next_token);

        if (next_token == 0 || next_token == 1 || next_token == 2 || next_token == '\n') break;

        // Process only the single newest token for the next step
        std::vector<std::vector<float>> next_input(1);
        if (next_token >= 0 && next_token < static_cast<int>(token_embeddings.size())) {
            next_input[0] = token_embeddings[next_token];
        } else {
            next_input[0].assign(config_.hidden_dim, 0.0f);
        }

        int current_pos = static_cast<int>(tokens.size()) - 1;
        seq = next_input;
        for (size_t b = 0; b < transformer_blocks.size(); ++b) {
            seq = transformer_blocks[b].forward_cached(seq, cache.layers[b], current_pos, use_bitlinear);
        }
    }

    if (bpe != nullptr) {
        return bpe->decode(tokens);
    } else {
        return detokenize_tokens(tokens);
    }
}

float LanguageModel::compute_loss(const std::vector<std::vector<float>>& seq, 
                                   const std::vector<int>& target_tokens) {
    float total_loss = 0.0f;
    int count = 0;
    
    for (size_t t = 1; t < target_tokens.size() && t - 1 < seq.size(); ++t) {
        std::vector<float> logits = get_logits(seq[t - 1]);
        std::vector<float> probs = softmax(logits);
        
        int target_idx = target_tokens[t];
        if (target_idx >= 0 && target_idx < static_cast<int>(probs.size())) {
            total_loss -= std::log(probs[target_idx] + 1e-9f);
            count++;
        }
    }
    
    return count > 0 ? total_loss / count : 0.0f;
}

float LanguageModel::compute_loss(const std::vector<std::vector<float>>& input_seq, 
                                   const std::vector<std::vector<float>>& target_seq) {
    std::vector<int> dummy_targets(target_seq.size(), 32);
    for (size_t i = 0; i < target_seq.size(); ++i) {
        dummy_targets[i] = target_seq[i].empty() ? 32 : static_cast<int>(target_seq[i][0]);
    }
    return compute_loss(input_seq, dummy_targets);
}

std::vector<std::vector<float>> LanguageModel::encode_text(const std::string& text) {
    std::vector<int> tokens = tokenize_text(text, static_cast<int>(token_embeddings.size()));
    std::vector<std::vector<float>> sequences(tokens.size());
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        int token_idx = tokens[i];
        if (token_idx >= 0 && token_idx < static_cast<int>(token_embeddings.size())) {
            sequences[i] = token_embeddings[token_idx];
        } else {
            sequences[i].assign(config_.hidden_dim, 0.0f);
        }
    }
    
    return sequences;
}

std::string LanguageModel::decode_sequence(const std::vector<std::vector<float>>& sequences) {
    std::vector<int> decoded_tokens;
    
    for (const auto& seq : sequences) {
        int   best_idx = 0;
        float max_val  = -1e9f;
        
        for (int i = 0; i < static_cast<int>(token_embeddings.size()); ++i) {
            float dot_product = 0.0f;
            for (int j = 0; j < config_.hidden_dim; ++j) {
                dot_product += seq[j] * token_embeddings[i][j];
            }
            if (dot_product > max_val) {
                max_val  = dot_product;
                best_idx = i;
            }
        }
        decoded_tokens.push_back(best_idx);
    }
    
    return detokenize_tokens(decoded_tokens);
}

bool LanguageModel::save_model(const std::string& filepath) const {
    std::ofstream file(filepath, std::ios::out);
    if (!file.is_open()) return false;

    file << config_.hidden_dim << " " << config_.num_heads << " "
         << config_.num_layers << " " << config_.max_seq_len << "\n";

    // 1. Embeddings
    file << token_embeddings.size() << " " << config_.hidden_dim << "\n";
    for (const auto& vec : token_embeddings) {
        for (float v : vec) file << v << " ";
        file << "\n";
    }

    // 2. Vocab Projection (LM Head)
    file << vocab_projection.size() << " " << config_.hidden_dim << "\n";
    for (const auto& vec : vocab_projection) {
        for (float v : vec) file << v << " ";
        file << "\n";
    }

    // 3. Transformer Blocks
    file << transformer_blocks.size() << "\n";
    for (const auto& block : transformer_blocks) {
        file << block.attention.query_weights.size() << "\n";
        for (size_t h = 0; h < block.attention.query_weights.size(); ++h) {
            file << block.attention.query_weights[h].size() << " " << block.attention.query_weights[h][0].size() << "\n";
            for (const auto& row : block.attention.query_weights[h]) {
                for (float v : row) file << v << " ";
                file << "\n";
            }
            file << block.attention.key_weights[h].size() << " " << block.attention.key_weights[h][0].size() << "\n";
            for (const auto& row : block.attention.key_weights[h]) {
                for (float v : row) file << v << " ";
                file << "\n";
            }
            file << block.attention.value_weights[h].size() << " " << block.attention.value_weights[h][0].size() << "\n";
            for (const auto& row : block.attention.value_weights[h]) {
                for (float v : row) file << v << " ";
                file << "\n";
            }
        }

        file << block.ffn.weight1.size() << " " << block.ffn.weight1[0].size() << "\n";
        for (const auto& row : block.ffn.weight1) {
            for (float v : row) file << v << " ";
            file << "\n";
        }
        file << block.ffn.weight2.size() << " " << block.ffn.weight2[0].size() << "\n";
        for (const auto& row : block.ffn.weight2) {
            for (float v : row) file << v << " ";
            file << "\n";
        }
    }

    return true;
}

bool LanguageModel::load_model(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::in);
    if (!file.is_open()) return false;

    file >> config_.hidden_dim >> config_.num_heads >> config_.num_layers >> config_.max_seq_len;

    // 1. Embeddings
    size_t vocab_sz, emb_dim;
    if (!(file >> vocab_sz >> emb_dim)) return false;
    token_embeddings.resize(vocab_sz, std::vector<float>(emb_dim));
    for (size_t i = 0; i < vocab_sz; ++i) {
        for (size_t j = 0; j < emb_dim; ++j) {
            file >> token_embeddings[i][j];
        }
    }

    // 2. Vocab Projection
    size_t proj_sz, proj_dim;
    if (!(file >> proj_sz >> proj_dim)) return false;
    vocab_projection.resize(proj_sz, std::vector<float>(proj_dim));
    for (size_t i = 0; i < proj_sz; ++i) {
        for (size_t j = 0; j < proj_dim; ++j) {
            file >> vocab_projection[i][j];
        }
    }

    // 3. Transformer Blocks
    size_t num_blocks;
    if (!(file >> num_blocks)) return false;
    transformer_blocks.resize(num_blocks);

    for (size_t b = 0; b < num_blocks; ++b) {
        size_t num_heads;
        file >> num_heads;
        
        transformer_blocks[b].input_dim = config_.hidden_dim;
        transformer_blocks[b].attention.input_dim = config_.hidden_dim;
        transformer_blocks[b].attention.query_weights.resize(num_heads);
        transformer_blocks[b].attention.key_weights.resize(num_heads);
        transformer_blocks[b].attention.value_weights.resize(num_heads);

        for (size_t h = 0; h < num_heads; ++h) {
            size_t r, c;
            file >> r >> c;
            transformer_blocks[b].attention.query_weights[h].resize(r, std::vector<float>(c));
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < c; ++j)
                    file >> transformer_blocks[b].attention.query_weights[h][i][j];

            file >> r >> c;
            transformer_blocks[b].attention.key_weights[h].resize(r, std::vector<float>(c));
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < c; ++j)
                    file >> transformer_blocks[b].attention.key_weights[h][i][j];

            file >> r >> c;
            transformer_blocks[b].attention.value_weights[h].resize(r, std::vector<float>(c));
            for (size_t i = 0; i < r; ++i)
                for (size_t j = 0; j < c; ++j)
                    file >> transformer_blocks[b].attention.value_weights[h][i][j];
        }

        size_t r1, c1, r2, c2;
        file >> r1 >> c1;
        transformer_blocks[b].ffn.input_dim = c1;
        transformer_blocks[b].ffn.hidden_dim = r1;
        transformer_blocks[b].ffn.weight1.resize(r1, std::vector<float>(c1));
        for (size_t i = 0; i < r1; ++i)
            for (size_t j = 0; j < c1; ++j)
                file >> transformer_blocks[b].ffn.weight1[i][j];

        file >> r2 >> c2;
        transformer_blocks[b].ffn.weight2.resize(r2, std::vector<float>(c2));
        for (size_t i = 0; i < r2; ++i)
            for (size_t j = 0; j < c2; ++j)
                file >> transformer_blocks[b].ffn.weight2[i][j];
    }

    return true;
}

} // namespace matmul_free
