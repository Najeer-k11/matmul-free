/**
 * MatMul-Free LLM - Simple C++ Implementation
 * 
 * This program demonstrates a basic language model that avoids traditional
 * matrix multiplication (O(n³)) by using:
 * - Element-wise operations where possible
 * - Dot product implementations instead of matmul
 * - Block-wise processing for large matrices
 * - Numerical stability techniques (log-sum-exp)
 */

#include "model/language_model.h"
#include "tokenizer/byte_tokenizer.h"
#include "tokenizer/bpe_tokenizer.h"
#include "quantization/ternary.h"
#include "sampling/sampling.h"
#include "benchmark/benchmark.h"
#include "core/math_ops.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace matmul_free;

// ============================================================================
// Helper Functions
// ============================================================================

void print_matrix(const std::vector<std::vector<float>>& matrix, const char* title = nullptr) {
    if (title != nullptr) std::cout << "\n" << title << ":\n";
    
    for (const auto& row : matrix) {
        for (float val : row) {
            std::cout << std::fixed << std::setprecision(4) << val << " ";
        }
        std::cout << "\n";
    }
}

// ============================================================================
// Main Program
// ============================================================================

int main() {
    std::cout << "=== MatMul-Free LLM Demo ===\n\n";
    
    // Create model configuration
    ModelConfig config;
    config.hidden_dim = 128;          // 128 hidden dim for higher capacity
    config.num_heads = 4;             // 4 attention heads (head_dim = 32)
    config.num_layers = 3;            // 3 transformer layers
    config.max_seq_len = 128;         // Max sequence length
    
    std::cout << "Model Configuration:\n";
    std::cout << "  Hidden Dimension: " << config.hidden_dim << "\n";
    std::cout << "  Number of Heads: " << config.num_heads << "\n";
    std::cout << "  Number of Layers: " << config.num_layers << "\n";
    std::cout << "  Max Sequence Length: " << config.max_seq_len << "\n\n";
    
    // Create language model instance
    LanguageModel model(config);
    
    std::cout << "Language Model created successfully!\n";
    std::cout << "Total transformer blocks: " << model.transformer_blocks.size() << "\n\n";
    
    // ========================================================================
    // Demo 1: Text Tokenization and Encoding
    // ========================================================================
    std::cout << "--- Demo 1: Text Processing ---\n";
    
    std::string test_text = "Hello, this is a simple demonstration of the matmul-free LLM.";
    std::cout << "\nOriginal text: \"" << test_text << "\"\n";
    
    // Tokenize and encode (all methods are inline now)
    auto tokens = tokenize_text(test_text, model.token_embeddings.size());
    std::vector<std::vector<float>> encoded = model.encode_text(test_text);
    
    std::cout << "Tokenized (first 5): ";
    for (int i = 0; i < std::min(5, static_cast<int>(tokens.size())); ++i) {
        std::cout << tokens[i] << " ";
    }
    if (static_cast<int>(tokens.size()) > 5) std::cout << "...";
    std::cout << "\n\n";
    
    // ========================================================================
    // Demo 2: Forward Pass Through Transformer Block
    // ========================================================================
    std::cout << "--- Demo 2: Transformer Block Forward Pass ---\n";
    
    if (!model.transformer_blocks.empty()) {
        auto input_seq = encoded;
        
        std::cout << "Input sequence shape: " 
                  << input_seq.size() << " tokens x " 
                  << input_seq[0].size() << " dimensions\n";
        
        // Process through first transformer block
        TransformerBlock& block = model.transformer_blocks[0];
        auto output = block.forward(input_seq);
        
        std::cout << "\nOutput shape: " 
                  << output.size() << " tokens x " 
                  << output[0].size() << " dimensions\n";
        
        // Print first few values for inspection
        std::cout << "First token - First 5 output values: ";
        for (int i = 0; i < std::min(5, static_cast<int>(output[0].size())); ++i) {
            std::cout << std::fixed << std::setprecision(4) << output[0][i] << " ";
        }
        std::cout << "\n";
    }
    
    // ========================================================================
    // Demo 3: Text Generation (Simple)
    // ========================================================================
    std::cout << "\n--- Demo 3: Simple Text Generation ---\n";
    
    std::string input_prompt = "The quick brown fox ";
    std::cout << "Input prompt: \"" << input_prompt << "\"\n";
    
    auto generated = model.generate(input_prompt, 10);
    std::cout << "Generated text: \"" << generated << "\"\n\n";
    
    // ========================================================================
    // Demo 4: Attention Weights Visualization
    // ========================================================================
    std::cout << "--- Demo 4: Attention Mechanism ---\n";
    
    if (!model.transformer_blocks.empty()) {
        auto input_seq = encoded;
        
        // Get attention weights from first block
        const auto& attn_layer = model.transformer_blocks[0].attention;
        
        std::cout << "Attention layer configuration:\n";
        std::cout << "  Input dimension: " << attn_layer.input_dim << "\n";
        std::cout << "  Hidden dimension: " << attn_layer.hidden_dim << "\n\n";
        
        // Process through attention to get weights
        auto output = attn_layer.forward(input_seq);
        
        std::cout << "Attention applied successfully!\n";
        std::cout << "Output shape: " 
                  << output.size() << " tokens x " 
                  << output[0].size() << "\n\n";
    }
    
    // ========================================================================
    // Demo 5: Loss Computation
    // ========================================================================
    std::cout << "--- Demo 5: Cross-Entropy Loss ---\n";
    
    if (!model.transformer_blocks.empty()) {
        auto input_seq = encoded;
        
        // Create a simple target sequence (repeat first token)
        std::vector<std::vector<float>> target_seq(input_seq.size());
        for (size_t i = 1; i < input_seq.size(); ++i) {
            int best_idx = 0;
            float max_val = -1e9f;
            
            for (int j = 0; j < static_cast<int>(input_seq[0].size()); ++j) {
                float dot_product = 0.0f;
                for (int k = 0; k < config.hidden_dim; ++k) {
                    dot_product += input_seq[i][k] * input_seq[0][k];
                }
                
                if (dot_product > max_val) {
                    max_val = dot_product;
                    best_idx = j;
                }
            }
            
            target_seq[i] = input_seq[best_idx];
        }
        
        float loss = model.compute_loss(input_seq, target_seq);
        
        std::cout << "Cross-entropy loss: " << std::fixed << std::setprecision(4) 
                  << loss << "\n\n";
    }
    
    // ========================================================================
    // Demo 6: FFN Layer Forward Pass
    // ============================================================================
    std::cout << "--- Demo 6: Feed-Forward Network ---\n";
    
    if (!model.transformer_blocks.empty()) {
        const auto& ffn = model.transformer_blocks[0].ffn;
        
        std::vector<float> input_vec(ffn.input_dim, 0.5f);
        
        std::cout << "Input to FFN (dimension " << ffn.input_dim << "):\n";
        
        auto output = ffn.forward(input_vec);
        
        // Print first 5 values for inspection
        std::cout << "Output from FFN (first 5 values): ";
        for (int i = 0; i < std::min(5, static_cast<int>(output.size())); ++i) {
            std::cout << std::fixed << std::setprecision(4) << output[i] << " ";
        }
        std::cout << "\n";
    }
    
    // ========================================================================
    // Demo 7: Active Model Training & Loss Convergence
    // ========================================================================
    std::cout << "\n--- Demo 7: Model Training & Backpropagation ---\n";
    
    std::vector<std::string> corpus;
    std::ifstream corpus_file("corpus.txt");
    if (corpus_file.is_open()) {
        std::string line;
        while (std::getline(corpus_file, line)) {
            if (!line.empty()) {
                corpus.push_back(line);
            }
        }
        corpus_file.close();
    }
    
    if (corpus.empty()) {
        corpus = {
            "once upon a time a smart fox lived in the green forest",
            "the fox liked to run and play near the big tree",
            "the little dragon liked to read books every evening",
            "the dragon read a book about a brave smart fox in the forest",
            "a friendly robot helped children learn science and mathematics",
            "the robot said hello to the smart fox and the little dragon",
            "matmul free language models perform fast inference without matrix multiplication",
            "deep neural networks run efficiently using ternary quantization",
            "the smart fox found a good book near the river in the forest",
            "children loved reading stories about the dragon and the fox",
            "the robot and the fox read books together in the forest",
            "learning language models is fun for the smart robot and children",
            "the brave dragon protected the forest and all the animals",
            "every evening the fox and the dragon read new books",
            "matmul free LLM generates text fast without floating point matrix multiplication"
        };
    }
    
    BPETokenizer bpe;
    bpe.build_vocab_from_corpus(corpus, 400);
    model.resize_vocab(bpe.vocab_size());

    std::cout << "Training corpus size: " << corpus.size() << " sentences / story lines\n";
    std::cout << "BPE Vocabulary size: " << bpe.vocab_size() << " subwords / tokens\n";
    std::cout << "Starting active training over 80 epochs with linear LR decay (0.025f -> 0.0001f)...\n";
    model.train(corpus, 80, 0.025f, false, &bpe);
    std::cout << "Training complete!\n";
    
    // ========================================================================
    // Demo 8: BitLinear 1.58-bit Ternary Quantization & RMSNorm (QAT with STE)
    // ========================================================================
    std::cout << "\n--- Demo 8: BitLinear 1.58-bit Ternary Quantization (QAT with STE) ---\n";
    
    if (!model.transformer_blocks.empty()) {
        const auto& ffn = model.transformer_blocks[0].ffn;
        
        float scale = 1.0f;
        auto ternary_w = quantize_weights_ternary(ffn.weight1, scale);
        
        std::cout << "Quantized FFN Weight1 to ternary {-1, 0, +1} matrix!\n";
        std::cout << "  Scale factor (gamma): " << std::fixed << std::setprecision(4) << scale << "\n";
        std::cout << "  Ternary matrix sample (first 3x5 values):\n";
        for (int i = 0; i < std::min(3, static_cast<int>(ternary_w.size())); ++i) {
            std::cout << "    [ ";
            for (int j = 0; j < std::min(5, static_cast<int>(ternary_w[i].size())); ++j) {
                int val = static_cast<int>(ternary_w[i][j]);
                std::cout << (val >= 0 ? " " : "") << val << " ";
            }
            std::cout << "]\n";
        }
        
        std::vector<float> input_vec(ffn.input_dim, 0.5f);
        auto bitlinear_output = ffn.forward_bitlinear(input_vec);
        
        std::cout << "\nOutput from BitLinear FFN (first 5 values): ";
        for (int i = 0; i < std::min(5, static_cast<int>(bitlinear_output.size())); ++i) {
            std::cout << std::fixed << std::setprecision(4) << bitlinear_output[i] << " ";
        }
        std::cout << "\nMultiplication-free BitLinear execution completed successfully!\n";

        std::cout << "\nRunning Quantization-Aware Training (QAT) fine-tuning over 30 epochs with STE & Best Checkpoint Restoration...\n";
        model.train(corpus, 30, 0.002f, true, &bpe);
        std::cout << "QAT Fine-tuning complete!\n";
    }
    
    // ========================================================================
    // Demo 9: Sampling Strategies & BPE Subword Tokenization
    // ========================================================================
    std::cout << "\n--- Demo 9: Sampling Strategies & BPE Tokenization ---\n";
    
    std::string bpe_input = "matmul free language model deep learning";
    auto bpe_tokens = bpe.encode(bpe_input);
    
    std::cout << "BPE Subword Tokenization:\n";
    std::cout << "  Input Text: \"" << bpe_input << "\"\n";
    std::cout << "  BPE Token IDs (" << bpe_tokens.size() << " tokens): ";
    for (int id : bpe_tokens) std::cout << id << " ";
    std::cout << "\n  Decoded Text: \"" << bpe.decode(bpe_tokens) << "\"\n\n";

    std::cout << "Autoregressive Sampling Generation Options:\n";
    std::cout << "  Greedy (Prompt: 'once upon a '): \"" << model.generate("once upon a ", 25, 0.0f, 0, 1.0f, false, &bpe) << "\"\n";
    std::cout << "  Temp=0.5 (Prompt: 'the smart '): \"" << model.generate("the smart ", 25, 0.5f, 3, 1.0f, false, &bpe) << "\"\n";
    std::cout << "  Top-K=3  (Prompt: 'the little'): \"" << model.generate("the little ", 25, 0.6f, 3, 1.0f, false, &bpe) << "\"\n";
    std::cout << "  Top-P=0.85(Prompt: 'a friendly'): \"" << model.generate("a friendly ", 25, 0.6f, 0, 0.85f, false, &bpe) << "\"\n";
    std::cout << "  BitLinear 1.58-bit Ternary Generation (Prompt: 'the smart '): \"" 
              << model.generate("the smart ", 25, 0.5f, 3, 1.0f, true, &bpe) << "\"\n";

    // ========================================================================
    // Demo 10: Model Checkpointing (Save & Load Verification)
    // ========================================================================
    std::cout << "\n--- Demo 10: Model Checkpointing (Save & Load) ---\n";
    
    std::string checkpoint_path = "model_checkpoint.bin";
    if (model.save_model(checkpoint_path)) {
        std::cout << "Successfully saved model checkpoint to '" << checkpoint_path << "'!\n";
        
        LanguageModel loaded_model(config);
        if (loaded_model.load_model(checkpoint_path)) {
            std::cout << "Successfully reloaded model checkpoint into fresh LanguageModel instance!\n";
            std::cout << "Reloaded token embeddings size: " << loaded_model.token_embeddings.size() << "\n";
        }
    }

    // ========================================================================
    // Demo 11: Performance Micro-Benchmark (FP32 MatMul vs. BitLinear)
    // ========================================================================
    std::cout << "\n--- Demo 11: Performance Micro-Benchmark ---\n";
    benchmark_matmul_vs_bitlinear(512, 512, 100);

    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "\n=== Demo Complete ===\n";
    std::cout << "The matmul-free LLM successfully demonstrated:\n";
    std::cout << "1. Text tokenization and encoding\n";
    std::cout << "2. Transformer block forward pass (without matrix multiplication)\n";
    std::cout << "3. Simple text generation\n";
    std::cout << "4. Attention mechanism\n";
    std::cout << "5. Cross-entropy loss computation\n";
    std::cout << "6. Feed-forward network operations\n";
    std::cout << "7. Model backpropagation and active training loop\n";
    std::cout << "8. BitLinear 1.58-bit ternary quantization & RMSNorm inference\n";
    std::cout << "9. Autoregressive Sampling (Temp/Top-K/Top-P) & BPE Subword Tokenization\n";
    std::cout << "10. Model Checkpointing (Save & Load serialization)\n";
    std::cout << "11. Performance Micro-Benchmarking (GEMM vs. Multiplication-Free BitLinear)\n\n";
    
    return 0;
}
