/**
 * MatMul-Free LLM - High Performance C++ CLI & Inference Engine
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
#include <chrono>
#include <string>
#include <vector>

using namespace matmul_free;

void print_help(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [COMMAND] [OPTIONS]\n\n";
    std::cout << "Commands:\n";
    std::cout << "  chat                              Start interactive REPL chat session (default)\n";
    std::cout << "  generate --prompt \"<text>\"        Generate text completion for a prompt\n";
    std::cout << "  train [--corpus FILE] [--epochs N] Train model on text corpus using AdamW\n";
    std::cout << "  benchmark                         Run FP32 GEMM vs BitLinear vs Popcount SIMD benchmarks\n";
    std::cout << "  demo                              Run sequential 11-stage feature demonstration suite\n";
    std::cout << "  help                              Show this help menu\n\n";
    std::cout << "Options:\n";
    std::cout << "  --prompt \"text\"                 Input text prompt for generation\n";
    std::cout << "  --corpus  filepath               Corpus text file for training (default: corpus.txt)\n";
    std::cout << "  --epochs  N                      Number of training epochs (default: 80)\n";
    std::cout << "  --max-len N                      Maximum token length to generate (default: 35)\n";
    std::cout << "  --temp    T                      Sampling temperature (default: 0.7)\n\n";
}

std::vector<std::string> load_corpus(const std::string& filepath) {
    std::vector<std::string> corpus;
    std::ifstream file(filepath);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) corpus.push_back(line);
        }
        file.close();
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
    return corpus;
}

void ensure_model_loaded(LanguageModel& model, BPETokenizer& bpe, int vocab_target = 1024, const std::string& checkpoint_path = "model_checkpoint.bin") {
    auto corpus = load_corpus("corpus.txt");
    bpe.build_vocab_from_corpus(corpus, vocab_target);
    model.resize_vocab(bpe.vocab_size());

    if (!model.load_model(checkpoint_path)) {
        std::cout << ">> Model checkpoint '" << checkpoint_path << "' not found. Training initial model (" << model.config_.num_layers << " layers, " << model.config_.hidden_dim << " dim)...\n";
        model.train(corpus, 80, 0.025f, false, &bpe);
        model.save_model(checkpoint_path);
        std::cout << ">> Initial model trained and saved to '" << checkpoint_path << "'.\n";
    }
}

void run_interactive_chat(LanguageModel& model, const BPETokenizer& bpe, bool use_bitlinear = false) {
    std::cout << "=======================================================\n";
    std::cout << "   MatMul-Free LLM - Interactive REPL Chat Engine      \n";
    std::cout << "=======================================================\n";
    std::cout << " [Model Size: ~20M Params | Hidden Dim: " << model.config_.hidden_dim << " | Layers: " << model.config_.num_layers << " | Heads: " << model.config_.num_heads << "]\n";
    std::cout << " [KV-Cache: ENABLED | Mode: " << (use_bitlinear ? "100% BitLinear 1.58-bit" : "Standard Fast Inference") << "]\n";
    std::cout << " Type your prompt below. Type 'exit', 'quit', or 'q' to stop.\n\n";

    std::string input;
    while (true) {
        std::cout << "\nUser > ";
        if (!std::getline(std::cin, input) || input == "exit" || input == "quit" || input == "q") {
            std::cout << "Exiting chat session. Goodbye!\n";
            break;
        }
        if (input.empty()) continue;

        std::cout << "Assistant > " << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        std::string response = model.generate_fast(input, 35, 0.7f, 3, 0.9f, use_bitlinear, &bpe);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << response << "\n";
        std::cout << "  [Generated in " << std::fixed << std::setprecision(2) << ms << " ms | O(1) KV-Cached]\n";
    }
}

void run_demo_suite(LanguageModel& model, BPETokenizer& bpe) {
    std::cout << "=== MatMul-Free LLM Sequential Demo Suite ===\n\n";

    // Demo 1
    std::cout << "--- Demo 1: Text Processing ---\n";
    std::string test_text = "Hello, this is a simple demonstration of the matmul-free LLM.";
    std::cout << "Original text: \"" << test_text << "\"\n";
    auto tokens = tokenize_text(test_text, model.token_embeddings.size());
    std::cout << "Tokenized (first 5): ";
    for (int i = 0; i < std::min(5, static_cast<int>(tokens.size())); ++i) std::cout << tokens[i] << " ";
    std::cout << "\n\n";

    // Demo 8
    std::cout << "--- Demo 8: BitLinear 1.58-bit Ternary Quantization (QAT with STE) ---\n";
    if (!model.transformer_blocks.empty()) {
        const auto& ffn = model.transformer_blocks[0].ffn;
        float scale = 1.0f;
        auto ternary_w = quantize_weights_ternary(ffn.weight1, scale);
        std::cout << "Quantized FFN Weight1 to ternary {-1, 0, +1} matrix! (Scale: " << std::fixed << std::setprecision(4) << scale << ")\n";
    }

    // Demo 9
    std::cout << "\n--- Demo 9: Sampling Strategies & BPE Tokenization ---\n";
    std::cout << "  Greedy (Prompt: 'once upon a '): \"" << model.generate("once upon a ", 25, 0.0f, 0, 1.0f, false, &bpe) << "\"\n";
    std::cout << "  KV-Cached Fast Generation:      \"" << model.generate_fast("once upon a ", 25, 0.0f, 0, 1.0f, false, &bpe) << "\"\n";
    std::cout << "  100% BitLinear + KV-Cache Fast Generation: \"" 
              << model.generate_fast("the smart ", 25, 0.5f, 3, 1.0f, true, &bpe) << "\"\n";

    // Demo 11
    std::cout << "\n--- Demo 11: Performance Micro-Benchmark ---\n";
    benchmark_matmul_vs_bitlinear(512, 512, 100);

    std::cout << "\n=== Demo Complete ===\n";
}

int main(int argc, char** argv) {
    ModelConfig config;
    config.hidden_dim = 64;           // Sane default for 14KB demo corpus
    config.num_heads = 4;             // 4 attention heads (head_dim = 16)
    config.num_layers = 2;            // 2 transformer layers
    config.max_seq_len = 256;         // Max sequence length

    std::string mode = "chat";
    std::string prompt = "once upon a time";
    std::string corpus_file = "corpus.txt";
    int epochs = 60;
    int max_len = 35;
    int vocab_size = 256;
    int patience = 8;
    float temp = 0.7f;
    float lr = -1.0f;
    bool use_bitlinear = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "chat" || arg == "train" || arg == "generate" || arg == "benchmark" || arg == "demo" || arg == "verify" || arg == "help") {
            mode = arg;
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--corpus" && i + 1 < argc) {
            corpus_file = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--patience" && i + 1 < argc) {
            patience = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            lr = std::stof(argv[++i]);
        } else if (arg == "--hidden-dim" && i + 1 < argc) {
            config.hidden_dim = std::stoi(argv[++i]);
        } else if (arg == "--layers" && i + 1 < argc) {
            config.num_layers = std::stoi(argv[++i]);
        } else if (arg == "--heads" && i + 1 < argc) {
            config.num_heads = std::stoi(argv[++i]);
        } else if (arg == "--vocab-size" && i + 1 < argc) {
            vocab_size = std::stoi(argv[++i]);
        } else if (arg == "--max-len" && i + 1 < argc) {
            max_len = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--bitlinear" || arg == "-b") {
            use_bitlinear = true;
        } else if (arg == "--help" || arg == "-h") {
            mode = "help";
        }
    }

    if (lr < 0.0f) {
        lr = 0.002f;
    }

    LanguageModel model(config);
    BPETokenizer bpe;

    if (mode == "help") {
        print_help(argv[0]);
        return 0;
    }

    if (mode == "verify") {
        std::cout << "=== Numerical Parity Verification ===\n";
        AttentionLayer attn;
        attn.input_dim = 128;
        int head_dim = 32;
        for (int h = 0; h < 4; ++h) {
            std::vector<std::vector<float>> q_w(head_dim, std::vector<float>(128, 0.01f * (h + 1)));
            std::vector<std::vector<float>> k_w(head_dim, std::vector<float>(128, 0.02f * (h + 1)));
            std::vector<std::vector<float>> v_w(head_dim, std::vector<float>(128, 0.03f * (h + 1)));
            attn.query_weights.push_back(q_w);
            attn.key_weights.push_back(k_w);
            attn.value_weights.push_back(v_w);
        }
        std::vector<std::vector<float>> inputs(16, std::vector<float>(128, 0.0f));
        for (size_t i = 0; i < inputs.size(); ++i) {
            for (size_t j = 0; j < inputs[i].size(); ++j) {
                inputs[i][j] = static_cast<float>(i + j) / 100.0f;
            }
        }
        auto out_cpu = attn.forward(inputs);
        attn.upload_to_gpu();
        auto out_gpu = attn.forward(inputs);

        float max_diff = 0.0f;
        for (size_t i = 0; i < out_cpu.size(); ++i) {
            for (size_t j = 0; j < out_cpu[i].size(); ++j) {
                float diff = std::abs(out_cpu[i][j] - out_gpu[i][j]);
                if (diff > max_diff) max_diff = diff;
            }
        }
        std::cout << "Attention Layer Forward Max Abs Diff (CPU vs GPU batched): " << max_diff << "\n";
        if (max_diff < 1e-4f) {
            std::cout << ">> PASS: Batched GPU Attention matches CPU within tolerance (< 1e-4)!\n";
        } else {
            std::cout << ">> FAIL: Discrepancy detected (diff: " << max_diff << " >= 1e-4)\n";
        }
        return 0;
    }

    if (mode == "benchmark") {
        std::cout << "=======================================================\n";
        std::cout << "   MatMul-Free LLM - Performance Micro-Benchmarks      \n";
        std::cout << "=======================================================\n";
        benchmark_matmul_vs_bitlinear(512, 512, 100);
        return 0;
    }

    if (mode == "train") {
        auto corpus = load_corpus(corpus_file);
        bpe.build_vocab_from_corpus(corpus, vocab_size);
        model.resize_vocab(bpe.vocab_size());

        size_t total_tokens = 0;
        for (const auto& text : corpus) {
            total_tokens += bpe.encode(text, true).size();
        }

        size_t h = config.hidden_dim;
        size_t v = bpe.vocab_size();
        size_t l = config.num_layers;
        size_t est_params = 2 * v * h + l * (11 * h * h);
        float params_per_token = total_tokens > 0 ? static_cast<float>(est_params) / static_cast<float>(total_tokens) : 0.0f;

        std::cout << ">> Starting AdamW training on corpus '" << corpus_file << "' (" << epochs << " epochs, LR: " << lr << ", " << vocab_size << " vocab target)...\n";
        std::cout << "   [Model Size: ~" << (est_params / 1000) << "K Params | Hidden Dim: " << config.hidden_dim
                  << " | Layers: " << config.num_layers << " | Heads: " << config.num_heads << "]\n";

        if (params_per_token > 200.0f) {
            std::cout << "\n[WARNING] Model has ~" << est_params << " params but the corpus tokenizes to only ~" << total_tokens
                      << " tokens (~" << static_cast<int>(params_per_token) << " params/token).\n"
                      << "          This is far into memorization territory — expect the model to memorize the corpus rather than generalize.\n"
                      << "          Consider a smaller --hidden-dim/--layers or a larger --corpus.\n\n";
        }

        model.train(corpus, epochs, lr, false, &bpe, patience);
        model.save_model("model_checkpoint.bin");
        std::cout << ">> Training complete! Checkpoint saved to 'model_checkpoint.bin'.\n";
        return 0;
    }

    ensure_model_loaded(model, bpe, vocab_size);

    if (mode == "generate") {
        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Generated: \"" << model.generate_fast(prompt, max_len, temp, 3, 0.9f, use_bitlinear, &bpe) << "\"\n";
        return 0;
    }

    if (mode == "demo") {
        run_demo_suite(model, bpe);
        return 0;
    }

    // Default mode: Interactive REPL Chat
    run_interactive_chat(model, bpe, use_bitlinear);
    return 0;
}
