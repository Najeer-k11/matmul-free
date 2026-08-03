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
    config.hidden_dim = 512;          // Scaled to 512 hidden dimension
    config.num_heads = 8;             // Scaled to 8 attention heads
    config.num_layers = 6;            // Scaled to 6 transformer layers
    config.max_seq_len = 256;         // Scaled to 256 max sequence length

    std::string mode = "chat";
    std::string prompt = "once upon a time";
    std::string corpus_file = "corpus.txt";
    int epochs = 80;
    int max_len = 35;
    int vocab_size = 1024;
    float temp = 0.7f;
    float lr = -1.0f;
    bool use_bitlinear = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "chat" || arg == "train" || arg == "generate" || arg == "benchmark" || arg == "demo" || arg == "help") {
            mode = arg;
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--corpus" && i + 1 < argc) {
            corpus_file = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
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
        lr = (config.hidden_dim > 256) ? 0.002f : 0.025f;
    }

    LanguageModel model(config);
    BPETokenizer bpe;

    if (mode == "help") {
        print_help(argv[0]);
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
        std::cout << ">> Starting AdamW training on corpus '" << corpus_file << "' (" << epochs << " epochs, LR: " << lr << ", " << vocab_size << " vocab target)...\n";
        auto corpus = load_corpus(corpus_file);
        bpe.build_vocab_from_corpus(corpus, vocab_size);
        model.resize_vocab(bpe.vocab_size());
        model.train(corpus, epochs, lr, false, &bpe);
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
