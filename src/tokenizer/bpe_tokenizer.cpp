#include "bpe_tokenizer.h"
#include <map>
#include <sstream>
#include <algorithm>

namespace matmul_free {

BPETokenizer::BPETokenizer() {
    vocab_ = {"<unk>", "<pad>", "<eos>"};
    for (int i = 32; i < 128; ++i) {
        vocab_.push_back(std::string(1, static_cast<char>(i)));
    }
    for (size_t i = 0; i < vocab_.size(); ++i) {
        vocab_map_[vocab_[i]] = static_cast<int>(i);
    }
}

void BPETokenizer::build_vocab_from_corpus(const std::vector<std::string>& corpus, int target_vocab_size) {
    vocab_ = {"<unk>", "<pad>", "<eos>"};
    vocab_map_.clear();
    merges_.clear();
    
    for (int i = 32; i < 128; ++i) {
        vocab_.push_back(std::string(1, static_cast<char>(i)));
    }
    for (size_t i = 0; i < vocab_.size(); ++i) {
        vocab_map_[vocab_[i]] = static_cast<int>(i);
    }

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

    while (static_cast<int>(vocab_.size()) < target_vocab_size) {
        std::map<std::pair<std::string, std::string>, int> pair_counts;
        
        for (const auto& seq : sequences) {
            if (seq.size() < 2) continue;
            for (size_t i = 0; i + 1 < seq.size(); ++i) {
                pair_counts[{seq[i], seq[i + 1]}]++;
            }
        }

        if (pair_counts.empty()) break;

        std::pair<std::string, std::string> best_pair;
        int max_freq = 0;
        for (const auto& kv : pair_counts) {
            if (kv.second > max_freq) {
                max_freq = kv.second;
                best_pair = kv.first;
            }
        }

        if (max_freq < 2) break;

        std::string new_token = best_pair.first + best_pair.second;
        merges_.push_back(best_pair);
        int new_id = static_cast<int>(vocab_.size());
        vocab_.push_back(new_token);
        vocab_map_[new_token] = new_id;

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

std::vector<int> BPETokenizer::encode(const std::string& text, bool add_eos) const {
    std::vector<std::string> symbols;
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 32 && uc < 128) {
            symbols.push_back(std::string(1, c));
        }
    }

    if (symbols.empty()) {
        if (add_eos) return {2};
        return {};
    }

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
            tokens.push_back(0);
        }
    }
    if (add_eos) {
        tokens.push_back(2); // Append <eos> token ID
    }
    return tokens;
}

std::string BPETokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    for (int token : tokens) {
        if (token >= 0 && token < static_cast<int>(vocab_.size())) {
            if (token == 0 || token == 1 || token == 2) continue; // Skip <unk>, <pad>, <eos>
            text += vocab_[token];
        }
    }
    return text;
}

} // namespace matmul_free
