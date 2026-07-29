#ifndef MATMUL_FREE_BPE_TOKENIZER_H
#define MATMUL_FREE_BPE_TOKENIZER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <utility>

namespace matmul_free {

class BPETokenizer {
public:
    BPETokenizer();
    void build_vocab_from_corpus(const std::vector<std::string>& corpus, int target_vocab_size = 400);
    std::vector<int> encode(const std::string& text, bool add_eos = false) const;
    std::string decode(const std::vector<int>& tokens) const;
    int vocab_size() const { return static_cast<int>(vocab_.size()); }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int> vocab_map_;
    std::vector<std::pair<std::string, std::string>> merges_;
};

} // namespace matmul_free

#endif // MATMUL_FREE_BPE_TOKENIZER_H
