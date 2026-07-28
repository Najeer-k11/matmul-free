#include "byte_tokenizer.h"

namespace matmul_free {

std::vector<int> tokenize_text(const std::string& text, int vocab_size) {
    (void)vocab_size;
    std::vector<int> tokens;
    
    for (char c : text) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc >= 32 && uc < 128) {
            tokens.push_back(uc);
        } else {
            tokens.push_back(0); // Unknown token
        }
    }
    
    return tokens;
}

std::string detokenize_tokens(const std::vector<int>& tokens) {
    std::string result;
    
    for (int token : tokens) {
        if (token == 0) continue;
        
        unsigned char uc = static_cast<unsigned char>(token);
        if (uc >= 32 && uc < 128) {
            result += static_cast<char>(uc);
        } else {
            result += '?';
        }
    }
    
    return result;
}

} // namespace matmul_free
