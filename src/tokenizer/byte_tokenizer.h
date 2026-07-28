#ifndef MATMUL_FREE_BYTE_TOKENIZER_H
#define MATMUL_FREE_BYTE_TOKENIZER_H

#include <vector>
#include <string>

namespace matmul_free {

/**
 * Byte-level tokenization (maps printable ASCII characters to token IDs).
 */
std::vector<int> tokenize_text(const std::string& text, int vocab_size = 256);

/**
 * Detokenize sequence of token IDs back into string.
 */
std::string detokenize_tokens(const std::vector<int>& tokens);

} // namespace matmul_free

#endif // MATMUL_FREE_BYTE_TOKENIZER_H
