#ifndef MATMUL_FREE_SAMPLING_H
#define MATMUL_FREE_SAMPLING_H

#include <vector>

namespace matmul_free {

/**
 * Sample token from logits using Temperature, Top-K, and Top-P (Nucleus) sampling.
 */
int sample_logits(const std::vector<float>& logits, float temperature = 1.0f, int top_k = 0, float top_p = 1.0f);

} // namespace matmul_free

#endif // MATMUL_FREE_SAMPLING_H
