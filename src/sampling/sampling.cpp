#include "sampling.h"
#include "../core/math_ops.h"
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <utility>

namespace matmul_free {

int sample_logits(const std::vector<float>& logits, float temperature, int top_k, float top_p) {
    if (logits.empty()) return 32;
    
    float temp = std::max(temperature, 1e-4f);
    std::vector<float> scaled_logits(logits.size(), -1e9f);
    for (size_t i = 0; i < logits.size(); ++i) {
        if (!std::isnan(logits[i]) && !std::isinf(logits[i])) {
            scaled_logits[i] = logits[i] / temp;
        }
    }
    
    std::vector<float> probs = softmax(scaled_logits);
    int n = static_cast<int>(probs.size());
    
    std::vector<std::pair<float, int>> indexed_probs(n);
    for (int i = 0; i < n; ++i) {
        indexed_probs[i] = {probs[i], i};
    }
    std::sort(indexed_probs.rbegin(), indexed_probs.rend());

    if (top_k > 0 && top_k < n) {
        for (int i = top_k; i < n; ++i) {
            indexed_probs[i].first = 0.0f;
        }
    }

    float cdf = 0.0f;
    for (int i = 0; i < n; ++i) {
        cdf += indexed_probs[i].first;
        if (cdf > top_p && i > 0) {
            for (int j = i + 1; j < n; ++j) {
                indexed_probs[j].first = 0.0f;
            }
            break;
        }
    }

    float total_p = 0.0f;
    for (int i = 0; i < n; ++i) {
        total_p += indexed_probs[i].first;
    }

    if (total_p <= 1e-9f) {
        return indexed_probs[0].second;
    }

    float r = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * total_p;
    float accum = 0.0f;
    for (int i = 0; i < n; ++i) {
        accum += indexed_probs[i].first;
        if (r <= accum) {
            return indexed_probs[i].second;
        }
    }

    return indexed_probs[0].second;
}

} // namespace matmul_free
