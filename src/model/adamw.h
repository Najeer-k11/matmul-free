#ifndef MATMUL_FREE_ADAMW_H
#define MATMUL_FREE_ADAMW_H

#include <vector>
#include <cmath>
#include <algorithm>

namespace matmul_free {

/**
 * AdamW Optimizer state for a 2D weight tensor with decoupled weight decay.
 */
struct AdamWMatrix {
    std::vector<std::vector<float>> m; // First moment vector (mean)
    std::vector<std::vector<float>> v; // Second moment vector (uncentered variance)

    void init_if_needed(size_t rows, size_t cols) {
        if (m.size() != rows || (rows > 0 && m[0].size() != cols)) {
            m.assign(rows, std::vector<float>(cols, 0.0f));
            v.assign(rows, std::vector<float>(cols, 0.0f));
        }
    }

    void update(std::vector<std::vector<float>>& param,
                const std::vector<std::vector<float>>& grad,
                float lr,
                int timestep,
                float beta1 = 0.9f,
                float beta2 = 0.999f,
                float eps = 1e-8f,
                float weight_decay = 0.01f) {
        if (param.empty()) return;
        init_if_needed(param.size(), param[0].size());

        float bc1 = 1.0f - std::pow(beta1, timestep);
        float bc2 = 1.0f - std::pow(beta2, timestep);
        if (bc1 <= 0.0f) bc1 = 1e-9f;
        if (bc2 <= 0.0f) bc2 = 1e-9f;

        for (size_t r = 0; r < param.size(); ++r) {
            for (size_t c = 0; c < param[r].size() && c < grad[r].size(); ++c) {
                float g = grad[r][c];
                if (std::isnan(g) || std::isinf(g)) continue;
                g = std::max(-1.0f, std::min(1.0f, g));

                m[r][c] = beta1 * m[r][c] + (1.0f - beta1) * g;
                v[r][c] = beta2 * v[r][c] + (1.0f - beta2) * (g * g);

                float m_hat = m[r][c] / bc1;
                float v_hat = v[r][c] / bc2;

                float update_step = m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param[r][c];
                update_step = std::max(-0.1f, std::min(0.1f, update_step));

                param[r][c] -= lr * update_step;
            }
        }
    }
};

} // namespace matmul_free

#endif // MATMUL_FREE_ADAMW_H
