#ifndef MATMUL_FREE_KV_CACHE_H
#define MATMUL_FREE_KV_CACHE_H

#include <vector>

namespace matmul_free {

/**
 * Key-Value cache stored per Transformer layer for fast autoregressive decoding.
 */
struct LayerKVCache {
    // k[head_idx][token_pos][head_dim]
    std::vector<std::vector<std::vector<float>>> k;
    // v[head_idx][token_pos][head_dim]
    std::vector<std::vector<std::vector<float>>> v;

    void clear() {
        k.clear();
        v.clear();
    }
};

/**
 * Model-wide Key-Value cache across all Transformer blocks.
 */
struct ModelKVCache {
    std::vector<LayerKVCache> layers;

    explicit ModelKVCache(size_t num_layers = 0) {
        layers.resize(num_layers);
    }

    void clear() {
        for (auto& layer : layers) {
            layer.clear();
        }
    }
};

} // namespace matmul_free

#endif // MATMUL_FREE_KV_CACHE_H
