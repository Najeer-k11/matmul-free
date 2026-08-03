# MatMul-Free LLM - Project Roadmap & Task List

## Phase 3: Model Training & Optimization
- [x] **Complete Backpropagation Pipeline**
  - [x] Implement `AttentionLayer::backward_and_update` with gradient calculation and weight updates.
  - [x] Implement `TransformerBlock::backward_and_update` chaining self-attention and FFN backpropagation.
  - [x] Implement `LanguageModel::backward_and_update` over token sequence embeddings.
- [x] **Optimizer Implementation**
  - [x] Add SGD (Stochastic Gradient Descent) optimizer with learning rate scaling for weights.
- [x] **Training Loop Integration**
  - [x] Train the language model on a sample text corpus in `LanguageModel::train`.
  - [x] Verify loss convergence over multiple epochs.

---

## Phase 4: Ternary Quantization & True MatMul-Free (BitLinear)
- [x] **BitLinear / Ternary Weight Layer**
  - [x] Quantize weights to ternary values $\{-1, 0, +1\}$ (1.58-bit representation).
  - [x] Replace floating-point multiplication in dot-products with conditional addition/subtraction.
- [x] **Sub-Layer Normalization (Sub-LN)**
  - [x] Add RMSNorm / LayerNorm before activation functions to maintain activation variance.

---

## Phase 5: Sampling, Generation & Tokenization
- [x] **Autoregressive Sampling Strategies**
  - [x] Implement Temperature scaling for output logits.
  - [x] Add Top-$K$ and Top-$P$ (Nucleus) sampling to `LanguageModel::generate`.
- [x] **Advanced Tokenizer**
  - [x] Implement Byte-Pair Encoding (BPE) or load a standard subword vocabulary.

---

## Phase 6: Model Persistence & Benchmarking
- [x] **Weight Save / Load (Checkpointing)**
  - [x] Add binary/JSON model serialisation and deserialisation methods (`save_model`, `load_model`).
- [x] **Performance & SIMD Benchmarking**
  - [x] Add micro-benchmarks comparing traditional GEMM runtime vs. MatMul-Free dot products (`benchmark_matmul_vs_bitlinear`).
  - [x] Demonstrate 4.00x memory footprint reduction factor and branchless SIMD speedup over FP32 MatMul.

---

## Phase 7: KV-Caching for $O(1)$ Fast Inference
- [x] **Layer & Model Key-Value Caching**
  - [x] Add `LayerKVCache` and `ModelKVCache` data structures ([`kv_cache.h`](file:///home/venx/Documents/personal/matmul-free/src/model/kv_cache.h)).
  - [x] Implement `AttentionLayer::forward_cached` to cache Key and Value vectors across generation steps.
  - [x] Implement `TransformerBlock::forward_cached` supporting Pre-LN RMSNorm and BitLinear.
- [x] **Fast Autoregressive Decoding**
  - [x] Implement `LanguageModel::generate_fast` using prompt prefill and $O(1)$ single-token decoding steps.
  - [x] Add KV-Cache latency benchmark comparing standard $O(N^2)$ generation against $O(1)$ KV-cached decoding.

---

## Phase 8: 100% MatMul-Free (BitLinear Across All Layers)
- [x] **Attention Layer BitLinear Quantization**
  - [x] Quantize $W_Q, W_K, W_V$ projection matrices per head to $\{-1, 0, +1\}$ values.
  - [x] Implement `AttentionLayer::forward_bitlinear` and `AttentionLayer::forward_bitlinear_cached`.
- [x] **Vocabulary Output Projection (LM Head) BitLinear Quantization**
  - [x] Quantize `vocab_projection` matrix to ternary values $\{-1, 0, +1\}$ scaled by $\gamma_{vocab}$.
  - [x] Implement `LanguageModel::get_logits_bitlinear` for multiplication-free logit generation.
  - [x] Demonstrate 100% MatMul-Free inference across Attention, FFN, and LM Head.

---

## Phase 11: AdamW Optimizer & Generation Polish
- [x] **AdamW Optimizer Implementation**
  - [x] Add `AdamWMatrix` struct with $m_t$ and $v_t$ momentum tracking ([`adamw.h`]).
  - [x] Update `LanguageModel::train` to use AdamW for parameter updates instead of basic SGD.
- [x] **Generation Polish**
  - [x] Implement subword repetition penalties and clean logit probability bounds.

---

## Phase 9: Bit-Parallel Popcount SIMD Kernels
- [x] **Bitwise Popcount Dot Product**
  - [x] Implement `bitlinear_popcount_simd` using 64-bit masks and hardware population count (`__builtin_popcount`).

---

## Phase 10: Interactive CLI Chat REPL & Tooling
- [x] **Production CLI Application**
  - [x] Implement REPL mode (`chat`), training subcommand (`train`), single-prompt mode (`generate`), and benchmark (`benchmark`).

---

## Phase 12: OpenMP Parallelized Attention & Prefill
- [x] **Multi-Threaded CPU Scaling**
  - [x] Parallelize attention computation and BitLinear dot products with OpenMP `#pragma omp parallel for`.
