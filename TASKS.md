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
