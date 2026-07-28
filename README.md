# matmul-free

A small transformer language model written from scratch in C++ that avoids traditional
BLAS-style matrix multiplication — attention, the feed-forward network, and the output
projection are all computed with explicit dot-product loops, and the feed-forward weights
can be quantized to ternary `{-1, 0, +1}` values (BitNet-style "1.58-bit" weights) so that
multiplication is replaced with conditional addition/subtraction.

This README documents what's actually in the repo today, how to build and run it, what each
part of the code does, and — honestly — what its current limitations are. It's a hobby/
research-style project, not a production inference engine.

---

## What this project actually is

- A **3-layer, 128-dim, 4-head transformer** with causal self-attention, rotary position
  embeddings (RoPE), a GELU feed-forward network, and Pre-LN RMSNorm before each sub-layer.
- A **from-scratch autodiff-free backward pass** — gradients for every layer (attention
  Q/K/V, RMSNorm, FFN, output projection, token embeddings) are derived and coded by hand,
  not computed by a framework.
- A **real byte-pair-encoding (BPE) tokenizer** built from `corpus.txt`: it starts from
  printable ASCII characters and iteratively merges the most frequent adjacent pair until it
  hits a target vocabulary size (400 by default).
- A **BitLinear ternary quantization path**: FFN weights can be quantized to `{-1, 0, +1}`
  (scaled by a single float `gamma` per matrix) and the forward pass computed with
  additions/subtractions instead of multiplications. There's a quantization-aware training
  (QAT) mode that trains the underlying float weights using a straight-through estimator so
  they tolerate being rounded to ternary values at inference time.
- **Sampling strategies** for generation: greedy, temperature scaling, top-k, and top-p
  (nucleus) sampling.
- **Checkpointing** (binary save/load of all weights) and a **micro-benchmark** comparing
  plain FP32 dot products against the ternary/packed/AVX2 paths.

### Current state (honest assessment)

With the bundled 208-line `corpus.txt` and default settings, training converges to a very
low loss (well under 1.0) within 30-80 epochs — at that scale, this means the model has
largely **memorized the training corpus** rather than learned to generalize to genuinely
new sentences. That's expected and fine for a demo of this size; don't read the "coherent"
generation samples as evidence of general language understanding — most of them are
paraphrases or direct recollections of lines in `corpus.txt`. Novel prompts that go beyond
the training data will still produce broken, ungrammatical output.

Training on this small a corpus is also numerically delicate: earlier versions of this
project went to NaN or collapsed into repeating-token loops partway through training before
Pre-LN RMSNorm, an analytical RMSNorm backward pass, and an explicit
loss-explosion/flatline detector (which reverts to the last good checkpoint) were added.
These safeguards are in place now, but if you significantly change the learning rate, model
size, or corpus, instability can still resurface — watch the per-epoch loss log.

---

## Repository layout

```text
matmul-free/
├── cpps.toml                    # Build config for the `cpps` runner (see below)
├── corpus.txt                   # Training corpus (208 short story lines)
├── TASKS.md                     # Project roadmap / phase checklist
├── LICENSE                      # MIT License
├── model_checkpoint.bin         # Written by Demo 10 the first time you run the program
└── src/
    ├── main.cpp                 # Entry point — runs 11 sequential demos
    ├── benchmark/
    │   ├── benchmark.h
    │   └── benchmark.cpp        # OpenMP & BitLinear micro-benchmarks
    ├── core/
    │   ├── math_ops.h
    │   └── math_ops.cpp         # Softmax, matmul, GELU, RMSNorm, RoPE
    ├── model/
    │   ├── config.h             # ModelConfig struct
    │   ├── attention.h
    │   ├── attention.cpp        # Multi-head attention layer
    │   ├── feed_forward.h
    │   ├── feed_forward.cpp     # Feed-forward network (FFN)
    │   ├── transformer_block.h
    │   ├── transformer_block.cpp# Transformer block with Pre-LN RMSNorm
    │   ├── language_model.h
    │   └── language_model.cpp   # LanguageModel class & training/generation loops
    ├── quantization/
    │   ├── ternary.h
    │   └── ternary.cpp          # Ternary quantization & SIMD/AVX2 kernels
    ├── sampling/
    │   ├── sampling.h
    │   └── sampling.cpp         # Temperature, Top-K, Top-P sampling
    └── tokenizer/
        ├── byte_tokenizer.h
        ├── byte_tokenizer.cpp   # Byte-level ASCII tokenizer
        ├── bpe_tokenizer.h
        └── bpe_tokenizer.cpp    # Subword BPE tokenizer
```

---

## How to build and run

### Option A — using the `cpps` tool (recommended)

[`cpps`](https://github.com/Najeer-k11/cpps) is a small cross-platform CLI that wraps
compiler detection, project scaffolding, and build/run in one command (comparable to
`cargo run` for Rust or `npm run` for Node). This repo already has a `cpps.toml`, so once
`cpps` is installed you don't need to think about compiler flags at all.

1. **Install `cpps`** — pick whichever matches your platform:
   - Windows: download the `.msi` from the [cpps Releases page](https://github.com/Najeer-k11/cpps/releases)
     (adds it to `PATH` automatically), or run the PowerShell install script from the cpps repo.
   - From source (any OS with a Rust toolchain): `cargo install --path .` inside a clone of
     the cpps repo.
   - `cpps doctor --fix` afterward will detect and, if needed, install a C++ compiler,
     CMake/Ninja, and vcpkg for you.

2. **Run this project:**
   ```bash
   cd matmul-free
   cpps run
   ```
   `cpps` reads `cpps.toml`, compiles everything under `src/` with the flags below, and
   immediately runs the resulting binary.

`cpps.toml` in this repo currently specifies:
```toml
[project]
name    = "matmul-free-1"
version = "0.1.0"
std     = "c++17"

[compiler]
preferred = "auto"
flags     = ["-Wall", "-O3", "-fopenmp"]

[build]
src_dir = "src"
out_dir = "build"
entry   = "src/main.cpp"
```

---

### Option B — compiling directly with a C++ compiler

No `cpps` required — just make sure your compiler supports C++17 and OpenMP.

**GCC / Clang (Linux/macOS/MinGW):**
```bash
g++ -O3 -std=c++17 -fopenmp \
    src/main.cpp \
    src/core/*.cpp \
    src/tokenizer/*.cpp \
    src/quantization/*.cpp \
    src/sampling/*.cpp \
    src/model/*.cpp \
    src/benchmark/*.cpp \
    -I src \
    -o matmul_free_llm

./matmul_free_llm
```

**MSVC (Developer Command Prompt for VS):**
```cmd
cl /O2 /std:c++17 /openmp /I src src\main.cpp src\core\*.cpp src\tokenizer\*.cpp src\quantization\*.cpp src\sampling\*.cpp src\model\*.cpp src\benchmark\*.cpp /Fe:matmul_free_llm.exe
matmul_free_llm.exe
```

OpenMP is optional — the code compiles and runs fine without `-fopenmp`, just single-threaded.

---

## What happens when you run it

`main()` runs 11 demos back-to-back, all against one `LanguageModel` instance
(hidden_dim=128, 4 heads, 3 layers, max_seq_len=128):

| # | Demo | What it shows |
|---|------|----------------|
| 1 | Text Processing | Byte-level tokenization of a sample sentence |
| 2 | Transformer Block Forward Pass | Runs one transformer block on the tokenized input |
| 3 | Simple Text Generation | Generates 10 tokens from an **untrained** model (expect gibberish here — this is before any training happens) |
| 4 | Attention Mechanism | Standalone run of the first block's attention layer |
| 5 | Cross-Entropy Loss | Loss computed against a synthetic target sequence |
| 6 | Feed-Forward Network | Standalone FFN forward pass on a constant input vector |
| 7 | Model Training & Backpropagation | Loads `corpus.txt`, builds a 400-token BPE vocabulary, trains for 80 epochs with linear LR decay (0.025 → 0.0001), a 90/10 train/val split, per-epoch NaN/explosion detection, and best-checkpoint restoration |
| 8 | BitLinear QAT | Quantizes the FFN's weight matrix to ternary and shows one BitLinear forward pass, then runs 30 epochs of quantization-aware fine-tuning (STE) at a lower learning rate (0.002) |
| 9 | Sampling & BPE | Encodes/decodes a sentence through the BPE tokenizer, then generates from five different prompts using greedy / temperature / top-k / top-p / BitLinear sampling |
| 10 | Checkpointing | Saves all weights to `model_checkpoint.bin` and reloads them into a fresh model instance to confirm round-tripping works |
| 11 | Performance Benchmark | Times FP32 dot products vs. BitLinear vs. packed 2-bit vs. AVX2 on a 512×512 matrix, 100 iterations, 5-trial median |

A representative (real) run produces training loss falling from ~5.8 to under 0.15 over the
80-epoch main run, and generation samples like:
```text
Greedy: "once upon a time a smart fox lived in the green forest..."
BitLinear: "the smart fox served as the principal mentor guiding student project"
```
and a benchmark table roughly like:
```text
FP32 MatMul Latency:        ~9.9 ms
BitLinear Latency:          ~5.5 ms
Packed 2-Bit SIMD Latency:  ~4.1 ms
Explicit AVX2 SIMD Latency: ~4.2 ms
Memory Footprint Reduction: 16.00x smaller (1024 KB -> 64 KB, packed ternary storage)
```
Exact numbers vary by machine and by how training happened to converge on a given run.

---

## Key configuration knobs

All of these are set in `main.cpp` — there's no config file or CLI flags yet, so changing
behavior means editing and recompiling.

- **Model size** — `ModelConfig` in `main()`: `hidden_dim`, `num_heads`, `num_layers`,
  `max_seq_len`.
- **Training corpus** — put your own text in `corpus.txt` (one line per training example);
  if that file is missing, a hardcoded 15-sentence fallback corpus is used instead.
- **BPE vocabulary size** — `bpe.build_vocab_from_corpus(corpus, 400)` — the second argument
  is the target vocabulary size (base ASCII characters + learned merges). Smaller corpora
  should generally use a smaller target so each token gets enough training exposure.
- **Main training** — `model.train(corpus, epochs, initial_lr, use_qat, &bpe)`. Set
  `use_qat = true` to fine-tune with ternary-quantized forward passes (STE) instead of the
  regular float forward pass; this is what Demo 8 does at a lower LR after Demo 7's normal
  training.
- **Generation** — `model.generate(prompt, max_length, temperature, top_k, top_p,
  use_bitlinear, &bpe)`. Set `temperature = 0.0` for greedy decoding; `top_k = 0` disables
  top-k filtering; `top_p = 1.0` disables nucleus filtering; `use_bitlinear = true` runs
  inference through the ternary-quantized FFN weights instead of full float weights.

---

## Known limitations / things to be aware of

- **Tiny corpus → memorization, not generalization.** 208 lines is enough to overfit a
  128-dim, 3-layer model quickly. Treat generation quality as a measure of "did it memorize
  the corpus," not "can it write general English."
- **Val-loss-based checkpoint selection can pick very early checkpoints.** With such a small
  held-out validation split, val loss starts rising almost immediately once the model starts
  fitting the training set, so the "best" checkpoint by val loss may be much less trained
  (and less fluent) than a later one you'd prefer qualitatively. Check the per-epoch sample
  generations logged during training, not just the final restored checkpoint, if output
  quality matters more than validation-loss minimization.
- **Multi-threading Acceleration is OpenMP, not CUDA.** CPU multi-threading parallelism is
  enabled via `#pragma omp parallel for` when OpenMP is available.
- **No CLI/config file yet** — model size, corpus path, training hyperparameters, and
  generation parameters are all hardcoded in `main.cpp` and require a recompile to change.
- **Single executable, no library API** — everything runs through the `main()` demo
  sequence; there's no separate way to just call `generate()` on an already-trained
  checkpoint without also re-running training first (unless you write your own small
  driver that calls `load_model()` directly).

---

## Roadmap

See `TASKS.md` for the phase-by-phase checklist (tokenization/backprop/optimizer → BitLinear
+ RMSNorm → sampling/BPE → checkpointing/benchmarking). All listed phases are currently
checked off; future work would likely mean expanding the corpus, adding a proper train/
inference CLI, and hardening the numerical-stability safeguards further (e.g. extending
the explosion detector, tuning LR schedules per phase).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
