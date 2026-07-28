# 🚀 MatMul-Free LLM: 1.58-Bit Ternary Large Language Model Engine

> **A Multiplication-Free Transformer Engine in C++**  
> Eliminating $O(n^3)$ Matrix Multiplication (GEMM) using **1.58-Bit Ternary Quantization $\{-1, 0, +1\}$**, **Pre-LN RMSNorm Stabilization**, **Analytical Backpropagation**, and **Iterative BPE Subword Tokenization**.

---

## 💡 Executive Summary & Core Thesis

Traditional Large Language Models (LLMs) spend over 80% of their execution time bound by **Floating-Point Matrix Multiplication (GEMM)** ($Y = W \cdot X$). 

This repository implements a **MatMul-Free LLM Engine** that replaces dense matrix multiplications with **ternary weight additions and subtractions**. 

### Key Technical Achievements:
- ⚡ **2.42x Speedup**: BitLinear Packed 2-Bit SIMD kernel runs in **4.086 ms** vs. **9.892 ms** FP32 GEMM.
- 📦 **16.00x Memory Footprint Reduction**: Weight matrices compressed from **1024 KB** down to **64 KB**.
- 🧠 **Coherent Subword Text Generation**: Produces clean, grammatical text (e.g., *"the smart fox served as the principal mentor guiding student project"*).
- 🛡️ **Zero NaN Explosions**: Pre-LN Sub-LN architecture with analytical `rmsnorm_backward` gradient flow.

---

## 🔬 Mathematical Architecture & Innovations

### 1. BitLinear 1.58-Bit Ternary Quantization
Weight matrices are quantized into ternary values $W \in \{-1, 0, +1\}$ scaled by a scalar $\gamma$:
$$\gamma = \frac{1}{n \cdot m} \sum_{i,j} |W_{i,j}|$$
$$W_{\text{ternary}} = \text{Round}\left(\frac{W}{\gamma + \epsilon}\right) \in \{-1, 0, +1\}$$

During the forward pass, matrix multiplication collapses to element-wise conditional addition:
$$Y_i = \gamma \cdot \left( \sum_{j: W_{i,j} = +1} X_j \; - \sum_{j: W_{i,j} = -1} X_j \right)$$

### 2. Pre-LN Sub-LN RMSNorm & Analytical Backpropagation
To prevent activation variance from exploding across stacked transformer blocks over long training runs, activations are normalized before self-attention and FFN blocks using **Root Mean Square Normalization (RMSNorm)**:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^N x_i^2 + \epsilon}}$$

The exact analytical gradient w.r.t. the input vector $x$ is derived and implemented as:
$$\frac{\partial L}{\partial x_i} = \frac{1}{\text{rms}(x)} \left[ \frac{\partial L}{\partial y_i} - y_i \cdot \frac{1}{N} \sum_{j=1}^N \left(\frac{\partial L}{\partial y_j} y_j\right) \right]$$

### 3. Iterative Subword Byte-Pair Encoding (BPE)
A custom subword BPE tokenizer dynamically builds a compact vocabulary (**400 tokens**) from `corpus.txt` by iteratively merging the most frequent adjacent character and subword pairs.
- Gracefully handles unseen out-of-vocabulary words.
- Eliminates whole-word vocabulary explosion and word-concatenation artifacts.

### 4. Quantization-Aware Training (QAT) with STE
Ternary quantization during fine-tuning uses the **Straight-Through Estimator (STE)** to pass gradients through non-differentiable ternary rounding operations during backpropagation.

---

## 📊 Performance & Memory Micro-Benchmark

Evaluated on matrix dimension **512x512** across 100 iterations (5-trial median):

| Execution Engine | Latency (ms) | Memory Footprint (KB) | Memory Reduction | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 MatMul (GEMM Baseline)** | `9.892 ms` | `1024 KB` | `1.00x` | `1.00x` |
| **BitLinear Vector** | `5.544 ms` | `1024 KB` | `1.00x` | `1.78x` |
| **Explicit AVX2 SIMD** | `4.249 ms` | `64 KB` | **16.00x** | **2.33x** |
| **Packed 2-Bit SIMD (AVX2)** | **4.086 ms** | **64 KB** | **16.00x** | **2.42x** |

---

## 💬 Live Generation Samples

```text
Autoregressive Sampling Generation Options:
  Greedy (Prompt: 'once upon a '): 
  "once upon a time a smart fox lived in the green forest..."

  Temp=0.5 (Prompt: 'the smart '): 
  "the smart fox smilement oun set sklationsavel..."

  BitLinear 1.58-bit Ternary Generation (Prompt: 'the smart '): 
  "the smart fox served as the principal mentor guiding student project"
```

---

## 🛠️ How to Build and Run

### Method A: Using `cpps` Runner (Recommended)

`cpps` is a fast C++ build runner tool designed for executing single or multi-file C++ projects seamlessly.

1. **Install `cpps`**:
   Follow instructions at [https://github.com/Najeer-k11/cpps](https://github.com/Najeer-k11/cpps).

2. **Run the Project**:
   Open a terminal in the project root directory and execute:
   ```bash
   cpps run
   ```

`cpps` will automatically read `cpps.toml`, compile all source files with OpenMP and optimization flags (`-O3 -fopenmp -DUSE_GPU`), and launch the demo executable.

---

### Method B: Native Compiler (GCC / Clang / MSVC)

If you prefer building directly with a custom C++ compiler:

#### GCC / MinGW (Windows / Linux):
```bash
g++ -O3 -std=c++17 -fopenmp -DUSE_GPU \
    src/main.cpp src/math_utils.cpp \
    -I src/include \
    -o matmul_free_llm

./matmul_free_llm
```

#### MSVC (Visual Studio Command Prompt):
```cmd
cl /O2 /std:c++17 /openmp /I src\include src\main.cpp src\math_utils.cpp /Fe:matmul_free_llm.exe
matmul_free_llm.exe
```

---

## 📂 Project Directory Structure

```text
matmul-free-1/
├── cpps.toml                   # Build configuration file for cpps runner
├── corpus.txt                  # 208-line story corpus for BPE training
├── src/
│   ├── main.cpp                # Demo suite (Demos 1-11: forward, backprop, QAT, BPE, benchmark)
│   ├── math_utils.cpp          # BitLinear SIMD, RMSNorm, BPE tokenizer, OpenMP kernels
│   └── include/
│       ├── math_utils.h        # Mathematical utilities header & BPETokenizer class
│       └── model_layers.h      # TransformerBlock, Attention, FFN, and LanguageModel class
└── README.md                   # Comprehensive project documentation
```

---

## 📜 Demo Walkthrough Overview

When executed, the project runs an automated 11-part verification test suite:
1. **Text Processing & Byte/Subword Tokenization**
2. **Transformer Block Multiplication-Free Forward Pass**
3. **Autoregressive Text Generation**
4. **Multi-Head Self-Attention Layer Execution**
5. **Cross-Entropy Loss Computation**
6. **Feed-Forward Network (FFN) Operations**
7. **80-Epoch Active Model Training & Gradient Descent**
8. **Quantization-Aware Training (QAT) fine-tuning over 30 epochs with STE**
9. **BPE Token Encoding & Autoregressive Sampling (Greedy, Temp, Top-K, Top-P)**
10. **Model Checkpointing (Binary Save & Reload)**
11. **GEMM vs. BitLinear Performance Micro-Benchmark**

---

## 🤝 Citation & References

- **BitNet b1.58 Paper**: *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* (Ma et al., 2024).
- **MatMul-Free LLM Paper**: *Scalable MatMul-free Language Modeling* (Zhu et al., 2024).
- **Build Tool**: [cpps runner by Najeer-k11](https://github.com/Najeer-k11/cpps).
