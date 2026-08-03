#ifndef MATMUL_FREE_GPU_BUFFER_H
#define MATMUL_FREE_GPU_BUFFER_H

#include <vector>
#include <stdexcept>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace matmul_free {

/**
 * Thread-local cuBLAS handle initializer.
 * Returns the process-wide cuBLAS handle, creating it on first call.
 */
cublasHandle_t get_cublas_handle();

/**
 * RAII GPU device matrix. Stores a flat row-major float array on GPU VRAM.
 * Use to_device() to upload from CPU, from_device() to download to CPU.
 * gemv() computes y = W * x using cuBLAS (all on GPU).
 */
struct GpuMatrix {
    float* d_ptr = nullptr;   // Device pointer (VRAM)
    int rows = 0;
    int cols = 0;

    GpuMatrix() = default;
    GpuMatrix(int rows, int cols);
    ~GpuMatrix();

    // Non-copyable, move-only
    GpuMatrix(const GpuMatrix&) = delete;
    GpuMatrix& operator=(const GpuMatrix&) = delete;
    GpuMatrix(GpuMatrix&& other) noexcept;
    GpuMatrix& operator=(GpuMatrix&& other) noexcept;

    /**
     * Upload a CPU matrix (rows x cols) to GPU VRAM. Allocates if needed.
     */
    void to_device(const std::vector<std::vector<float>>& cpu_matrix);

    /**
     * Download GPU VRAM back to a CPU matrix.
     */
    std::vector<std::vector<float>> from_device() const;

    /**
     * GPU matrix-vector multiply: out = W * x  (length: rows)
     * x must already be in GPU VRAM (d_x), result goes to d_out.
     * Both d_x and d_out are device pointers.
     */
    void gemv_device(const float* d_x, float* d_out) const;

    /**
     * Convenience: takes a CPU vector, uploads it, computes gemv, returns CPU vector.
     * Uses temporary GPU buffers — only use when GPU-resident buffers are unavailable.
     */
    std::vector<float> gemv_cpu(const std::vector<float>& x) const;

    bool is_allocated() const { return d_ptr != nullptr; }
};

/**
 * A GPU-resident float vector buffer for activations/intermediates.
 */
struct GpuVector {
    float* d_ptr = nullptr;
    int size = 0;

    GpuVector() = default;
    explicit GpuVector(int size);
    ~GpuVector();

    GpuVector(const GpuVector&) = delete;
    GpuVector& operator=(const GpuVector&) = delete;
    GpuVector(GpuVector&& other) noexcept;
    GpuVector& operator=(GpuVector&& other) noexcept;

    void upload(const std::vector<float>& cpu_vec);
    std::vector<float> download() const;
    bool is_allocated() const { return d_ptr != nullptr; }
};

} // namespace matmul_free

#else // CPU fallback stubs — compiles fine without CUDA

namespace matmul_free {

struct GpuMatrix {
    int rows = 0, cols = 0;
    void to_device(const std::vector<std::vector<float>>&) {}
    std::vector<std::vector<float>> from_device() const { return {}; }
    bool is_allocated() const { return false; }
};

struct GpuVector {
    int size = 0;
    void upload(const std::vector<float>&) {}
    std::vector<float> download() const { return {}; }
    bool is_allocated() const { return false; }
};

} // namespace matmul_free

#endif // USE_CUDA

#endif // MATMUL_FREE_GPU_BUFFER_H
