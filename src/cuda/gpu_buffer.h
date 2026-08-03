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
 * Singleton scratch buffer manager for persistent, reusable GPU device memory allocations.
 */
class ScratchBufferManager {
public:
    static ScratchBufferManager& instance() {
        static ScratchBufferManager mgr;
        return mgr;
    }

    float* get_float_in(size_t count) {
        if (count > cap_in_float_) {
            if (d_in_float_) cudaFree(d_in_float_);
            cap_in_float_ = count * 2;
            cudaMalloc(&d_in_float_, cap_in_float_ * sizeof(float));
        }
        return d_in_float_;
    }

    float* get_float_out(size_t count) {
        if (count > cap_out_float_) {
            if (d_out_float_) cudaFree(d_out_float_);
            cap_out_float_ = count * 2;
            cudaMalloc(&d_out_float_, cap_out_float_ * sizeof(float));
        }
        return d_out_float_;
    }

    float* get_float_weight(size_t count) {
        if (count > cap_w_float_) {
            if (d_w_float_) cudaFree(d_w_float_);
            cap_w_float_ = count * 2;
            cudaMalloc(&d_w_float_, cap_w_float_ * sizeof(float));
        }
        return d_w_float_;
    }

    int8_t* get_int8_weight(size_t count) {
        if (count > cap_w_int8_) {
            if (d_w_int8_) cudaFree(d_w_int8_);
            cap_w_int8_ = count * 2;
            cudaMalloc(&d_w_int8_, cap_w_int8_ * sizeof(int8_t));
        }
        return d_w_int8_;
    }

    uint8_t* get_uint8_weight(size_t count) {
        if (count > cap_w_uint8_) {
            if (d_w_uint8_) cudaFree(d_w_uint8_);
            cap_w_uint8_ = count * 2;
            cudaMalloc(&d_w_uint8_, cap_w_uint8_ * sizeof(uint8_t));
        }
        return d_w_uint8_;
    }

    ~ScratchBufferManager() {
        if (d_in_float_)  cudaFree(d_in_float_);
        if (d_out_float_) cudaFree(d_out_float_);
        if (d_w_float_)   cudaFree(d_w_float_);
        if (d_w_int8_)    cudaFree(d_w_int8_);
        if (d_w_uint8_)   cudaFree(d_w_uint8_);
    }

private:
    ScratchBufferManager() = default;

    float* d_in_float_ = nullptr;
    size_t cap_in_float_ = 0;

    float* d_out_float_ = nullptr;
    size_t cap_out_float_ = 0;

    float* d_w_float_ = nullptr;
    size_t cap_w_float_ = 0;

    int8_t* d_w_int8_ = nullptr;
    size_t cap_w_int8_ = 0;

    uint8_t* d_w_uint8_ = nullptr;
    size_t cap_w_uint8_ = 0;
};

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

    /**
     * Batched sequence GEMM: computes Out[T x rows] = In[T x cols] * W^T
     * Takes the full sequence matrix (T tokens x cols features) in one GPU launch.
     * Far more efficient than calling gemv_cpu() T times.
     */
    std::vector<std::vector<float>> gemm_sequence(const std::vector<std::vector<float>>& seq_input) const;

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
