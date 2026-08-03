#include "gpu_buffer.h"

#if defined(USE_CUDA)

#include "../core/timing.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace matmul_free {

// ── cuBLAS handle ────────────────────────────────────────────────────────────

static cublasHandle_t g_cublas_handle = nullptr;

cublasHandle_t get_cublas_handle() {
    if (!g_cublas_handle) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS init failed: " << status << "\n";
            g_cublas_handle = nullptr;
        }
    }
    return g_cublas_handle;
}

// ── GpuMatrix ────────────────────────────────────────────────────────────────

GpuMatrix::GpuMatrix(int r, int c) : rows(r), cols(c) {
    cudaMalloc(&d_ptr, static_cast<size_t>(r) * c * sizeof(float));
    cudaMemset(d_ptr, 0, static_cast<size_t>(r) * c * sizeof(float));
}

GpuMatrix::~GpuMatrix() {
    if (d_ptr) { cudaFree(d_ptr); d_ptr = nullptr; }
}

GpuMatrix::GpuMatrix(GpuMatrix&& other) noexcept
    : d_ptr(other.d_ptr), rows(other.rows), cols(other.cols) {
    other.d_ptr = nullptr; other.rows = 0; other.cols = 0;
}

GpuMatrix& GpuMatrix::operator=(GpuMatrix&& other) noexcept {
    if (this != &other) {
        if (d_ptr) cudaFree(d_ptr);
        d_ptr = other.d_ptr; rows = other.rows; cols = other.cols;
        other.d_ptr = nullptr; other.rows = 0; other.cols = 0;
    }
    return *this;
}

void GpuMatrix::to_device(const std::vector<std::vector<float>>& cpu_matrix) {
    if (cpu_matrix.empty() || cpu_matrix[0].empty()) return;
    int r = static_cast<int>(cpu_matrix.size());
    int c = static_cast<int>(cpu_matrix[0].size());

    // Realloc if size changed
    if (r != rows || c != cols) {
        if (d_ptr) cudaFree(d_ptr);
        rows = r; cols = c;
        cudaMalloc(&d_ptr, static_cast<size_t>(r) * c * sizeof(float));
    }

    std::vector<float> flat(r * c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            flat[i * c + j] = cpu_matrix[i][j];

    cudaMemcpy(d_ptr, flat.data(), static_cast<size_t>(r) * c * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<std::vector<float>> GpuMatrix::from_device() const {
    if (!d_ptr || rows == 0 || cols == 0) return {};
    std::vector<float> flat(static_cast<size_t>(rows) * cols);
    cudaMemcpy(flat.data(), d_ptr, flat.size() * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<std::vector<float>> out(rows, std::vector<float>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out[i][j] = flat[i * cols + j];
    return out;
}

void GpuMatrix::gemv_device(const float* d_x, float* d_out) const {
    if (!d_ptr || rows == 0 || cols == 0) return;
    cublasHandle_t handle = get_cublas_handle();
    if (!handle) return;

    // cuBLAS SGEMV: y = alpha * A * x + beta * y
    // cuBLAS is column-major; our matrix is row-major.
    // To compute y = W * x (row-major W, rows×cols):
    // Treat as column-major W^T (cols×rows) and use CUBLAS_OP_T:
    //   y = A^T * x where A is cols×rows stored in our row-major array.
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle,
                CUBLAS_OP_T,    // transpose: treat col-major as row-major
                cols, rows,     // dimensions of A as col-major (cols×rows)
                &alpha,
                d_ptr, cols,    // A (our row-major flat array, leading dim = cols)
                d_x, 1,
                &beta,
                d_out, 1);
}

std::vector<float> GpuMatrix::gemv_cpu(const std::vector<float>& x) const {
    if (!d_ptr || rows == 0 || cols == 0 || x.empty()) return {};
    timing::ScopedCudaTimerAccumulator cuda_timer;

    float* d_x = ScratchBufferManager::instance().get_float_in(cols);
    float* d_y = ScratchBufferManager::instance().get_float_out(rows);

    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, rows * sizeof(float));

    gemv_device(d_x, d_y);
    cudaDeviceSynchronize();

    std::vector<float> result(rows);
    cudaMemcpy(result.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}

// ── GpuVector ────────────────────────────────────────────────────────────────

GpuVector::GpuVector(int sz) : size(sz) {
    cudaMalloc(&d_ptr, sz * sizeof(float));
    cudaMemset(d_ptr, 0, sz * sizeof(float));
}

GpuVector::~GpuVector() {
    if (d_ptr) { cudaFree(d_ptr); d_ptr = nullptr; }
}

GpuVector::GpuVector(GpuVector&& other) noexcept
    : d_ptr(other.d_ptr), size(other.size) {
    other.d_ptr = nullptr; other.size = 0;
}

GpuVector& GpuVector::operator=(GpuVector&& other) noexcept {
    if (this != &other) {
        if (d_ptr) cudaFree(d_ptr);
        d_ptr = other.d_ptr; size = other.size;
        other.d_ptr = nullptr; other.size = 0;
    }
    return *this;
}

void GpuVector::upload(const std::vector<float>& cpu_vec) {
    if (cpu_vec.empty()) return;
    int sz = static_cast<int>(cpu_vec.size());
    if (sz != size) {
        if (d_ptr) cudaFree(d_ptr);
        size = sz;
        cudaMalloc(&d_ptr, sz * sizeof(float));
    }
    cudaMemcpy(d_ptr, cpu_vec.data(), sz * sizeof(float), cudaMemcpyHostToDevice);
}

std::vector<float> GpuVector::download() const {
    if (!d_ptr || size == 0) return {};
    std::vector<float> out(size);
    cudaMemcpy(out.data(), d_ptr, size * sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

// ── Batched sequence GEMM ────────────────────────────────────────────────────

std::vector<std::vector<float>> GpuMatrix::gemm_sequence(
        const std::vector<std::vector<float>>& seq_input) const {
    if (!d_ptr || rows == 0 || cols == 0 || seq_input.empty()) return {};
    timing::ScopedCudaTimerAccumulator cuda_timer;

    int T   = static_cast<int>(seq_input.size());   // num tokens
    int K   = cols;                                  // input features  (= W cols)
    int N   = rows;                                  // output features (= W rows)

    // Pack input into flat row-major [T x K]
    std::vector<float> flat_in(T * K, 0.0f);
    for (int t = 0; t < T; ++t) {
        int lim = static_cast<int>(seq_input[t].size());
        for (int k = 0; k < K && k < lim; ++k)
            flat_in[t * K + k] = seq_input[t][k];
    }

    float* d_in = ScratchBufferManager::instance().get_float_in(static_cast<size_t>(T) * K);
    float* d_out = ScratchBufferManager::instance().get_float_out(static_cast<size_t>(T) * N);
    cudaMemset(d_out, 0, static_cast<size_t>(T) * N * sizeof(float));
    cudaMemcpy(d_in, flat_in.data(), flat_in.size() * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle = get_cublas_handle();
    if (handle) {
        // Compute Out[T x N] = In[T x K] * W^T[K x N]
        // Using cuBLAS column-major convention:
        //   Op(A) = W^T   (cols x rows stored row-major → treat as col-major N x K, no-transpose → CUBLAS_OP_N)
        //   Op(B) = In^T  (T x K stored row-major → treat as K x T col-major, no-transpose → CUBLAS_OP_N)
        //   C = W * In^T  → columns of C = rows of Out (need transpose of C = Out)
        // Simpler: C[N x T] = W[N x K] * In^T[K x T]
        // In cuBLAS (col-major): C = alpha * A * B + beta * C
        //   A = W (N x K row-major) treated as col-major K x N → CUBLAS_OP_T, lda=K
        //   B = In (T x K row-major) treated as col-major K x T → CUBLAS_OP_N, ldb=K
        //   C = Out (N x T col-major), ldc=N
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle,
                    CUBLAS_OP_T,   // W^T
                    CUBLAS_OP_N,   // In as-is (col-major K x T)
                    N, T, K,       // m, n, k
                    &alpha,
                    d_ptr, K,      // A = W, lda = K (row-major W stored as col-major W^T)
                    d_in,  K,      // B = In, ldb = K
                    &beta,
                    d_out, N);     // C = Out, ldc = N
        cudaDeviceSynchronize();
    }

    std::vector<float> flat_out(static_cast<size_t>(T) * N);
    cudaMemcpy(flat_out.data(), d_out, flat_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Unpack col-major [N x T] output (ldc=N) to row-major [T x N] result
    std::vector<std::vector<float>> result(T, std::vector<float>(N));
    for (int t = 0; t < T; ++t)
        for (int n = 0; n < N; ++n)
            result[t][n] = flat_out[t * N + n];

    return result;
}

} // namespace matmul_free

#endif // USE_CUDA
