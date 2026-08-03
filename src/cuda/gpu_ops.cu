#include "gpu_ops.h"
#include "gpu_buffer.h"
#include "../core/timing.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace matmul_free {

__global__ void kernel_matmul_vector(const float* weight, const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float acc = 0.0f;
        const float* w_row = weight + row * cols;
        for (int col = 0; col < cols; ++col) {
            acc += w_row[col] * input[col];
        }
        output[row] = acc;
    }
}

__global__ void kernel_bitlinear_vector(const int8_t* weight, const float* input, float* output, int rows, int cols, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float acc = 0.0f;
        const int8_t* w_row = weight + row * cols;
        for (int col = 0; col < cols; ++col) {
            acc += static_cast<float>(w_row[col]) * input[col];
        }
        output[row] = acc * scale;
    }
}

__global__ void kernel_bitlinear_packed(const uint8_t* packed_weight, const float* input, float* output, int rows, int cols, int packed_cols, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float acc = 0.0f;
        const uint8_t* p_row = packed_weight + row * packed_cols;
        int col = 0;
        for (int p = 0; p < packed_cols && col < cols; ++p) {
            uint8_t b = p_row[p];
            for (int shift = 0; shift < 8 && col < cols; shift += 2) {
                uint8_t code = (b >> shift) & 0b11;
                float mult = (code == 1) ? 1.0f : ((code == 2) ? -1.0f : 0.0f);
                acc += mult * input[col++];
            }
        }
        output[row] = acc * scale;
    }
}

bool is_cuda_available() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
}

void print_cuda_device_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "  [CUDA GPU Acceleration: ACTIVE]\n";
        std::cout << "  Device 0: " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Total VRAM: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    } else {
        std::cout << "  [CUDA GPU Acceleration: NOT AVAILABLE]\n";
    }
}

std::vector<float> cuda_matmul_vector(const std::vector<std::vector<float>>& weight,
                                      const std::vector<float>& input) {
    if (weight.empty() || weight[0].empty() || input.empty()) return {};
    timing::ScopedCudaTimerAccumulator cuda_timer;
    int rows = static_cast<int>(weight.size());
    int cols = static_cast<int>(weight[0].size());

    std::vector<float> flat_w(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat_w[i * cols + j] = weight[i][j];
        }
    }

    float* d_w = ScratchBufferManager::instance().get_float_weight(static_cast<size_t>(rows) * cols);
    float* d_x = ScratchBufferManager::instance().get_float_in(cols);
    float* d_y = ScratchBufferManager::instance().get_float_out(rows);

    cudaMemcpy(d_w, flat_w.data(), rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, input.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    kernel_matmul_vector<<<grid_size, block_size>>>(d_w, d_x, d_y, rows, cols);
    cudaDeviceSynchronize();

    std::vector<float> output(rows);
    cudaMemcpy(output.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}

std::vector<float> cuda_bitlinear_vector(const std::vector<std::vector<int8_t>>& weight_ternary,
                                         const std::vector<float>& input,
                                         float scale) {
    if (weight_ternary.empty() || weight_ternary[0].empty() || input.empty()) return {};
    timing::ScopedCudaTimerAccumulator cuda_timer;
    int rows = static_cast<int>(weight_ternary.size());
    int cols = static_cast<int>(weight_ternary[0].size());

    std::vector<int8_t> flat_w(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat_w[i * cols + j] = weight_ternary[i][j];
        }
    }

    int8_t* d_w = ScratchBufferManager::instance().get_int8_weight(static_cast<size_t>(rows) * cols);
    float* d_x = ScratchBufferManager::instance().get_float_in(cols);
    float* d_y = ScratchBufferManager::instance().get_float_out(rows);

    cudaMemcpy(d_w, flat_w.data(), rows * cols * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, input.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    kernel_bitlinear_vector<<<grid_size, block_size>>>(d_w, d_x, d_y, rows, cols, scale);
    cudaDeviceSynchronize();

    std::vector<float> output(rows);
    cudaMemcpy(output.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}

std::vector<float> cuda_bitlinear_packed(const std::vector<std::vector<uint8_t>>& packed_weight,
                                         const std::vector<float>& input,
                                         float scale) {
    if (packed_weight.empty() || packed_weight[0].empty() || input.empty()) return {};
    timing::ScopedCudaTimerAccumulator cuda_timer;
    int rows = static_cast<int>(packed_weight.size());
    int packed_cols = static_cast<int>(packed_weight[0].size());
    int cols = static_cast<int>(input.size());

    std::vector<uint8_t> flat_w(rows * packed_cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < packed_cols; ++j) {
            flat_w[i * packed_cols + j] = packed_weight[i][j];
        }
    }

    uint8_t* d_w = ScratchBufferManager::instance().get_uint8_weight(static_cast<size_t>(rows) * packed_cols);
    float* d_x = ScratchBufferManager::instance().get_float_in(cols);
    float* d_y = ScratchBufferManager::instance().get_float_out(rows);

    cudaMemcpy(d_w, flat_w.data(), rows * packed_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, input.data(), cols * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (rows + block_size - 1) / block_size;
    kernel_bitlinear_packed<<<grid_size, block_size>>>(d_w, d_x, d_y, rows, cols, packed_cols, scale);
    cudaDeviceSynchronize();

    std::vector<float> output(rows);
    cudaMemcpy(output.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    return output;
}

__global__ void kernel_matmul_matrix(const float* A, const float* W, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        const float* a_row = A + row * K;
        const float* w_row = W + col * K;
        for (int k = 0; k < K; ++k) {
            sum += a_row[k] * w_row[k];
        }
        C[row * N + col] = sum;
    }
}

std::vector<std::vector<float>> cuda_matmul_matrix(const std::vector<std::vector<float>>& A,
                                                   const std::vector<std::vector<float>>& W) {
    if (A.empty() || A[0].empty() || W.empty() || W[0].empty()) return {};
    int M = static_cast<int>(A.size());
    int K = static_cast<int>(A[0].size());
    int N = static_cast<int>(W.size());

    std::vector<float> flat_A(M * K);
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            flat_A[i * K + k] = A[i][k];
        }
    }

    std::vector<float> flat_W(N * K);
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < K; ++k) {
            flat_W[j * K + k] = W[j][k];
        }
    }

    float *d_A = nullptr, *d_W = nullptr, *d_C = nullptr;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_W, N * K * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, flat_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, flat_W.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    kernel_matmul_matrix<<<grid, block>>>(d_A, d_W, d_C, M, K, N);
    cudaDeviceSynchronize();

    std::vector<float> flat_C(M * N);
    cudaMemcpy(flat_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_C);

    std::vector<std::vector<float>> C(M, std::vector<float>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i][j] = flat_C[i * N + j];
        }
    }

    return C;
}

__global__ void kernel_bitlinear_sequence(
    const uint8_t* packed_weight,
    const float* A,
    float* C,
    int T, int K, int N, int packed_K,
    float scale
) {
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < T && n < N) {
        float acc = 0.0f;
        const uint8_t* w_row = packed_weight + n * packed_K;
        const float* a_row = A + t * K;
        int k = 0;

        for (int p = 0; p < packed_K && k < K; ++p) {
            uint8_t byte_val = w_row[p];
            #pragma unroll
            for (int shift = 0; shift < 8 && k < K; shift += 2) {
                uint8_t code = (byte_val >> shift) & 0b11;
                if (code == 1) {
                    acc += a_row[k];
                } else if (code == 2) {
                    acc -= a_row[k];
                }
                k++;
            }
        }

        C[t * N + n] = acc * scale;
    }
}

std::vector<std::vector<float>> cuda_bitlinear_sequence(
    const std::vector<std::vector<uint8_t>>& packed_weight,
    const std::vector<std::vector<float>>& A,
    int unpacked_cols,
    float scale
) {
    if (packed_weight.empty() || packed_weight[0].empty() || A.empty() || A[0].empty()) return {};
    timing::ScopedCudaTimerAccumulator cuda_timer;

    int T = static_cast<int>(A.size());
    int K = unpacked_cols;
    int N = static_cast<int>(packed_weight.size());
    int packed_K = static_cast<int>(packed_weight[0].size());

    std::vector<uint8_t> flat_W(N * packed_K);
    for (int n = 0; n < N; ++n) {
        for (int p = 0; p < packed_K; ++p) {
            flat_W[n * packed_K + p] = packed_weight[n][p];
        }
    }

    std::vector<float> flat_A(T * K, 0.0f);
    for (int t = 0; t < T; ++t) {
        int lim = static_cast<int>(A[t].size());
        for (int k = 0; k < K && k < lim; ++k) {
            flat_A[t * K + k] = A[t][k];
        }
    }

    uint8_t* d_W = ScratchBufferManager::instance().get_uint8_weight(static_cast<size_t>(N) * packed_K);
    float* d_A = ScratchBufferManager::instance().get_float_in(static_cast<size_t>(T) * K);
    float* d_C = ScratchBufferManager::instance().get_float_out(static_cast<size_t>(T) * N);

    cudaMemcpy(d_W, flat_W.data(), N * packed_K * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, flat_A.data(), T * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (T + block.y - 1) / block.y);
    kernel_bitlinear_sequence<<<grid, block>>>(d_W, d_A, d_C, T, K, N, packed_K, scale);
    cudaDeviceSynchronize();

    std::vector<float> flat_C(T * N);
    cudaMemcpy(flat_C.data(), d_C, T * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<std::vector<float>> C(T, std::vector<float>(N));
    for (int t = 0; t < T; ++t) {
        for (int n = 0; n < N; ++n) {
            C[t][n] = flat_C[t * N + n];
        }
    }

    return C;
}

} // namespace matmul_free
