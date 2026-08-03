#ifndef MATMUL_FREE_TIMING_H
#define MATMUL_FREE_TIMING_H

#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdint>

namespace matmul_free {
namespace timing {

struct PerformanceStats {
    double total_epoch_time_ms = 0.0;
    double forward_time_ms = 0.0;
    double backward_attn_time_ms = 0.0;
    double backward_ffn_time_ms = 0.0;
    double cuda_call_time_ms = 0.0;
    uint64_t cuda_call_count = 0;

    void reset() {
        total_epoch_time_ms = 0.0;
        forward_time_ms = 0.0;
        backward_attn_time_ms = 0.0;
        backward_ffn_time_ms = 0.0;
        cuda_call_time_ms = 0.0;
        cuda_call_count = 0;
    }
};

extern PerformanceStats g_stats;

class ScopedTimerAccumulator {
public:
    explicit ScopedTimerAccumulator(double& accum_ms)
        : accum_ms_(accum_ms), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimerAccumulator() {
        auto end = std::chrono::high_resolution_clock::now();
        accum_ms_ += std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    double& accum_ms_;
    std::chrono::high_resolution_clock::time_point start_;
};

class ScopedCudaTimerAccumulator {
public:
    ScopedCudaTimerAccumulator()
        : start_(std::chrono::high_resolution_clock::now()) {
        g_stats.cuda_call_count++;
    }
    ~ScopedCudaTimerAccumulator() {
        auto end = std::chrono::high_resolution_clock::now();
        g_stats.cuda_call_time_ms += std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point start_;
};

void print_summary(std::ostream& os = std::cout);

} // namespace timing
} // namespace matmul_free

#endif // MATMUL_FREE_TIMING_H
