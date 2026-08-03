#include "timing.h"

namespace matmul_free {
namespace timing {

PerformanceStats g_stats;

void print_summary(std::ostream& os) {
    os << "\n=======================================================\n";
    os << "              Training Timing Summary                  \n";
    os << "=======================================================\n";
    os << " Total Epoch Time:           " << std::fixed << std::setprecision(2) << g_stats.total_epoch_time_ms << " ms\n";
    os << " Total Forward Pass Time:    " << std::fixed << std::setprecision(2) << g_stats.forward_time_ms << " ms\n";
    os << " Total Backward Attn Time:   " << std::fixed << std::setprecision(2) << g_stats.backward_attn_time_ms << " ms\n";
    os << " Total Backward FFN Time:    " << std::fixed << std::setprecision(2) << g_stats.backward_ffn_time_ms << " ms\n";
    os << " Total CUDA API/Call Time:   " << std::fixed << std::setprecision(2) << g_stats.cuda_call_time_ms << " ms (" << g_stats.cuda_call_count << " calls)\n";
    os << "=======================================================\n";
}

} // namespace timing
} // namespace matmul_free
