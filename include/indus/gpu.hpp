#pragma once

#include <string>
#include <cstddef>
#include <cstdint>
#include "indus/types.hpp"
#include "indus/model.hpp"
#include "indus/options.hpp"

namespace indus::gpu {

using indus::Model;
using indus::Options;
using indus::Solution;

struct GpuDeviceInfo {
    bool available = false;
    std::string device_name = "None";
    size_t total_vram_bytes = 0;
    size_t free_vram_bytes = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multi_processor_count = 0;
};

// Fixed-size device diagnostics copied at periodic intervals (zero full-vector transfer)
struct DeviceDiagnostics {
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    double objective = 0.0;
    double duality_gap = 0.0;
    int converged = 0;
    int numerical_error = 0;
};

// Comprehensive timing and PCIe transfer diagnostics
struct GpuTimingDiagnostics {
    double upload_time_sec = 0.0;
    double iteration_time_sec = 0.0;
    double download_time_sec = 0.0;
    double total_time_sec = 0.0;
    int64_t host_device_transfers = 0;
    int64_t iterations = 0;
    std::string device_name = "None";
    std::string kernel_mode = "warp_aggregated";
};

// Probe the current hardware for CUDA support
[[nodiscard]] GpuDeviceInfo probe_device() noexcept;

// Check if CUDA acceleration is available on this system
[[nodiscard]] bool is_cuda_available() noexcept;

// Retrieve timing and PCIe telemetry from the most recent GPU solve
[[nodiscard]] GpuTimingDiagnostics get_last_gpu_timing() noexcept;

// Solve LP using GPU CUDA Restarted PDHG with zero-host-transfer loop.
// If CUDA is unavailable, cleanly logs and falls back to CPU PDHG reference solver.
Solution solve_pdhg_gpu(const Model& model, const Options& options = Options());

} // namespace indus::gpu
