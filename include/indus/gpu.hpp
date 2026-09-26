#pragma once

#include <string>
#include <cstddef>
#include "indus/types.hpp"
#include "indus/model.hpp"
#include "indus/options.hpp"

namespace indus::gpu {

struct GpuDeviceInfo {
    bool available = false;
    std::string device_name = "None";
    size_t total_vram_bytes = 0;
    size_t free_vram_bytes = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multi_processor_count = 0;
};

// Probe the current hardware for CUDA support
[[nodiscard]] GpuDeviceInfo probe_device() noexcept;

// Check if CUDA acceleration is available on this system
[[nodiscard]] bool is_cuda_available() noexcept;

// Solve LP using GPU CUDA Restarted PDHG with zero-host-transfer loop.
// If CUDA is unavailable, cleanly logs and falls back to CPU PDHG reference solver.
Solution solve_pdhg_gpu(const Model& model, const Options& options = Options());

} // namespace indus::gpu
