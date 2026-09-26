#include "indus/gpu.hpp"
#include "indus/pdhg.hpp"
#include <iostream>

#ifdef INDUS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace indus::gpu {

#ifdef INDUS_HAS_CUDA
// Forward declare actual CUDA solver implemented in pdhg_cuda.cu
Solution solve_pdhg_cuda_internal(const Model& model, const Options& options);
#endif

GpuDeviceInfo probe_device() noexcept {
    GpuDeviceInfo info;
#ifdef INDUS_HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        info.available = false;
        info.device_name = "None (No CUDA devices found)";
        return info;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        info.available = false;
        info.device_name = "Error querying CUDA device";
        return info;
    }

    info.available = true;
    info.device_name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multi_processor_count = prop.multiProcessorCount;

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        info.free_vram_bytes = free_bytes;
        info.total_vram_bytes = total_bytes;
    } else {
        info.total_vram_bytes = prop.totalGlobalMem;
    }
#else
    info.available = false;
    info.device_name = "None (CUDA support disabled at compile time / CPU only)";
    info.total_vram_bytes = 0;
    info.free_vram_bytes = 0;
    info.compute_capability_major = 0;
    info.compute_capability_minor = 0;
    info.multi_processor_count = 0;
#endif
    return info;
}

bool is_cuda_available() noexcept {
    return probe_device().available;
}

Solution solve_pdhg_gpu(const Model& model, const Options& options) {
    GpuDeviceInfo info = probe_device();
    if (!info.available) {
        std::cout << "[INDUS-GPU] CUDA is not available on this machine (" << info.device_name
                  << "). Smoothly falling back to CPU Restarted PDHG solver.\n";
        return pdhg::solve_pdhg_cpu(model, options);
    }

#ifdef INDUS_HAS_CUDA
    std::cout << "[INDUS-GPU] Dispatching to NVIDIA CUDA device: " << info.device_name
              << " (" << (info.total_vram_bytes / (1024 * 1024)) << " MB VRAM)\n";
    return solve_pdhg_cuda_internal(model, options);
#else
    return pdhg::solve_pdhg_cpu(model, options);
#endif
}

#ifndef INDUS_HAS_CUDA
GpuTimingDiagnostics get_last_gpu_timing() noexcept {
    GpuTimingDiagnostics diag;
    diag.device_name = "None (CPU Fallback)";
    diag.kernel_mode = "CPU Reference Engine";
    diag.host_device_transfers = 0;
    return diag;
}
#endif

} // namespace indus::gpu
