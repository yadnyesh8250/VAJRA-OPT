#include "indus/gpu.hpp"
#include "indus/pdhg.hpp"
#include "indus/tolerances.hpp"
#include "src/linalg/scaling.hpp"

#ifdef INDUS_HAS_CUDA
#include "src/solvers/gpu/spmv_kernels.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace indus::gpu {

namespace {

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    explicit CudaBuffer(size_t n) { allocate(n); }
    ~CudaBuffer() { free(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            free();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    void allocate(size_t n) {
        free();
        if (n > 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)));
            count_ = n;
        }
    }

    void upload(const T* host_data, size_t n) {
        if (n > count_) allocate(n);
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(ptr_, host_data, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void download(T* host_data, size_t n) const {
        if (n > 0 && ptr_) {
            CUDA_CHECK(cudaMemcpy(host_data, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    void zero() {
        if (count_ > 0 && ptr_) {
            CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
        }
    }

    void free() noexcept {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    [[nodiscard]] T* get() noexcept { return ptr_; }
    [[nodiscard]] const T* get() const noexcept { return ptr_; }
    [[nodiscard]] size_t size() const noexcept { return count_; }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

} // namespace

Solution solve_pdhg_cuda_internal(const Model& model, const Options& options) {
    const auto start_time = std::chrono::high_resolution_clock::now();

    const int m = model.num_rows;
    const int n = model.num_cols;
    const bool is_max = (model.sense == ObjSense::kMaximize);

    Solution sol;
    sol.algorithm_used = "pdhg_cuda";
    sol.col_value.resize(static_cast<size_t>(n), 0.0);
    sol.row_value.resize(static_cast<size_t>(m), 0.0);
    sol.row_dual.resize(static_cast<size_t>(m), 0.0);
    sol.col_dual.resize(static_cast<size_t>(n), 0.0);
    sol.col_basis_status.resize(static_cast<size_t>(n), BasisStatus::kUnknown);
    sol.row_basis_status.resize(static_cast<size_t>(m), BasisStatus::kUnknown);

    if (m == 0 || n == 0) {
        return pdhg::solve_pdhg_cpu(model, options);
    }

    // Scaling & preparation
    la::ScalingFactors scaling_factors;
    la::SparseMatrixCSC work_A = model.A;
    std::vector<double> work_c = model.c;
    if (is_max) {
        for (double& cj : work_c) cj = -cj;
    }
    std::vector<double> work_row_lower = model.row_lower;
    std::vector<double> work_row_upper = model.row_upper;
    std::vector<double> work_col_lower = model.col_lower;
    std::vector<double> work_col_upper = model.col_upper;

    const bool enable_scaling = options.enable_scaling;
    if (enable_scaling) {
        scaling_factors = la::RuizScaler::compute_and_scale(
            work_A, work_c,
            work_row_lower, work_row_upper,
            work_col_lower, work_col_upper,
            10
        );
    }

    // Construct CSR representation of A for forward SpMV
    la::SparseMatrixCSR csr_A = work_A.to_csr();

    // Estimate spectral norm ||A||_2
    const double spectral_norm = pdhg::estimate_spectral_norm(work_A, 25, 1e-4);
    const double safety = 0.95;
    double tau = safety / spectral_norm;
    double sigma = safety / spectral_norm;

    // Initial primal iterate x0
    std::vector<double> h_x(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        const double lj = work_col_lower[static_cast<size_t>(j)];
        const double uj = work_col_upper[static_cast<size_t>(j)];
        double val = 0.0;
        if (val < lj) val = lj;
        if (val > uj) val = uj;
        h_x[static_cast<size_t>(j)] = val;
    }

    // UPLOAD ONCE TO GPU VRAM BEFORE ITERATION LOOP
    const auto upload_start = std::chrono::high_resolution_clock::now();

    CudaBuffer<int64_t> d_row_ptr(static_cast<size_t>(m + 1));
    CudaBuffer<int> d_col_idx(static_cast<size_t>(csr_A.nnz()));
    CudaBuffer<double> d_csr_values(static_cast<size_t>(csr_A.nnz()));

    d_row_ptr.upload(csr_A.row_ptr.data(), static_cast<size_t>(m + 1));
    d_col_idx.upload(csr_A.col_idx.data(), static_cast<size_t>(csr_A.nnz()));
    d_csr_values.upload(csr_A.values.data(), static_cast<size_t>(csr_A.nnz()));

    CudaBuffer<int64_t> d_col_ptr(static_cast<size_t>(n + 1));
    CudaBuffer<int> d_row_idx(static_cast<size_t>(work_A.nnz()));
    CudaBuffer<double> d_csc_values(static_cast<size_t>(work_A.nnz()));

    d_col_ptr.upload(work_A.col_ptr.data(), static_cast<size_t>(n + 1));
    d_row_idx.upload(work_A.row_idx.data(), static_cast<size_t>(work_A.nnz()));
    d_csc_values.upload(work_A.values.data(), static_cast<size_t>(work_A.nnz()));

    CudaBuffer<double> d_row_lower(static_cast<size_t>(m));
    CudaBuffer<double> d_row_upper(static_cast<size_t>(m));
    CudaBuffer<double> d_col_lower(static_cast<size_t>(n));
    CudaBuffer<double> d_col_upper(static_cast<size_t>(n));
    CudaBuffer<double> d_c(static_cast<size_t>(n));

    d_row_lower.upload(work_row_lower.data(), static_cast<size_t>(m));
    d_row_upper.upload(work_row_upper.data(), static_cast<size_t>(m));
    d_col_lower.upload(work_col_lower.data(), static_cast<size_t>(n));
    d_col_upper.upload(work_col_upper.data(), static_cast<size_t>(n));
    d_c.upload(work_c.data(), static_cast<size_t>(n));

    // Iteration buffers in VRAM
    CudaBuffer<double> d_x(static_cast<size_t>(n));
    CudaBuffer<double> d_x_bar(static_cast<size_t>(n));
    CudaBuffer<double> d_y(static_cast<size_t>(m));
    CudaBuffer<double> d_w(static_cast<size_t>(m)); // A * x_bar
    CudaBuffer<double> d_v(static_cast<size_t>(n)); // A^T * y
    CudaBuffer<double> d_residuals(2);             // [0] = max_prim_viol, [1] = max_dual_viol

    d_x.upload(h_x.data(), static_cast<size_t>(n));
    d_x_bar.upload(h_x.data(), static_cast<size_t>(n));
    d_y.zero();

    const auto upload_end = std::chrono::high_resolution_clock::now();
    const double upload_time_sec = std::chrono::duration<double>(upload_end - upload_start).count();

    // Warp-level kernel configurations (32 threads per warp, 8 warps per block = 256 threads)
    const int threads_per_warp = 32;
    const int warps_per_block = 8;
    const int block_size = warps_per_block * threads_per_warp;
    const int grid_warp_rows = (m + warps_per_block - 1) / warps_per_block;
    const int grid_warp_cols = (n + warps_per_block - 1) / warps_per_block;
    const int grid_scalar_rows = (m + block_size - 1) / block_size;
    const int grid_scalar_cols = (n + block_size - 1) / block_size;
    const int grid_residuals = (std::max(m, n) + block_size - 1) / block_size;

    const int64_t max_iter = options.iteration_limit;
    const int check_freq = static_cast<int>(options.get_int("check_frequency", 200));
    const double tol_val = options.get_double("tolerance", 1e-6);

    const auto iter_start = std::chrono::high_resolution_clock::now();
    int64_t iter = 0;
    bool converged = false;
    double h_residuals[2] = {1e20, 1e20};

    // VRAM-RESIDENT ITERATION LOOP: ZERO HOST-DEVICE VECTOR TRANSFERS INSIDE LOOP
    for (iter = 1; iter <= max_iter; ++iter) {
        // 1. Forward Warp-Aggregated SpMV: d_w = A * d_x_bar
        kernels::spmv_csr_warp_kernel<<<grid_warp_rows, block_size>>>(
            m, d_row_ptr.get(), d_col_idx.get(), d_csr_values.get(),
            d_x_bar.get(), d_w.get());

        // 2. Dual Update & Moreau Projection: d_y = prox(d_y + sigma * d_w)
        kernels::dual_update_kernel<<<grid_scalar_rows, block_size>>>(
            m, sigma, d_w.get(), d_row_lower.get(), d_row_upper.get(), d_y.get());

        // 3. Transpose Warp-Aggregated SpMV: d_v = A^T * d_y
        kernels::spmv_csc_transpose_warp_kernel<<<grid_warp_cols, block_size>>>(
            n, d_col_ptr.get(), d_row_idx.get(), d_csc_values.get(),
            d_y.get(), d_v.get());

        // 4. Primal Update & Bound Clipping & Halpern Extrapolation
        kernels::primal_update_kernel<<<grid_scalar_cols, block_size>>>(
            n, tau, d_v.get(), d_c.get(), d_col_lower.get(), d_col_upper.get(),
            d_x.get(), d_x_bar.get());

        // Periodic stopping check: evaluates residuals on-device and downloads ONLY 2 scalars (16 bytes)
        if (iter % check_freq == 0 || iter == max_iter) {
            d_residuals.zero();
            kernels::compute_residuals_device_kernel<<<grid_residuals, block_size>>>(
                m, n, d_w.get(), d_row_lower.get(), d_row_upper.get(),
                d_v.get(), d_c.get(), d_x.get(), d_col_lower.get(), d_col_upper.get(),
                d_residuals.get());

            // Download ONLY the 2 scalar residuals (no full vector transfer)
            d_residuals.download(h_residuals, 2);

            if (h_residuals[0] <= tol_val && h_residuals[1] <= tol_val) {
                converged = true;
                sol.status = SolveStatus::kOptimal;
                sol.status_message = "CUDA PDHG converged to required tolerance";
                break;
            }

            const auto now = std::chrono::high_resolution_clock::now();
            if (std::chrono::duration<double>(now - start_time).count() > options.time_limit) {
                sol.status = SolveStatus::kTimeLimit;
                break;
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto iter_end = std::chrono::high_resolution_clock::now();

    // DOWNLOAD FULL ITERATES ONCE UPON COMPLETION
    d_x.download(sol.col_value.data(), static_cast<size_t>(n));
    d_y.download(sol.row_dual.data(), static_cast<size_t>(m));

    if (scaling_factors.is_scaled) {
        la::RuizScaler::unscale_primal(scaling_factors, sol.col_value);
        la::RuizScaler::unscale_dual(scaling_factors, sol.row_dual);
    }

    // Convert dual multipliers: y_sol = -y_pdhg
    for (int i = 0; i < m; ++i) {
        sol.row_dual[static_cast<size_t>(i)] = is_max ? sol.row_dual[static_cast<size_t>(i)] : -sol.row_dual[static_cast<size_t>(i)];
    }

    // Compute row activities and objective
    model.A.multiply(sol.col_value, sol.row_value);

    double unscaled_obj = model.objective_offset;
    for (int j = 0; j < n; ++j) {
        unscaled_obj += model.c[static_cast<size_t>(j)] * sol.col_value[static_cast<size_t>(j)];
    }
    sol.objective_value = unscaled_obj;
    sol.best_dual_bound = unscaled_obj;

    // Reduced costs
    std::vector<double> a_trans_y(static_cast<size_t>(n), 0.0);
    model.A.multiply_transpose(sol.row_dual, a_trans_y);
    for (int j = 0; j < n; ++j) {
        sol.col_dual[static_cast<size_t>(j)] = model.c[static_cast<size_t>(j)] - a_trans_y[static_cast<size_t>(j)];
    }

    sol.iterations = iter;
    sol.solve_time_seconds = std::chrono::duration<double>(iter_end - start_time).count();
    if (!converged && sol.status == SolveStatus::kNotSolved) {
        sol.status = SolveStatus::kIterationLimit;
        sol.status_message = "CUDA PDHG reached iteration limit";
    }

    sol.recompute_quality(model);
    return sol;
}

} // namespace indus::gpu
#endif // INDUS_HAS_CUDA
