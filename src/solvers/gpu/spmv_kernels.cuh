#pragma once

#ifdef INDUS_HAS_CUDA
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "indus/gpu.hpp"

namespace indus::gpu::kernels {

// Device-side atomicMax for doubles using atomicCAS
__device__ inline void atomic_max_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

// Device-side atomicAdd for doubles
__device__ inline void atomic_add_double(double* address, double val) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, val);
#else
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

// 1. Forward CSR SpMV: Warp-aggregated cooperative reduction
// Each 32-thread warp computes one row of matrix A using __shfl_down_sync
__global__ void spmv_csr_warp_kernel(
    int num_rows,
    const int64_t* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int warp_id = global_thread_id >> 5;
    const int lane_id = global_thread_id & 31;
    const int row = warp_id;

    if (row < num_rows) {
        const int64_t row_start = row_ptr[row];
        const int64_t row_end   = row_ptr[row + 1];

        // Safe handling of zero rows / empty constraints
        if (row_start >= row_end) {
            if (lane_id == 0) y[row] = 0.0;
            return;
        }

        double sum = 0.0;
        for (int64_t idx = row_start + lane_id; idx < row_end; idx += 32) {
            sum += values[idx] * x[col_idx[idx]];
        }

        // Intra-warp tree reduction using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

// Forward CSR SpMV: Adaptive scalar kernel for very short/sparse rows
__global__ void spmv_csr_scalar_kernel(
    int num_rows,
    const int64_t* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const double* __restrict__ values,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        int64_t start = row_ptr[row];
        int64_t end = row_ptr[row + 1];
        double sum = 0.0;
        for (int64_t idx = start; idx < end; ++idx) {
            sum += values[idx] * x[col_idx[idx]];
        }
        y[row] = sum;
    }
}

// 2. Transpose SpMV: Warp-aggregated cooperative reduction
// Using CSC column representation (which is CSR of A^T) with zero write conflicts
__global__ void spmv_csc_transpose_warp_kernel(
    int num_cols,
    const int64_t* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const double* __restrict__ values,
    const double* __restrict__ y,
    double* __restrict__ v)
{
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int warp_id = global_thread_id >> 5;
    const int lane_id = global_thread_id & 31;
    const int col = warp_id;

    if (col < num_cols) {
        const int64_t col_start = col_ptr[col];
        const int64_t col_end   = col_ptr[col + 1];

        // Safe handling of empty columns
        if (col_start >= col_end) {
            if (lane_id == 0) v[col] = 0.0;
            return;
        }

        double sum = 0.0;
        for (int64_t idx = col_start + lane_id; idx < col_end; idx += 32) {
            sum += values[idx] * y[row_idx[idx]];
        }

        // Intra-warp tree reduction using shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            v[col] = sum;
        }
    }
}

// Transpose SpMV: Adaptive scalar kernel for short columns
__global__ void spmv_csc_transpose_scalar_kernel(
    int num_cols,
    const int64_t* __restrict__ col_ptr,
    const int* __restrict__ row_idx,
    const double* __restrict__ values,
    const double* __restrict__ y,
    double* __restrict__ v)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col < num_cols) {
        int64_t start = col_ptr[col];
        int64_t end = col_ptr[col + 1];
        double sum = 0.0;
        for (int64_t idx = start; idx < end; ++idx) {
            sum += values[idx] * y[row_idx[idx]];
        }
        v[col] = sum;
    }
}

// 3. Dual update with closed-form Moreau proximal projection
// Handles equality (l=u), <=, >=, ranged (l<u), and free rows branchlessly
__global__ void dual_update_kernel(
    int num_rows,
    double sigma,
    const double* __restrict__ Ax,
    const double* __restrict__ row_lower,
    const double* __restrict__ row_upper,
    double* __restrict__ y)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_rows) {
        double y_hat = y[i] + sigma * Ax[i];
        double l = row_lower[i];
        double u = row_upper[i];
        double val = y_hat / sigma;
        double clamped = (val < l) ? l : ((val > u) ? u : val);
        y[i] = y_hat - sigma * clamped;
    }
}

// 4. Primal update, coordinate bounds projection, and Halpern/Malitsky extrapolation
// x_j^{k+1} = clamp(x_j^k - tau * (v_j + c_j), l_j, u_j)
// x_bar_j^{k+1} = 2 * x_j^{k+1} - x_j^k
__global__ void primal_update_kernel(
    int num_cols,
    double tau,
    const double* __restrict__ Aty,
    const double* __restrict__ c,
    const double* __restrict__ col_lower,
    const double* __restrict__ col_upper,
    double* __restrict__ x,
    double* __restrict__ x_bar)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < num_cols) {
        double x_old = x[j];
        double grad = Aty[j] + c[j];
        double x_new = x_old - tau * grad;
        double l = col_lower[j];
        double u = col_upper[j];
        x_new = (x_new < l) ? l : ((x_new > u) ? u : x_new);
        x[j] = x_new;
        x_bar[j] = 2.0 * x_new - x_old;
    }
}

// 5. Device-Side Diagnostics Reduction Kernel
// Computes max primal residual, dual residual, objective, and duality gap in VRAM
__global__ void compute_diagnostics_device_kernel(
    int num_rows,
    int num_cols,
    const double* __restrict__ Ax,
    const double* __restrict__ row_lower,
    const double* __restrict__ row_upper,
    const double* __restrict__ Aty,
    const double* __restrict__ c,
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ col_lower,
    const double* __restrict__ col_upper,
    double tol_val,
    DeviceDiagnostics* __restrict__ d_diag)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // A. Primal row feasibility & NaN/Inf detection
    if (idx < num_rows) {
        double axi = Ax[idx];
        double yi = y[idx];
        if (isnan(axi) || isinf(axi) || isnan(yi) || isinf(yi)) {
            d_diag->numerical_error = 1;
        }

        double li = row_lower[idx];
        double ui = row_upper[idx];
        double viol = 0.0;
        if (axi < li) viol = li - axi;
        if (axi > ui) viol = axi - ui;
        if (viol > 0.0) {
            atomic_max_double(&d_diag->primal_residual, viol);
        }
    }

    // B. Primal column bounds, Dual feasibility, and Objective
    if (idx < num_cols) {
        double xj = x[idx];
        double cj = c[idx];
        double atyj = Aty[idx];
        if (isnan(xj) || isinf(xj) || isnan(cj) || isinf(cj) || isnan(atyj) || isinf(atyj)) {
            d_diag->numerical_error = 1;
        }

        double lj = col_lower[idx];
        double uj = col_upper[idx];

        // Column bounds violation
        double p_viol = 0.0;
        if (xj < lj) p_viol = lj - xj;
        if (xj > uj) p_viol = xj - uj;
        if (p_viol > 0.0) {
            atomic_max_double(&d_diag->primal_residual, p_viol);
        }

        // Reduced cost: dj = cj + (A^T y)_j
        double dj = cj + atyj;
        double d_viol = 0.0;
        if (xj > lj + 1e-5 && dj > 1e-6) d_viol = dj;
        if (xj < uj - 1e-5 && dj < -1e-6) d_viol = -dj;
        if (lj <= -1e20 && uj >= 1e20) d_viol = fabs(dj);
        if (d_viol > 0.0) {
            atomic_max_double(&d_diag->dual_residual, d_viol);
        }

        // Objective contribution
        atomic_add_double(&d_diag->objective, cj * xj);
    }
}

// Extrapolation reset for adaptive restart
__global__ void reset_extrapolation_kernel(
    int num_cols,
    const double* __restrict__ x,
    double* __restrict__ x_bar)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < num_cols) {
        x_bar[j] = x[j];
    }
}

} // namespace indus::gpu::kernels
#endif // INDUS_HAS_CUDA
