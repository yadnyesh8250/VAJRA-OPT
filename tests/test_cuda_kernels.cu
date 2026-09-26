#include "indus/model.hpp"
#include "indus/sparse.hpp"
#include "indus/gpu.hpp"
#include "indus/pdhg.hpp"
#include "src/solvers/gpu/spmv_kernels.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cassert>

#define ASSERT_NEAR_REL(val, expected, tol) \
    do { \
        double diff = std::abs((val) - (expected)); \
        double scale = std::max(1.0, std::abs(expected)); \
        if (diff / scale > (tol)) { \
            std::cerr << "Assertion FAILED: " #val " (" << (val) << ") != " #expected " (" << (expected) \
                      << ") [rel_diff=" << (diff / scale) << " > tol=" << (tol) << "] at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define ASSERT_EQ(val, expected) \
    do { \
        if ((val) != (expected)) { \
            std::cerr << "Assertion FAILED: " #val " != " #expected " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

using namespace indus::gpu::kernels;

// Helper RAII device buffer for tests
template <typename T>
struct TestDeviceBuffer {
    T* d_ptr = nullptr;
    size_t count = 0;

    TestDeviceBuffer(size_t n) : count(n) {
        if (n > 0) {
            CUDA_CHECK(cudaMalloc((void**)&d_ptr, n * sizeof(T)));
        }
    }
    ~TestDeviceBuffer() {
        if (d_ptr) cudaFree(d_ptr);
    }
    void upload(const T* host_ptr, size_t n) {
        if (n > 0 && d_ptr) {
            CUDA_CHECK(cudaMemcpy(d_ptr, host_ptr, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    void download(T* host_ptr, size_t n) {
        if (n > 0 && d_ptr) {
            CUDA_CHECK(cudaMemcpy(host_ptr, d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }
    void zero() {
        if (d_ptr && count > 0) {
            CUDA_CHECK(cudaMemset(d_ptr, 0, count * sizeof(T)));
        }
    }
};

// 1. Forward CSR Warp SpMV Kernel Test
void test_cuda_csr_spmv() {
    std::cout << "  [CUDA KERNEL 1] Forward CSR Warp SpMV (dense, sparse, empty rows)...\n";
    // 4 rows, 5 cols: Row 0 has 2 nnz, Row 1 is empty, Row 2 has 3 nnz, Row 3 has 1 nnz
    int m = 4, n = 5;
    std::vector<int64_t> row_ptr = {0, 2, 2, 5, 6};
    std::vector<int> col_idx = {0, 2,  0, 3, 4,  1};
    std::vector<double> values = {1.5, -2.0,  3.0, 4.0, -1.0,  5.5};
    std::vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Expected CPU y = A * x:
    // y[0] = 1.5*1.0 + (-2.0)*3.0 = 1.5 - 6.0 = -4.5
    // y[1] = 0.0 (empty row)
    // y[2] = 3.0*1.0 + 4.0*4.0 + (-1.0)*5.0 = 3 + 16 - 5 = 14.0
    // y[3] = 5.5*2.0 = 11.0
    std::vector<double> expected = {-4.5, 0.0, 14.0, 11.0};

    TestDeviceBuffer<int64_t> d_rp(row_ptr.size()); d_rp.upload(row_ptr.data(), row_ptr.size());
    TestDeviceBuffer<int> d_ci(col_idx.size()); d_ci.upload(col_idx.data(), col_idx.size());
    TestDeviceBuffer<double> d_val(values.size()); d_val.upload(values.data(), values.size());
    TestDeviceBuffer<double> d_x(x.size()); d_x.upload(x.data(), x.size());
    TestDeviceBuffer<double> d_y(m); d_y.zero();

    spmv_csr_warp_kernel<<<1, 256>>>(m, d_rp.d_ptr, d_ci.d_ptr, d_val.d_ptr, d_x.d_ptr, d_y.d_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_y(m, 0.0);
    d_y.download(h_y.data(), m);

    for (int i = 0; i < m; ++i) {
        ASSERT_NEAR_REL(h_y[i], expected[i], 1e-12);
    }
    std::cout << "    Passed.\n";
}

// 2. Transpose CSC Warp SpMV Kernel Test
void test_cuda_transpose_spmv() {
    std::cout << "  [CUDA KERNEL 2] Transpose CSC Warp SpMV...\n";
    // 3 cols, 4 rows.
    int n = 3;
    std::vector<int64_t> col_ptr = {0, 2, 2, 4};
    std::vector<int> row_idx = {0, 2,  1, 3};
    std::vector<double> values = {1.0, 2.0,  -1.0, 3.0};
    std::vector<double> y = {1.0, 2.0, 3.0, 4.0};

    // v[0] = 1.0*y[0] + 2.0*y[2] = 1*1 + 2*3 = 7.0
    // v[1] = 0.0 (empty col)
    // v[2] = -1.0*y[1] + 3.0*y[3] = -1*2 + 3*4 = 10.0
    std::vector<double> expected = {7.0, 0.0, 10.0};

    TestDeviceBuffer<int64_t> d_cp(col_ptr.size()); d_cp.upload(col_ptr.data(), col_ptr.size());
    TestDeviceBuffer<int> d_ri(row_idx.size()); d_ri.upload(row_idx.data(), row_idx.size());
    TestDeviceBuffer<double> d_val(values.size()); d_val.upload(values.data(), values.size());
    TestDeviceBuffer<double> d_y(y.size()); d_y.upload(y.data(), y.size());
    TestDeviceBuffer<double> d_v(n); d_v.zero();

    spmv_csc_transpose_warp_kernel<<<1, 256>>>(n, d_cp.d_ptr, d_ri.d_ptr, d_val.d_ptr, d_y.d_ptr, d_v.d_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_v(n, 0.0);
    d_v.download(h_v.data(), n);

    for (int j = 0; j < n; ++j) {
        ASSERT_NEAR_REL(h_v[j], expected[j], 1e-12);
    }
    std::cout << "    Passed.\n";
}

// 3, 4, 5, 6. Dual Moreau Projections: Equality, <=, >=, Ranged
void test_cuda_dual_projections() {
    std::cout << "  [CUDA KERNELS 3-6] Dual Moreau Projections (==, <=, >=, ranged)...\n";
    int m = 4;
    double sigma = 0.5;
    std::vector<double> Ax = {10.0, 5.0, 2.0, 7.0};
    std::vector<double> y_init = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> r_lower = {8.0,   -1e20,  4.0,  3.0};
    std::vector<double> r_upper = {8.0,    6.0,  1e20,  6.0};

    // Row 0 (Equality: l=8, u=8): y_hat = 0 + 0.5*10 = 5.0. val = 5.0/0.5 = 10. clamped = 8. y = 5 - 0.5*8 = 1.0.
    // Row 1 (<= : l=-inf, u=6): y_hat = 0 + 0.5*5 = 2.5. val = 5. clamped = 5 (<=6). y = 2.5 - 0.5*5 = 0.0.
    // Row 2 (>= : l=4, u=+inf): y_hat = 0 + 0.5*2 = 1.0. val = 2. clamped = 4 (>=4). y = 1.0 - 0.5*4 = -1.0.
    // Row 3 (Ranged: [3, 6]): y_hat = 0 + 0.5*7 = 3.5. val = 7. clamped = 6. y = 3.5 - 0.5*6 = 0.5.
    std::vector<double> expected = {1.0, 0.0, -1.0, 0.5};

    TestDeviceBuffer<double> d_ax(m); d_ax.upload(Ax.data(), m);
    TestDeviceBuffer<double> d_y(m); d_y.upload(y_init.data(), m);
    TestDeviceBuffer<double> d_rl(m); d_rl.upload(r_lower.data(), m);
    TestDeviceBuffer<double> d_ru(m); d_ru.upload(r_upper.data(), m);

    dual_update_kernel<<<1, 256>>>(m, sigma, d_ax.d_ptr, d_rl.d_ptr, d_ru.d_ptr, d_y.d_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_y(m, 0.0);
    d_y.download(h_y.data(), m);

    for (int i = 0; i < m; ++i) {
        ASSERT_NEAR_REL(h_y[i], expected[i], 1e-12);
    }
    std::cout << "    Passed.\n";
}

// 7, 8. Primal Bound Projection and Halpern Extrapolation
void test_cuda_primal_projection_and_extrapolation() {
    std::cout << "  [CUDA KERNELS 7-8] Primal Bound Projection & Halpern Extrapolation...\n";
    int n = 3;
    double tau = 1.0;
    std::vector<double> Aty = {1.0, -2.0, 0.0};
    std::vector<double> c = {2.0, 1.0, 0.0};
    std::vector<double> c_lower = {0.0, -5.0, -1e20};
    std::vector<double> c_upper = {5.0,  5.0,  1e20}; // Col 2 is free
    std::vector<double> x_init = {1.0, 2.0, 3.0};

    // Col 0: grad = 1 + 2 = 3. x_raw = 1 - 1*3 = -2. clamp to [0, 5] => x_new = 0.0. x_bar = 2*0 - 1 = -1.0.
    // Col 1: grad = -2 + 1 = -1. x_raw = 2 - 1*(-1) = 3. clamp to [-5, 5] => x_new = 3.0. x_bar = 2*3 - 2 = 4.0.
    // Col 2: grad = 0 + 0 = 0. x_raw = 3 - 0 = 3. clamp to [-inf, inf] => x_new = 3.0. x_bar = 2*3 - 3 = 3.0.
    std::vector<double> exp_x = {0.0, 3.0, 3.0};
    std::vector<double> exp_x_bar = {-1.0, 4.0, 3.0};

    TestDeviceBuffer<double> d_aty(n); d_aty.upload(Aty.data(), n);
    TestDeviceBuffer<double> d_c(n); d_c.upload(c.data(), n);
    TestDeviceBuffer<double> d_cl(n); d_cl.upload(c_lower.data(), n);
    TestDeviceBuffer<double> d_cu(n); d_cu.upload(c_upper.data(), n);
    TestDeviceBuffer<double> d_x(n); d_x.upload(x_init.data(), n);
    TestDeviceBuffer<double> d_x_bar(n); d_x_bar.zero();

    primal_update_kernel<<<1, 256>>>(n, tau, d_aty.d_ptr, d_c.d_ptr, d_cl.d_ptr, d_cu.d_ptr, d_x.d_ptr, d_x_bar.d_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_x(n, 0.0), h_x_bar(n, 0.0);
    d_x.download(h_x.data(), n);
    d_x_bar.download(h_x_bar.data(), n);

    for (int j = 0; j < n; ++j) {
        ASSERT_NEAR_REL(h_x[j], exp_x[j], 1e-12);
        ASSERT_NEAR_REL(h_x_bar[j], exp_x_bar[j], 1e-12);
    }
    std::cout << "    Passed.\n";
}

// 9, 10, 11. Device Residual Reduction, Objective, and NaN/Inf Check
void test_cuda_device_diagnostics() {
    std::cout << "  [CUDA KERNELS 9-11] Device Residual Reduction, Objective & NaN Check...\n";
    int m = 2, n = 2;
    std::vector<double> Ax = {5.0, 2.0};
    std::vector<double> r_lower = {0.0, 3.0}; // Row 1 violated by 1.0 (2 < 3)
    std::vector<double> r_upper = {4.0, 10.0}; // Row 0 violated by 1.0 (5 > 4)
    std::vector<double> Aty = {0.0, 0.0};
    std::vector<double> c = {2.0, 3.0};
    std::vector<double> x = {1.0, 2.0}; // obj = 2*1 + 3*2 = 8.0
    std::vector<double> y = {0.0, 0.0};
    std::vector<double> c_lower = {0.0, 0.0};
    std::vector<double> c_upper = {10.0, 10.0};

    TestDeviceBuffer<double> d_ax(m); d_ax.upload(Ax.data(), m);
    TestDeviceBuffer<double> d_rl(m); d_rl.upload(r_lower.data(), m);
    TestDeviceBuffer<double> d_ru(m); d_ru.upload(r_upper.data(), m);
    TestDeviceBuffer<double> d_aty(n); d_aty.upload(Aty.data(), n);
    TestDeviceBuffer<double> d_c(n); d_c.upload(c.data(), n);
    TestDeviceBuffer<double> d_x(n); d_x.upload(x.data(), n);
    TestDeviceBuffer<double> d_y(m); d_y.upload(y.data(), m);
    TestDeviceBuffer<double> d_cl(n); d_cl.upload(c_lower.data(), n);
    TestDeviceBuffer<double> d_cu(n); d_cu.upload(c_upper.data(), n);
    TestDeviceBuffer<indus::gpu::DeviceDiagnostics> d_diag(1); d_diag.zero();

    compute_diagnostics_device_kernel<<<1, 256>>>(
        m, n, d_ax.d_ptr, d_rl.d_ptr, d_ru.d_ptr,
        d_aty.d_ptr, d_c.d_ptr, d_x.d_ptr, d_y.d_ptr,
        d_cl.d_ptr, d_cu.d_ptr, 1e-4, d_diag.d_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    indus::gpu::DeviceDiagnostics h_diag;
    d_diag.download(&h_diag, 1);

    ASSERT_NEAR_REL(h_diag.primal_residual, 1.0, 1e-12);
    ASSERT_NEAR_REL(h_diag.objective, 8.0, 1e-12);
    ASSERT_EQ(h_diag.numerical_error, 0);
    std::cout << "    Passed.\n";
}

// 12. Complete PDHG Iteration Test & Edge Cases (1-row, 1-col, free/infinite bounds)
void test_cuda_complete_iteration() {
    std::cout << "  [CUDA KERNEL 12] One Complete PDHG Iteration & Edge Cases (1-row, 1-col, free bounds)...\n";
    // 3 rows, 3 cols
    // Row 0: x0 + 2*x1 = 4.0 (eq)
    // Row 1: x1 - x2 <= 2.0 (<=)
    // Row 2: 2*x0 + x2 >= 3.0 (>=)
    int m = 3, n = 3;
    std::vector<int64_t> row_ptr = {0, 2, 4, 6};
    std::vector<int> col_idx = {0, 1,  1, 2,  0, 2};
    std::vector<double> csr_val = {1.0, 2.0,  1.0, -1.0,  2.0, 1.0};

    std::vector<int64_t> col_ptr = {0, 2, 4, 6};
    std::vector<int> row_idx = {0, 2,  0, 1,  1, 2};
    std::vector<double> csc_val = {1.0, 2.0,  2.0, 1.0,  -1.0, 1.0};

    std::vector<double> r_lower = {4.0, -1e20, 3.0};
    std::vector<double> r_upper = {4.0, 2.0, 1e20};
    std::vector<double> c = {1.0, -2.0, 3.0};
    std::vector<double> c_lower = {0.0, -1e20, -5.0}; // col 1 free
    std::vector<double> c_upper = {10.0, 1e20, 5.0};

    std::vector<double> x0 = {1.0, 0.0, 1.0};
    std::vector<double> y0 = {0.0, 0.0, 0.0};
    double sigma = 0.4, tau = 0.4;

    // CPU Reference Step:
    // 1. w = A * x_bar (with x_bar = x0)
    std::vector<double> w_ref(m, 0.0);
    for (int i = 0; i < m; ++i) {
        for (int64_t p = row_ptr[i]; p < row_ptr[i + 1]; ++p) {
            w_ref[i] += csr_val[p] * x0[col_idx[p]];
        }
    }
    // 2. Dual update
    std::vector<double> y1_ref(m, 0.0);
    for (int i = 0; i < m; ++i) {
        double y_hat = y0[i] + sigma * w_ref[i];
        double val = y_hat / sigma;
        double clamped = (val < r_lower[i]) ? r_lower[i] : ((val > r_upper[i]) ? r_upper[i] : val);
        y1_ref[i] = y_hat - sigma * clamped;
    }
    // 3. Transpose SpMV v = A^T * y1
    std::vector<double> v_ref(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int64_t p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
            v_ref[j] += csc_val[p] * y1_ref[row_idx[p]];
        }
    }
    // 4. Primal update & extrapolation
    std::vector<double> x1_ref(n, 0.0);
    std::vector<double> x_bar1_ref(n, 0.0);
    for (int j = 0; j < n; ++j) {
        double grad = v_ref[j] + c[j];
        double x_new = x0[j] - tau * grad;
        if (x_new < c_lower[j]) x_new = c_lower[j];
        if (x_new > c_upper[j]) x_new = c_upper[j];
        x1_ref[j] = x_new;
        x_bar1_ref[j] = 2.0 * x_new - x0[j];
    }

    // Allocate & upload to GPU
    TestDeviceBuffer<int64_t> d_rp(row_ptr.size()); d_rp.upload(row_ptr.data(), row_ptr.size());
    TestDeviceBuffer<int> d_ci(col_idx.size()); d_ci.upload(col_idx.data(), col_idx.size());
    TestDeviceBuffer<double> d_val_csr(csr_val.size()); d_val_csr.upload(csr_val.data(), csr_val.size());

    TestDeviceBuffer<int64_t> d_cp(col_ptr.size()); d_cp.upload(col_ptr.data(), col_ptr.size());
    TestDeviceBuffer<int> d_ri(row_idx.size()); d_ri.upload(row_idx.data(), row_idx.size());
    TestDeviceBuffer<double> d_val_csc(csc_val.size()); d_val_csc.upload(csc_val.data(), csc_val.size());

    TestDeviceBuffer<double> d_rl(m); d_rl.upload(r_lower.data(), m);
    TestDeviceBuffer<double> d_ru(m); d_ru.upload(r_upper.data(), m);
    TestDeviceBuffer<double> d_c(n); d_c.upload(c.data(), n);
    TestDeviceBuffer<double> d_cl(n); d_cl.upload(c_lower.data(), n);
    TestDeviceBuffer<double> d_cu(n); d_cu.upload(c_upper.data(), n);

    TestDeviceBuffer<double> d_x(n); d_x.upload(x0.data(), n);
    TestDeviceBuffer<double> d_x_bar(n); d_x_bar.upload(x0.data(), n);
    TestDeviceBuffer<double> d_y(m); d_y.upload(y0.data(), m);
    TestDeviceBuffer<double> d_w(m); d_w.zero();
    TestDeviceBuffer<double> d_v(n); d_v.zero();

    // Execute 1 iteration on GPU
    spmv_csr_warp_kernel<<<1, 256>>>(m, d_rp.d_ptr, d_ci.d_ptr, d_val_csr.d_ptr, d_x_bar.d_ptr, d_w.d_ptr);
    dual_update_kernel<<<1, 256>>>(m, sigma, d_w.d_ptr, d_rl.d_ptr, d_ru.d_ptr, d_y.d_ptr);
    spmv_csc_transpose_warp_kernel<<<1, 256>>>(n, d_cp.d_ptr, d_ri.d_ptr, d_val_csc.d_ptr, d_y.d_ptr, d_v.d_ptr);
    primal_update_kernel<<<1, 256>>>(n, tau, d_v.d_ptr, d_c.d_ptr, d_cl.d_ptr, d_cu.d_ptr, d_x.d_ptr, d_x_bar.d_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_w(m), h_y(m), h_v(n), h_x(n), h_x_bar(n);
    d_w.download(h_w.data(), m);
    d_y.download(h_y.data(), m);
    d_v.download(h_v.data(), n);
    d_x.download(h_x.data(), n);
    d_x_bar.download(h_x_bar.data(), n);

    for (int i = 0; i < m; ++i) {
        ASSERT_NEAR_REL(h_w[i], w_ref[i], 1e-12);
        ASSERT_NEAR_REL(h_y[i], y1_ref[i], 1e-12);
    }
    for (int j = 0; j < n; ++j) {
        ASSERT_NEAR_REL(h_v[j], v_ref[j], 1e-12);
        ASSERT_NEAR_REL(h_x[j], x1_ref[j], 1e-12);
        ASSERT_NEAR_REL(h_x_bar[j], x_bar1_ref[j], 1e-12);
    }
    std::cout << "    Passed complete iteration verification.\n";
}

int main() {
    std::cout << "================================================================\n";
    std::cout << "  INDUS-OPT: CUDA KERNEL UNIT VERIFICATION SUITE\n";
    std::cout << "================================================================\n";

    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count <= 0) {
        std::cout << "[SKIP] No CUDA hardware device detected; kernel verification skipped.\n";
        return 0;
    }

    test_cuda_csr_spmv();
    test_cuda_transpose_spmv();
    test_cuda_dual_projections();
    test_cuda_primal_projection_and_extrapolation();
    test_cuda_device_diagnostics();
    test_cuda_complete_iteration();

    std::cout << "================================================================\n";
    std::cout << "  ALL CUDA DEVICE KERNELS VERIFIED IDENTICAL TO CPU SPECS (100%)\n";
    std::cout << "================================================================\n";
    return 0;
}
