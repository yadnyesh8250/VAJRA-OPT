#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip>
#include <string>

#include "indus/types.hpp"
#include "indus/tolerances.hpp"
#include "indus/version.hpp"
#include "indus/sparse.hpp"
#include "src/linalg/lu.hpp"
#include "src/linalg/ldl.hpp"
#include "src/linalg/scaling.hpp"

using namespace indus;
using namespace indus::la;

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_tests_failed++; \
            return; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        std::cout << "[RUNNING] " << #fn << "... "; \
        const int before_failed = g_tests_failed; \
        fn(); \
        if (g_tests_failed == before_failed) { \
            std::cout << "PASS\n"; \
            g_tests_passed++; \
        } \
    } while (0)

// Helper: infinity norm of difference between two vectors
double vec_diff_inf(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

void test_types_and_tolerances() {
    TEST_ASSERT(claims_a_point(SolveStatus::kOptimal), "Optimal claims a point");
    TEST_ASSERT(!claims_a_point(SolveStatus::kInfeasible), "Infeasible does not claim point");
    TEST_ASSERT(std::string(to_string(SolveStatus::kOptimal)) == "OPTIMAL", "SolveStatus to string");
    TEST_ASSERT(std::string(to_string(BasisStatus::kBasic)) == "BASIC", "BasisStatus to string");
    TEST_ASSERT(tol::kMarkowitzThreshold == 0.01, "Markowitz threshold must equal 0.01");
    TEST_ASSERT(tol::kPrimalFeasibility == 1e-7, "Primal feasibility tolerance must equal 1e-7");
}

void test_sparse_matrix_basics() {
    // 3 x 3 matrix:
    // [ 10.0   0.0   2.0 ]
    // [  3.0   9.0   0.0 ]
    // [  0.0   7.0   8.0 ]
    std::vector<Triplet> triplets = {
        {0, 0, 10.0}, {0, 2, 2.0},
        {1, 0, 3.0}, {1, 1, 9.0},
        {2, 1, 7.0}, {2, 2, 8.0},
        {0, 0, 5.0} // Duplicate to test summation -> (0, 0) becomes 15.0
    };

    SparseMatrixCSC csc = SparseMatrixCSC::from_triplets(3, 3, triplets, true);
    TEST_ASSERT(csc.m == 3 && csc.n == 3, "Dimensions match");
    TEST_ASSERT(csc.coeff(0, 0) == 15.0, "Duplicate summation for (0, 0)");
    TEST_ASSERT(csc.coeff(0, 2) == 2.0, "Coeff (0, 2)");
    TEST_ASSERT(csc.coeff(1, 0) == 3.0, "Coeff (1, 0)");
    TEST_ASSERT(csc.coeff(1, 1) == 9.0, "Coeff (1, 1)");
    TEST_ASSERT(csc.coeff(2, 1) == 7.0, "Coeff (2, 1)");
    TEST_ASSERT(csc.coeff(2, 2) == 8.0, "Coeff (2, 2)");
    TEST_ASSERT(csc.coeff(0, 1) == 0.0, "Zero entry query");

    // SpMV forward
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y(3, 0.0);
    csc.multiply(x, y);
    // Row 0: 15*1 + 0*2 + 2*3 = 21
    // Row 1: 3*1 + 9*2 + 0*3 = 21
    // Row 2: 0*1 + 7*2 + 8*3 = 38
    TEST_ASSERT(std::abs(y[0] - 21.0) < 1e-12, "SpMV y[0] == 21");
    TEST_ASSERT(std::abs(y[1] - 21.0) < 1e-12, "SpMV y[1] == 21");
    TEST_ASSERT(std::abs(y[2] - 38.0) < 1e-12, "SpMV y[2] == 38");

    // Transpose into CSR
    SparseMatrixCSR csr;
    csc.transpose_into(csr);
    TEST_ASSERT(csr.m == 3 && csr.n == 3, "CSR dimensions");
    TEST_ASSERT(csr.coeff(2, 0) == 2.0, "CSR transpose coeff (2, 0) == CSC(0, 2)");
    TEST_ASSERT(csr.coeff(0, 2) == 0.0, "CSR transpose coeff (0, 2) == CSC(2, 0) == 0");

    // Norms
    // Row inf: row 0: 17, row 1: 12, row 2: 15 -> max is 17
    TEST_ASSERT(std::abs(csc.norm_inf() - 17.0) < 1e-12, "Norm inf == 17");
    // Col 1: col 0: 18, col 1: 16, col 2: 10 -> max is 18
    TEST_ASSERT(std::abs(csc.norm_1() - 18.0) < 1e-12, "Norm 1 == 18");
}

void test_sparse_lu_factorization() {
    // 5 x 5 well-conditioned sparse basis matrix
    // [ 4  1  0  0  0 ]
    // [ 1  5  2  0  0 ]
    // [ 0  2  6  1  0 ]
    // [ 0  0  1  7  3 ]
    // [ 0  0  0  3  8 ]
    std::vector<Triplet> triplets = {
        {0, 0, 4.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 5.0}, {1, 2, 2.0},
        {2, 1, 2.0}, {2, 2, 6.0}, {2, 3, 1.0},
        {3, 2, 1.0}, {3, 3, 7.0}, {3, 4, 3.0},
        {4, 3, 3.0}, {4, 4, 8.0}
    };
    SparseMatrixCSC B = SparseMatrixCSC::from_triplets(5, 5, triplets);

    SparseLU lu;
    FactorizationStatus status = lu.factorize(B, tol::kMarkowitzThreshold);
    TEST_ASSERT(status == FactorizationStatus::kSuccess, "LU factorize status == Success");
    TEST_ASSERT(!lu.is_repaired(), "Basis is not singular, no repair needed");

    // Test FTRAN: B * x = b
    std::vector<double> x_true = {1.5, -2.0, 3.5, 0.5, -1.0};
    std::vector<double> b(5, 0.0);
    B.multiply(x_true, b);

    std::vector<double> x_sol = b;
    lu.ftran(x_sol);

    double residual_f = vec_diff_inf(x_sol, x_true);
    TEST_ASSERT(residual_f < 1e-12, "FTRAN solve residual < 1e-12");

    // Test BTRAN: Bᵀ * y = c
    std::vector<double> y_true = {2.0, -1.0, 4.0, 1.0, -3.0};
    std::vector<double> c(5, 0.0);
    B.multiply_transpose(y_true, c);

    std::vector<double> y_sol = c;
    lu.btran(y_sol);

    double residual_b = vec_diff_inf(y_sol, y_true);
    TEST_ASSERT(residual_b < 1e-12, "BTRAN solve residual < 1e-12");
}

void test_sparse_lu_singularity_repair() {
    // 5 x 5 singular matrix (Row 4 is identical to Row 3)
    std::vector<Triplet> triplets = {
        {0, 0, 2.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 3.0},
        {2, 2, 4.0},
        {3, 3, 5.0}, {3, 4, 2.0},
        {4, 3, 5.0}, {4, 4, 2.0} // Dependent row!
    };
    SparseMatrixCSC B = SparseMatrixCSC::from_triplets(5, 5, triplets);

    SparseLU lu;
    FactorizationStatus status = lu.factorize(B, tol::kMarkowitzThreshold);
    TEST_ASSERT(status == FactorizationStatus::kSingularRepaired, "Singular matrix detected and repaired");
    TEST_ASSERT(lu.is_repaired(), "is_repaired() flag is set");
    TEST_ASSERT(!lu.repaired_rows().empty(), "Repaired rows reported");

    // FTRAN should still solve without throwing or NaN
    std::vector<double> b = {1.0, 2.0, 3.0, 4.0, 5.0};
    lu.ftran(b);
    for (double val : b) {
        TEST_ASSERT(!std::isnan(val) && !std::isinf(val), "FTRAN on repaired basis yields finite numbers");
    }
}

void test_sparse_lu_pfi_updates() {
    // Base matrix
    std::vector<Triplet> triplets = {
        {0, 0, 3.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 4.0}
    };
    SparseMatrixCSC B = SparseMatrixCSC::from_triplets(2, 2, triplets);

    SparseLU lu;
    lu.factorize(B);

    // Replace column 1 with entering column a_enter = [2.0, 5.0]ᵀ
    // Leaving row is 1
    std::vector<double> a_enter = {2.0, 5.0};
    std::vector<double> alpha = a_enter;
    lu.ftran(alpha); // alpha = B⁻¹ a_enter

    bool pfi_ok = lu.update_basis_pfi(1, alpha);
    TEST_ASSERT(pfi_ok, "PFI basis update succeeded");
    TEST_ASSERT(lu.num_etas() == 1, "One eta recorded");

    // The new basis B_new has col 0 = [3, 1]ᵀ, col 1 = [2, 5]ᵀ
    std::vector<Triplet> new_triplets = {
        {0, 0, 3.0}, {0, 1, 2.0},
        {1, 0, 1.0}, {1, 1, 5.0}
    };
    SparseMatrixCSC B_new = SparseMatrixCSC::from_triplets(2, 2, new_triplets);

    std::vector<double> x_true = {4.0, -1.0};
    std::vector<double> b(2, 0.0);
    B_new.multiply(x_true, b);

    // Solve using PFI updated LU
    std::vector<double> x_sol = b;
    lu.ftran(x_sol);

    double diff = vec_diff_inf(x_sol, x_true);
    TEST_ASSERT(diff < 1e-12, "PFI FTRAN matches direct solve within 1e-12");

    // Also test BTRAN with PFI
    std::vector<double> y_true = {1.5, 3.0};
    std::vector<double> c(2, 0.0);
    B_new.multiply_transpose(y_true, c);

    std::vector<double> y_sol = c;
    lu.btran(y_sol);

    double diff_b = vec_diff_inf(y_sol, y_true);
    TEST_ASSERT(diff_b < 1e-12, "PFI BTRAN matches direct transpose solve within 1e-12");
}

void test_sparse_ldl_and_psd_certification() {
    // 1. Positive Definite Symmetric Matrix
    // [ 4  1  0 ]
    // [ 1  5  2 ]
    // [ 0  2  6 ]
    std::vector<Triplet> spd_triplets = {
        {0, 0, 4.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 5.0}, {1, 2, 2.0},
        {2, 1, 2.0}, {2, 2, 6.0}
    };
    SparseMatrixCSC A_spd = SparseMatrixCSC::from_triplets(3, 3, spd_triplets);

    SparseLDL ldl;
    bool ok = ldl.factorize(A_spd, true);
    TEST_ASSERT(ok, "LDL factorize succeeded");
    TEST_ASSERT(ldl.is_positive_definite(), "A_spd certified as Positive Definite");
    TEST_ASSERT(ldl.inertia().num_positive == 3, "3 positive pivots");
    TEST_ASSERT(ldl.inertia().num_negative == 0, "0 negative pivots");

    // Solve A_spd * x = b
    std::vector<double> x_true = {2.0, -1.0, 3.0};
    std::vector<double> b(3, 0.0);
    A_spd.multiply(x_true, b);

    std::vector<double> x_sol = b;
    ldl.solve(x_sol);
    TEST_ASSERT(vec_diff_inf(x_sol, x_true) < 1e-12, "LDL solve residual < 1e-12");

    // 2. Indefinite Matrix (Negative Curvature / Non-Convex QP Hessian)
    // [  1   3 ]
    // [  3   1 ]
    // det = 1 - 9 = -8 < 0 -> eigenvalues are 4 and -2!
    std::vector<Triplet> indef_triplets = {
        {0, 0, 1.0}, {0, 1, 3.0},
        {1, 0, 3.0}, {1, 1, 1.0}
    };
    SparseMatrixCSC A_indef = SparseMatrixCSC::from_triplets(2, 2, indef_triplets);

    SparseLDL ldl_indef;
    ldl_indef.factorize(A_indef, false);
    TEST_ASSERT(!ldl_indef.is_positive_definite(), "Indefinite matrix is NOT positive definite");
    TEST_ASSERT(!ldl_indef.is_positive_semidefinite(), "Indefinite matrix is NOT positive semi-definite");
    TEST_ASSERT(ldl_indef.inertia().num_negative == 1, "Exactly 1 negative pivot detected");

    std::vector<double> v;
    bool has_witness = ldl_indef.get_negative_curvature_witness(v);
    TEST_ASSERT(has_witness, "Negative curvature witness extracted");

    // Verify vᵀ A v < 0
    std::vector<double> Av(2, 0.0);
    A_indef.multiply(v, Av);
    double quad_form = v[0] * Av[0] + v[1] * Av[1];
    TEST_ASSERT(quad_form < -1e-6, "Witness certifies vᵀ A v < 0 strictly");
}

void test_ruiz_and_pock_chambolle_scaling() {
    // Create an ill-conditioned matrix with wide dynamic range
    // Row 0 has scale ~1e-4, Row 1 has scale ~1e4
    std::vector<Triplet> triplets = {
        {0, 0, 1e-4}, {0, 1, 2e-4},
        {1, 0, 3e4},  {1, 1, 5e4}
    };
    SparseMatrixCSC A = SparseMatrixCSC::from_triplets(2, 2, triplets);
    std::vector<double> c = {1.0, 1.0};
    std::vector<double> r_l = {0.0, 0.0};
    std::vector<double> r_u = {1.0, 1e5};
    std::vector<double> c_l = {0.0, 0.0};
    std::vector<double> c_u = {10.0, 10.0};

    ScalingFactors sf = RuizScaler::compute_and_scale(A, c, r_l, r_u, c_l, c_u, 10, 0.05);
    TEST_ASSERT(sf.is_scaled, "Ruiz scaling completed");

    // Verify that after Ruiz scaling, row inf-norms are close to 1.0
    double row0_norm = std::max(std::abs(A.coeff(0, 0)), std::abs(A.coeff(0, 1)));
    double row1_norm = std::max(std::abs(A.coeff(1, 0)), std::abs(A.coeff(1, 1)));
    TEST_ASSERT(std::abs(row0_norm - 1.0) < 0.25, "Row 0 equilibrated near 1.0");
    TEST_ASSERT(std::abs(row1_norm - 1.0) < 0.25, "Row 1 equilibrated near 1.0");

    // Test unscaling roundtrip
    std::vector<double> x_scaled = {1.0, 1.0};
    RuizScaler::unscale_primal(sf, x_scaled);
    // x = C * x_scaled -> should not be 1.0
    TEST_ASSERT(x_scaled[0] != 1.0, "Primal unscaling modifies vector correctly");

    // Test Pock-Chambolle Preconditioner
    std::vector<double> d_r, d_c;
    PockChambollePreconditioner::compute(A, d_r, d_c);
    TEST_ASSERT(d_r.size() == 2 && d_c.size() == 2, "Preconditioner vector sizes match");
    TEST_ASSERT(d_r[0] > 0.0 && d_r[1] > 0.0, "Row preconditioners positive");
    TEST_ASSERT(d_c[0] > 0.0 && d_c[1] > 0.0, "Col preconditioners positive");
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  INDUS-OPT / SIDDHANTA: Phase 1 Linear Algebra Core Test\n";
    std::cout << "  Sovereign Clean-Room Mathematical Foundation Audit\n";
    std::cout << "=========================================================\n";

    RUN_TEST(test_types_and_tolerances);
    RUN_TEST(test_sparse_matrix_basics);
    RUN_TEST(test_sparse_lu_factorization);
    RUN_TEST(test_sparse_lu_singularity_repair);
    RUN_TEST(test_sparse_lu_pfi_updates);
    RUN_TEST(test_sparse_ldl_and_psd_certification);
    RUN_TEST(test_ruiz_and_pock_chambolle_scaling);

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Total Tests Passed: " << g_tests_passed << "\n";
    std::cout << "Total Tests Failed: " << g_tests_failed << "\n";
    std::cout << "---------------------------------------------------------\n";

    if (g_tests_failed == 0) {
        std::cout << ">>> PHASE 1 GATE CHECK: ALL MATHEMATICAL TESTS PASSED! <<<\n";
        return 0;
    } else {
        std::cout << ">>> PHASE 1 GATE CHECK: FAILED <<<\n";
        return 1;
    }
}
