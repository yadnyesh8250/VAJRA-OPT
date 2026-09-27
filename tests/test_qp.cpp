#include "indus/qp.hpp"
#include "indus/model.hpp"
#include "indus/options.hpp"
#include "indus/verifier.hpp"
#include "indus/io.hpp"
#include "indus/milp.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_passed = 0;
int g_failed = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            g_failed++; \
            return; \
        } \
    } while(0)

#define TEST_PASS(msg) \
    do { \
        std::cout << "  [PASS] " << msg << "\n"; \
        g_passed++; \
    } while(0)

indus::la::SparseMatrixCSC make_csc(int m, int n,
                                    const std::vector<int>& rows,
                                    const std::vector<int>& cols,
                                    const std::vector<double>& vals) {
    std::vector<indus::la::Triplet> triplets;
    triplets.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        triplets.push_back({rows[i], cols[i], vals[i]});
    }
    return indus::la::SparseMatrixCSC::from_triplets(m, n, triplets);
}

// -----------------------------------------------------------------------------
// Test 1: Unconstrained positive-definite QP with analytical solution
// min 0.5 * (2*x1^2 + 2*x2^2) - 4*x1 - 6*x2
// Analytical solution: x1 = 2, x2 = 3, obj = -13.0
// -----------------------------------------------------------------------------
void test_unconstrained_positive_definite() {
    std::cout << "\n[TEST 1] test_unconstrained_positive_definite\n";
    indus::Model model;
    model.name = "unconstrained_qp";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {-1e20, -1e20};
    model.col_upper = { 1e20,  1e20};
    model.c = {-4.0, -6.0};
    model.sense = indus::ObjSense::kMinimize;

    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(sol.algorithm_used == "convex_qp", "Algorithm must be convex_qp");
    TEST_ASSERT(sol.col_value.size() == 2, "col_value size must be 2");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-4, "x1 must be 2.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 3.0) < 1e-4, "x2 must be 3.0");
    TEST_ASSERT(std::abs(sol.objective_value - (-13.0)) < 1e-4, "Objective must be -13.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_ASSERT(vres.is_qp, "Must be classified as QP");
    TEST_ASSERT(vres.is_convex, "Must be convex");
    TEST_ASSERT(vres.max_stationarity_residual < 1e-4, "Stationarity residual must be small");

    TEST_PASS("test_unconstrained_positive_definite passed");
}

// -----------------------------------------------------------------------------
// Test 2: One-variable bounded QP
// min 0.5 * 2 * x0^2 - 6*x0  s.t.  0 <= x0 <= 2
// Unconstrained min is x0 = 3; bound activates at x0 = 2.
// obj = 0.5 * 2 * (4) - 6 * (2) = -8.0
// -----------------------------------------------------------------------------
void test_one_variable_bounded() {
    std::cout << "\n[TEST 2] test_one_variable_bounded\n";
    indus::Model model;
    model.name = "one_var_bounded_qp";
    model.num_cols = 1;
    model.num_rows = 0;
    model.col_lower = {0.0};
    model.col_upper = {2.0};
    model.c = {-6.0};
    model.sense = indus::ObjSense::kMinimize;

    model.Q = make_csc(1, 1, {0}, {0}, {2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-5, "x0 must be 2.0");
    TEST_ASSERT(std::abs(sol.objective_value - (-8.0)) < 1e-5, "Objective must be -8.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_one_variable_bounded passed");
}

// -----------------------------------------------------------------------------
// Test 3: Equality-constrained QP
// min 0.5 * (2*x1^2 + 2*x2^2)  s.t.  x1 + x2 = 2
// By symmetry, x1 = 1, x2 = 1, obj = 0.5 * (2*1 + 2*1) = 2.0
// -----------------------------------------------------------------------------
void test_equality_constrained() {
    std::cout << "\n[TEST 3] test_equality_constrained\n";
    indus::Model model;
    model.name = "eq_constrained_qp";
    model.num_cols = 2;
    model.num_rows = 1;
    model.col_lower = {-1e20, -1e20};
    model.col_upper = { 1e20,  1e20};
    model.c = {0.0, 0.0};
    model.row_lower = {2.0};
    model.row_upper = {2.0};
    model.sense = indus::ObjSense::kMinimize;

    model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 1.0) < 1e-4, "x1 must be 1.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 1.0) < 1e-4, "x2 must be 1.0");
    TEST_ASSERT(std::abs(sol.objective_value - 2.0) < 1e-4, "Objective must be 2.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_equality_constrained passed");
}

// -----------------------------------------------------------------------------
// Test 4: Inequality-constrained QP
// min 0.5 * (2*x1^2 + 2*x2^2)  s.t.  x1 + x2 >= 4, x1, x2 >= 0
// Minimum on boundary: x1 = 2, x2 = 2, obj = 0.5*(2*4 + 2*4) = 8.0
// -----------------------------------------------------------------------------
void test_inequality_constrained() {
    std::cout << "\n[TEST 4] test_inequality_constrained\n";
    indus::Model model;
    model.name = "ineq_constrained_qp";
    model.num_cols = 2;
    model.num_rows = 1;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1e20, 1e20};
    model.c = {0.0, 0.0};
    model.row_lower = {4.0};
    model.row_upper = {1e20};
    model.sense = indus::ObjSense::kMinimize;

    model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-4, "x1 must be 2.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 2.0) < 1e-4, "x2 must be 2.0");
    TEST_ASSERT(std::abs(sol.objective_value - 8.0) < 1e-4, "Objective must be 8.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_inequality_constrained passed");
}

// -----------------------------------------------------------------------------
// Test 5: Ranged-constraint QP
// min 0.5 * (2*x1^2 + 2*x2^2) - 8*x1  s.t.  1 <= x1 + x2 <= 2,  x >= 0
// Unconstrained min is x1 = 4, x2 = 0. Constraint x1 + x2 <= 2 forces x1 = 2, x2 = 0.
// obj = 0.5 * (2*4) - 8*(2) = 4 - 16 = -12.0
// -----------------------------------------------------------------------------
void test_ranged_constraint() {
    std::cout << "\n[TEST 5] test_ranged_constraint\n";
    indus::Model model;
    model.name = "ranged_qp";
    model.num_cols = 2;
    model.num_rows = 1;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {-8.0, 0.0};
    model.row_lower = {1.0};
    model.row_upper = {2.0};
    model.sense = indus::ObjSense::kMinimize;

    model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-4, "x1 must be 2.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 0.0) < 1e-4, "x2 must be 0.0");
    TEST_ASSERT(std::abs(sol.objective_value - (-12.0)) < 1e-4, "Objective must be -12.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_ranged_constraint passed");
}

// -----------------------------------------------------------------------------
// Test 6: Sparse multi-variable QP
// n = 5, Q diagonal Q_ii = 2*(i+1), c_i = -2*(i+1)
// s.t. sum(x_i) = 5.0, x_i >= 0
// -----------------------------------------------------------------------------
void test_sparse_multivariable() {
    std::cout << "\n[TEST 6] test_sparse_multivariable\n";
    int n = 5;
    indus::Model model;
    model.name = "sparse_multi_qp";
    model.num_cols = n;
    model.num_rows = 1;
    model.col_lower.assign(n, 0.0);
    model.col_upper.assign(n, 10.0);
    model.c.resize(n);
    for (int i = 0; i < n; ++i) {
        model.c[i] = -2.0 * (i + 1);
    }
    model.row_lower = {5.0};
    model.row_upper = {5.0};
    model.sense = indus::ObjSense::kMinimize;

    std::vector<int> a_rows(n, 0);
    std::vector<int> a_cols(n);
    std::vector<double> a_vals(n, 1.0);
    for (int i = 0; i < n; ++i) a_cols[i] = i;
    model.A = make_csc(1, n, a_rows, a_cols, a_vals);

    std::vector<int> q_rows(n), q_cols(n);
    std::vector<double> q_vals(n);
    for (int i = 0; i < n; ++i) {
        q_rows[i] = i;
        q_cols[i] = i;
        q_vals[i] = 2.0 * (i + 1);
    }
    model.Q = make_csc(n, n, q_rows, q_cols, q_vals);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    double sum_x = 0.0;
    for (int i = 0; i < n; ++i) sum_x += sol.col_value[i];
    TEST_ASSERT(std::abs(sum_x - 5.0) < 1e-4, "Sum of variables must equal 5.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_ASSERT(vres.max_stationarity_residual < 1e-4, "Stationarity must be small");
    TEST_PASS("test_sparse_multivariable passed");
}

// -----------------------------------------------------------------------------
// Test 7: Semidefinite QP with flat direction
// min 0.5 * (2*x1^2) + x2  s.t.  x1 + x2 >= 2, x1, x2 >= 0
// Q = diag(2, 0) is positive semidefinite.
// Solution: x1 = 0.5, x2 = 1.5, obj = 0.5*(2*0.25) + 1.5 = 1.75
// -----------------------------------------------------------------------------
void test_semidefinite_qp() {
    std::cout << "\n[TEST 7] test_semidefinite_qp\n";
    indus::Model model;
    model.name = "psd_semidefinite_qp";
    model.num_cols = 2;
    model.num_rows = 1;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {0.0, 1.0};
    model.row_lower = {2.0};
    model.row_upper = {1e20};
    model.sense = indus::ObjSense::kMinimize;

    model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    model.Q = make_csc(2, 2, {0}, {0}, {2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 0.5) < 1e-3, "x1 must be 0.5");
    TEST_ASSERT(std::abs(sol.col_value[1] - 1.5) < 1e-3, "x2 must be 1.5");
    TEST_ASSERT(std::abs(sol.objective_value - 1.75) < 1e-3, "Objective must be 1.75");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_semidefinite_qp passed");
}

// -----------------------------------------------------------------------------
// Test 8: QP with objective offset
// Same as Test 2 with offset = 42.5 -> obj = -8.0 + 42.5 = 34.5
// -----------------------------------------------------------------------------
void test_objective_offset() {
    std::cout << "\n[TEST 8] test_objective_offset\n";
    indus::Model model;
    model.name = "offset_qp";
    model.num_cols = 1;
    model.num_rows = 0;
    model.col_lower = {0.0};
    model.col_upper = {2.0};
    model.c = {-6.0};
    model.objective_offset = 42.5;
    model.sense = indus::ObjSense::kMinimize;

    model.Q = make_csc(1, 1, {0}, {0}, {2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-5, "x0 must be 2.0");
    TEST_ASSERT(std::abs(sol.objective_value - 34.5) < 1e-5, "Objective must be 34.5");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_objective_offset passed");
}

// -----------------------------------------------------------------------------
// Test 9: Minimization vs Maximization sign test
// Maximize 0.5 * (-2*x1^2 - 2*x2^2) + 4*x1 + 6*x2  s.t.  x >= 0
// -Q is positive definite -> concave objective -> convex maximization
// Optimum at x1 = 2, x2 = 3, obj = -0.5*(2*4 + 2*9) + 4*2 + 6*3 = -13 + 26 = 13.0
// -----------------------------------------------------------------------------
void test_maximization() {
    std::cout << "\n[TEST 9] test_maximization\n";
    indus::Model model;
    model.name = "max_concave_qp";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {4.0, 6.0};
    model.sense = indus::ObjSense::kMaximize;

    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {-2.0, -2.0});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-4, "x1 must be 2.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 3.0) < 1e-4, "x2 must be 3.0");
    TEST_ASSERT(std::abs(sol.objective_value - 13.0) < 1e-4, "Objective must be 13.0");

    indus::verifier::VerificationResult vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verification must pass");
    TEST_PASS("test_maximization passed");
}

// -----------------------------------------------------------------------------
// Test 10: Invalid asymmetric Q rejection
// Q_01 = 5.0, Q_10 = 0.0 -> asymmetric -> MODEL_ERROR
// -----------------------------------------------------------------------------
void test_asymmetric_q_rejection() {
    std::cout << "\n[TEST 10] test_asymmetric_q_rejection\n";
    indus::Model model;
    model.name = "asymmetric_qp";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1.0, 1.0};
    model.c = {0.0, 0.0};

    // Explicitly asymmetric Q: entry at (0, 1) is 5.0, entry at (1, 0) is 0.0
    model.Q = make_csc(2, 2, {0, 0, 1}, {0, 1, 1}, {2.0, 5.0, 2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kModelError, "Asymmetric Q must produce kModelError");
    TEST_PASS("test_asymmetric_q_rejection passed");
}

// -----------------------------------------------------------------------------
// Test 11: Nonconvex Q rejection
// min 0.5 * (2*x1^2 - 2*x2^2) -> indefinite Q -> MODEL_ERROR
// -----------------------------------------------------------------------------
void test_nonconvex_q_rejection() {
    std::cout << "\n[TEST 11] test_nonconvex_q_rejection\n";
    indus::Model model;
    model.name = "nonconvex_qp";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {0.0, 0.0};
    model.sense = indus::ObjSense::kMinimize;

    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, -2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kModelError, "Indefinite Q must produce kModelError");
    TEST_PASS("test_nonconvex_q_rejection passed");
}

// -----------------------------------------------------------------------------
// Test 12: MIQP rejection
// QP model with an integer variable -> UNSUPPORTED
// -----------------------------------------------------------------------------
void test_miqp_rejection() {
    std::cout << "\n[TEST 12] test_miqp_rejection\n";
    indus::Model model;
    model.name = "miqp_model";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {-1.0, -1.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kContinuous};
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kUnsupported, "MIQP must return kUnsupported");
    TEST_PASS("test_miqp_rejection passed");
}

// -----------------------------------------------------------------------------
// Test 13: Infeasible QP
// Constraints: x1 + x2 <= 1 and x1 + x2 >= 2
// -----------------------------------------------------------------------------
void test_infeasible_qp() {
    std::cout << "\n[TEST 13] test_infeasible_qp\n";
    indus::Model model;
    model.name = "infeasible_qp";
    model.num_cols = 2;
    model.num_rows = 2;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {0.0, 0.0};
    model.row_lower = {-1e20, 2.0};
    model.row_upper = {1.0, 1e20};

    model.A = make_csc(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1.0, 1.0, 1.0, 1.0});
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kInfeasible, "Infeasible QP must return kInfeasible");
    TEST_PASS("test_infeasible_qp passed");
}

// -----------------------------------------------------------------------------
// Test 14: Unbounded QP
// min -x1 with x1 >= 0 and no upper bound, Q has 0 curvature in x1
// -----------------------------------------------------------------------------
void test_unbounded_qp() {
    std::cout << "\n[TEST 14] test_unbounded_qp\n";
    indus::Model model;
    model.name = "unbounded_qp";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1e20, 10.0};
    model.c = {-10.0, 0.0};
    // Q is zero for x0, positive for x1
    model.Q = make_csc(2, 2, {1}, {1}, {2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kUnbounded || sol.status == indus::SolveStatus::kModelError,
                "Unbounded QP must return kUnbounded or kModelError");
    TEST_PASS("test_unbounded_qp passed");
}

// -----------------------------------------------------------------------------
// Test 15: Iteration-limit behavior
// -----------------------------------------------------------------------------
void test_iteration_limit() {
    std::cout << "\n[TEST 15] test_iteration_limit\n";
    indus::Model model;
    model.name = "iter_limit_qp";
    model.num_cols = 2;
    model.num_rows = 1;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {-4.0, -6.0};
    model.row_lower = {5.0};
    model.row_upper = {5.0};
    model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Options opts;
    opts.iteration_limit = 0;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kIterationLimit, "0 iterations must return kIterationLimit");
    TEST_PASS("test_iteration_limit passed");
}

// -----------------------------------------------------------------------------
// Test 16: Time-limit behavior
// -----------------------------------------------------------------------------
void test_time_limit() {
    std::cout << "\n[TEST 16] test_time_limit\n";
    indus::Model model;
    model.name = "time_limit_qp";
    model.num_cols = 10;
    model.num_rows = 5;
    model.col_lower.assign(10, 0.0);
    model.col_upper.assign(10, 10.0);
    model.c.assign(10, -1.0);
    model.row_lower.assign(5, 1.0);
    model.row_upper.assign(5, 2.0);

    std::vector<int> a_rows, a_cols;
    std::vector<double> a_vals;
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 10; ++c) {
            a_rows.push_back(r);
            a_cols.push_back(c);
            a_vals.push_back(1.0);
        }
    }
    model.A = make_csc(5, 10, a_rows, a_cols, a_vals);

    std::vector<int> q_rows(10), q_cols(10);
    std::vector<double> q_vals(10, 2.0);
    for (int i = 0; i < 10; ++i) { q_rows[i] = i; q_cols[i] = i; }
    model.Q = make_csc(10, 10, q_rows, q_cols, q_vals);

    indus::Options opts;
    opts.time_limit = 1e-9; // Exceedingly tight limit
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kTimeLimit || sol.status == indus::SolveStatus::kIterationLimit,
                "Exceeded time limit must report kTimeLimit");
    TEST_PASS("test_time_limit passed");
}

// -----------------------------------------------------------------------------
// Test 17: NaN and Infinite entry rejection
// -----------------------------------------------------------------------------
void test_nan_inf_rejection() {
    std::cout << "\n[TEST 17] test_nan_inf_rejection\n";
    indus::Model model;
    model.name = "nan_qp";
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1.0, 1.0};
    model.c = {0.0, 0.0};

    // NaN entry in Q
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {std::numeric_limits<double>::quiet_NaN(), 1.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kModelError, "NaN in Q must produce kModelError");

    // Inf entry in Q
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {std::numeric_limits<double>::infinity(), 1.0});
    sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kModelError, "Inf in Q must produce kModelError");

    TEST_PASS("test_nan_inf_rejection passed");
}

// -----------------------------------------------------------------------------
// Test 18: Objective, gradient, and stationarity verifier rejection tests
// -----------------------------------------------------------------------------
void test_verifier_rejection() {
    std::cout << "\n[TEST 18] test_verifier_rejection\n";
    indus::Model model;
    model.name = "verifier_test_qp";
    model.num_cols = 2;
    model.num_rows = 1;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.c = {0.0, 0.0};
    model.row_lower = {2.0};
    model.row_upper = {2.0};
    model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve must succeed");

    indus::verifier::VerificationResult vres_ok = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres_ok.passed, "True solution must pass verifier");

    // 1. Wrong objective value
    indus::Solution sol_wrong_obj = sol;
    sol_wrong_obj.objective_value += 100.0;
    indus::verifier::VerificationResult vres_wrong_obj = indus::verifier::verify_solution(model, sol_wrong_obj);
    TEST_ASSERT(!vres_wrong_obj.passed, "Wrong objective value must fail verification");

    // 2. Violated variable bound
    indus::Solution sol_violated_bound = sol;
    sol_violated_bound.col_value[0] = -1.0;
    indus::verifier::VerificationResult vres_violated_bound = indus::verifier::verify_solution(model, sol_violated_bound);
    TEST_ASSERT(!vres_violated_bound.passed, "Violated bound must fail verification");

    // 3. Violated row constraint
    indus::Solution sol_violated_row = sol;
    sol_violated_row.col_value[0] += 5.0; // x1 + x2 = 7 != 2
    indus::verifier::VerificationResult vres_violated_row = indus::verifier::verify_solution(model, sol_violated_row);
    TEST_ASSERT(!vres_violated_row.passed, "Violated row constraint must fail verification");

    // 4. Large stationarity violation (arbitrary point with bad duals)
    indus::Solution sol_non_stat = sol;
    sol_non_stat.col_value = {0.0, 2.0}; // feasible for x1+x2=2, but not stationary/optimal
    sol_non_stat.row_dual = {0.0};
    sol_non_stat.col_dual = {0.0, 0.0};
    sol_non_stat.recompute_quality(model);
    indus::verifier::VerificationResult vres_non_stat = indus::verifier::verify_solution(model, sol_non_stat);
    TEST_ASSERT(!vres_non_stat.passed, "Non-stationary point must fail verification");

    TEST_PASS("test_verifier_rejection passed");
}

// -----------------------------------------------------------------------------
// Test 19: Repeated solve determinism
// -----------------------------------------------------------------------------
void test_repeated_solve_determinism() {
    std::cout << "\n[TEST 19] test_repeated_solve_determinism\n";
    indus::Model model;
    model.name = "determinism_qp";
    model.num_cols = 3;
    model.num_rows = 1;
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {5.0, 5.0, 5.0};
    model.c = {-1.0, -2.0, -3.0};
    model.row_lower = {3.0};
    model.row_upper = {3.0};
    model.A = make_csc(1, 3, {0, 0, 0}, {0, 1, 2}, {1.0, 1.0, 1.0});
    model.Q = make_csc(3, 3, {0, 1, 2}, {0, 1, 2}, {2.0, 4.0, 6.0});

    indus::Solution sol0 = indus::solve(model);
    TEST_ASSERT(sol0.status == indus::SolveStatus::kOptimal, "Run 0 must be optimal");

    for (int rep = 1; rep <= 5; ++rep) {
        indus::Solution sol_rep = indus::solve(model);
        TEST_ASSERT(sol_rep.status == sol0.status, "Status must match");
        TEST_ASSERT(sol_rep.iterations == sol0.iterations, "Iteration counts must be identical");
        TEST_ASSERT(sol_rep.objective_value == sol0.objective_value, "Objective values must be bitwise identical");
        for (size_t j = 0; j < sol0.col_value.size(); ++j) {
            TEST_ASSERT(sol_rep.col_value[j] == sol0.col_value[j], "col_value must be bitwise identical");
        }
    }
    TEST_PASS("test_repeated_solve_determinism passed");
}

// -----------------------------------------------------------------------------
// Test 20: LP and MILP regression protection
// -----------------------------------------------------------------------------
void test_lp_milp_regression() {
    std::cout << "\n[TEST 20] test_lp_milp_regression\n";
    
    // 1. Continuous LP
    indus::Model lp_model;
    lp_model.name = "regression_lp";
    lp_model.num_cols = 2;
    lp_model.num_rows = 1;
    lp_model.col_lower = {0.0, 0.0};
    lp_model.col_upper = {10.0, 10.0};
    lp_model.c = {1.0, 2.0};
    lp_model.row_lower = {-1e20};
    lp_model.row_upper = {4.0};
    lp_model.sense = indus::ObjSense::kMaximize;
    lp_model.A = make_csc(1, 2, {0, 0}, {0, 1}, {1.0, 1.0});

    indus::Solution lp_sol = indus::solve(lp_model);
    TEST_ASSERT(lp_sol.status == indus::SolveStatus::kOptimal, "LP status must be kOptimal");
    TEST_ASSERT(lp_sol.algorithm_used == "auto" || lp_sol.algorithm_used == "simplex" || lp_sol.algorithm_used == "dual_simplex", "LP algorithm must be simplex or auto");
    TEST_ASSERT(std::abs(lp_sol.objective_value - 8.0) < 1e-4, "LP obj must be 8.0");

    // 2. Binary MILP
    indus::Model milp_model;
    milp_model.name = "regression_milp";
    milp_model.num_cols = 2;
    milp_model.num_rows = 1;
    milp_model.col_lower = {0.0, 0.0};
    milp_model.col_upper = {1.0, 1.0};
    milp_model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};
    milp_model.c = {3.0, 4.0};
    milp_model.row_lower = {-1e20};
    milp_model.row_upper = {2.5}; // 2*x1 + 3*x2 <= 2.5: only x1=1 (val 3) or x2=0 feasible
    milp_model.sense = indus::ObjSense::kMaximize;
    milp_model.A = make_csc(1, 2, {0, 0}, {0, 1}, {2.0, 3.0});

    indus::Solution milp_sol = indus::solve(milp_model);
    TEST_ASSERT(milp_sol.status == indus::SolveStatus::kOptimal, "MILP status must be kOptimal");
    TEST_ASSERT(milp_sol.algorithm_used == "milp_branch_and_bound" || milp_sol.algorithm_used == "branch_and_bound", "MILP algorithm must be branch_and_bound");
    TEST_ASSERT(std::abs(milp_sol.objective_value - 3.0) < 1e-4, "MILP obj must be 3.0");

    TEST_PASS("test_lp_milp_regression passed");
}

// -----------------------------------------------------------------------------
// Test 21: Serialization of Unnamed and Partially Named QP Models
// -----------------------------------------------------------------------------
void test_qp_serialization() {
    std::cout << "\n[TEST 21] test_qp_serialization\n";
    indus::Model model;
    model.name = ""; // Unnamed
    model.num_cols = 2;
    model.num_rows = 0;
    model.col_lower = {0.0, 0.0};
    model.col_upper = {5.0, 5.0};
    model.c = {-2.0, -4.0};
    model.Q = make_csc(2, 2, {0, 1}, {0, 1}, {2.0, 2.0});

    indus::Solution sol = indus::solve(model);
    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve must be optimal");

    std::string tmp_sol = "/tmp/test_unnamed_qp.sol";
    std::string tmp_json = "/tmp/test_unnamed_qp.json";

    indus::io::write_solution(sol, model, tmp_sol);
    TEST_ASSERT(std::filesystem::exists(tmp_sol), "Writing .sol must succeed");

    indus::io::write_json(sol, model, tmp_json);
    TEST_ASSERT(std::filesystem::exists(tmp_json), "Writing .json must succeed");

    std::ifstream json_file(tmp_json);
    TEST_ASSERT(json_file.is_open(), "JSON file must open");
    std::string content((std::istreambuf_iterator<char>(json_file)), std::istreambuf_iterator<char>());
    TEST_ASSERT(content.find("\"problem_class\": \"QP\"") != std::string::npos, "JSON must contain problem_class QP");
    TEST_ASSERT(content.find("\"algorithm\": \"convex_qp\"") != std::string::npos, "JSON must contain algorithm convex_qp");
    TEST_ASSERT(content.find("\"max_stationarity_residual\"") != std::string::npos, "JSON must contain max_stationarity_residual");

    std::filesystem::remove(tmp_sol);
    std::filesystem::remove(tmp_json);

    TEST_PASS("test_qp_serialization passed");
}

} // namespace

int main() {
    std::cout << "========================================================\n";
    std::cout << "VAJRA-OPT Native Convex QP Test Suite (Phase 8)\n";
    std::cout << "========================================================\n";

    test_unconstrained_positive_definite();
    test_one_variable_bounded();
    test_equality_constrained();
    test_inequality_constrained();
    test_ranged_constraint();
    test_sparse_multivariable();
    test_semidefinite_qp();
    test_objective_offset();
    test_maximization();
    test_asymmetric_q_rejection();
    test_nonconvex_q_rejection();
    test_miqp_rejection();
    test_infeasible_qp();
    test_unbounded_qp();
    test_iteration_limit();
    test_time_limit();
    test_nan_inf_rejection();
    test_verifier_rejection();
    test_repeated_solve_determinism();
    test_lp_milp_regression();
    test_qp_serialization();

    std::cout << "\n========================================================\n";
    std::cout << "QP Test Results: " << g_passed << " passed, " << g_failed << " failed.\n";
    std::cout << "========================================================\n";

    return (g_failed == 0) ? 0 : 1;
}
