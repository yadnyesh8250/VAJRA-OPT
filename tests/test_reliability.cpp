#include "indus/model.hpp"
#include "indus/options.hpp"
#include "indus/gpu.hpp"
#include "indus/tolerances.hpp"
#include "indus/verifier.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>

namespace {

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define ASSERT_NEAR(val, exp, tol) \
    do { \
        if (std::abs((val) - (exp)) > (tol)) { \
            std::cerr << "Assertion FAILED: " #val " (" << (val) << ") != " #exp " (" << (exp) \
                      << ") [diff=" << std::abs((val) - (exp)) << " > " << (tol) << "] at " \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

// 1. Empty Model (0 variables, 0 constraints)
void test_empty_model() {
    std::cout << "[Reliability 1] Empty model (0 variables, 0 constraints)...\n";
    indus::Model model;
    model.name = "empty_model";
    model.num_rows = 0;
    model.num_cols = 0;
    model.objective_offset = 42.5;

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal);
    ASSERT_NEAR(sol.objective_value, 42.5, 1e-9);
    ASSERT_TRUE(sol.col_value.empty());
    std::cout << "  Passed: Empty model solved trivially to objective offset.\n";
}

// 2. Zero Variables with Constraints
void test_zero_variables_with_constraints() {
    std::cout << "[Reliability 2] Zero variables with constraints...\n";
    // Feasible: 0 in [-5, 10]
    {
        indus::Model model;
        model.name = "zero_vars_feasible";
        model.num_rows = 1;
        model.num_cols = 0;
        model.row_lower = {-5.0};
        model.row_upper = {10.0};
        model.row_names = {"c1"};
        model.objective_offset = 10.0;
        model.A.set_from_triplets(1, 0, {});

        indus::Options opts;
        indus::Solution sol = indus::solve(model, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal);
        ASSERT_NEAR(sol.objective_value, 10.0, 1e-9);
    }
    // Infeasible: 0 not in [5, 10]
    {
        indus::Model model;
        model.name = "zero_vars_infeasible";
        model.num_rows = 1;
        model.num_cols = 0;
        model.row_lower = {5.0};
        model.row_upper = {10.0};
        model.row_names = {"c1"};
        model.A.set_from_triplets(1, 0, {});

        indus::Options opts;
        indus::Solution sol = indus::solve(model, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kInfeasible);
    }
    std::cout << "  Passed: Zero-variable feasible and infeasible constraints handled correctly.\n";
}

// 3. Zero Constraints with Box-Bounded Variables
void test_zero_constraints_with_variables() {
    std::cout << "[Reliability 3] Zero constraints with box-bounded variables...\n";
    indus::Model model;
    model.name = "zero_constraints";
    model.num_rows = 0;
    model.num_cols = 3;
    model.sense = indus::ObjSense::kMinimize;
    model.c = {2.0, -4.0, 0.0};
    model.col_lower = {-2.0, 0.0, -10.0};
    model.col_upper = {5.0, 10.0, 10.0};
    model.col_names = {"x1", "x2", "x3"};
    model.objective_offset = 5.0;
    model.A.set_from_triplets(0, 3, {});

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal);
    // Min 2*x1 - 4*x2 + 0*x3:
    // x1 should be at lower bound (-2.0) -> cost: -4.0
    // x2 should be at upper bound (10.0) -> cost: -40.0
    // x3 can be 0.0 -> cost: 0.0
    // Total obj = 5.0 - 4.0 - 40.0 = -39.0
    ASSERT_NEAR(sol.col_value[0], -2.0, 1e-9);
    ASSERT_NEAR(sol.col_value[1], 10.0, 1e-9);
    ASSERT_NEAR(sol.objective_value, -39.0, 1e-9);
    std::cout << "  Passed: Unconstrained box-bounded LP solved to exact optimum.\n";
}

// 4. Invalid Bounds and Model Validation
void test_invalid_bounds_and_validation() {
    std::cout << "[Reliability 4] Input validation and malformed bounds detection...\n";

    // Crossed col bounds: lower > upper
    {
        indus::Model m;
        m.num_rows = 1;
        m.num_cols = 1;
        m.c = {1.0};
        m.col_lower = {10.0};
        m.col_upper = {5.0}; // lower > upper!
        m.row_lower = {0.0};
        m.row_upper = {10.0};
        m.A.m = 1; m.A.n = 1;
        m.A.col_ptr = {0, 1};
        m.A.row_idx = {0};
        m.A.values = {1.0};

        bool caught = false;
        try {
            m.validate();
        } catch (const std::invalid_argument& e) {
            caught = true;
        }
        ASSERT_TRUE(caught);
    }

    // Crossed row bounds: lower > upper
    {
        indus::Model m;
        m.num_rows = 1;
        m.num_cols = 1;
        m.c = {1.0};
        m.col_lower = {0.0};
        m.col_upper = {10.0};
        m.row_lower = {15.0};
        m.row_upper = {5.0}; // lower > upper!
        m.A.m = 1; m.A.n = 1;
        m.A.col_ptr = {0, 1};
        m.A.row_idx = {0};
        m.A.values = {1.0};

        bool caught = false;
        try {
            m.validate();
        } catch (const std::invalid_argument& e) {
            caught = true;
        }
        ASSERT_TRUE(caught);
    }

    // Dimension mismatch between A and num_rows
    {
        indus::Model m;
        m.num_rows = 2; // claimed 2, but A.m = 1
        m.num_cols = 1;
        m.c = {1.0};
        m.col_lower = {0.0};
        m.col_upper = {10.0};
        m.row_lower = {0.0, 0.0};
        m.row_upper = {10.0, 10.0};
        m.A.m = 1; m.A.n = 1;
        m.A.col_ptr = {0, 1};
        m.A.row_idx = {0};
        m.A.values = {1.0};

        bool caught = false;
        try {
            m.validate();
        } catch (const std::invalid_argument& e) {
            caught = true;
        }
        ASSERT_TRUE(caught);
    }

    std::cout << "  Passed: All invalid bounds and dimension mismatches safely rejected.\n";
}

// 5. Infeasible and Unbounded Models
void test_infeasible_and_unbounded() {
    std::cout << "[Reliability 5] Infeasible and unbounded LP status differentiation...\n";

    // Infeasible LP: x1 + x2 <= 2 and x1 + x2 >= 5 with x1, x2 >= 0
    {
        indus::Model m;
        m.num_rows = 2;
        m.num_cols = 2;
        m.c = {1.0, 1.0};
        m.col_lower = {0.0, 0.0};
        m.col_upper = {1e20, 1e20};
        m.row_lower = {-1e20, 5.0};
        m.row_upper = {2.0, 1e20};
        m.A.set_from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}});

        indus::Options opts;
        indus::Solution sol = indus::solve(m, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kInfeasible);
    }

    // Unbounded LP: Maximize x1 s.t. x1 - x2 <= 0, x1, x2 >= 0
    {
        indus::Model m;
        m.sense = indus::ObjSense::kMaximize;
        m.num_rows = 1;
        m.num_cols = 2;
        m.c = {1.0, 0.0};
        m.col_lower = {0.0, 0.0};
        m.col_upper = {1e20, 1e20};
        m.row_lower = {-1e20};
        m.row_upper = {0.0};
        m.A.set_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, -1.0}});

        indus::Options opts;
        indus::Solution sol = indus::solve(m, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kUnbounded);
    }

    std::cout << "  Passed: Infeasible and unbounded models correctly detected and distinguished.\n";
}

// 6. Duplicate / Redundant Constraints
void test_duplicate_constraints() {
    std::cout << "[Reliability 6] Duplicate and identical redundant constraints...\n";
    // Maximize 2 x1 + 3 x2 s.t. x1 + x2 <= 4, x1 + x2 <= 4 (duplicate), 2 x1 + x2 <= 6, x >= 0
    indus::Model m;
    m.sense = indus::ObjSense::kMaximize;
    m.num_rows = 3;
    m.num_cols = 2;
    m.c = {2.0, 3.0};
    m.col_lower = {0.0, 0.0};
    m.col_upper = {1e20, 1e20};
    m.row_lower = {-1e20, -1e20, -1e20};
    m.row_upper = {4.0, 4.0, 6.0}; // row 0 and row 1 are identical duplicates

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 1.0},
        {2, 0, 2.0}, {2, 1, 1.0}
    };
    m.A.set_from_triplets(3, 2, tri);

    indus::Options opts;
    indus::Solution sol = indus::solve(m, opts);

    ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal);
    // Optimum at x1=0, x2=4 -> obj = 12.0 (or x1=2, x2=2 -> obj = 10, max is 12)
    ASSERT_NEAR(sol.objective_value, 12.0, 1e-6);
    std::cout << "  Passed: Duplicate constraints handled cleanly without numerical singularity.\n";
}

// 7. Extreme and Ill-Conditioned Coefficients
void test_extreme_coefficients() {
    std::cout << "[Reliability 7] Extremely large and small matrix coefficients...\n";
    // 1e6 x1 + 1e-5 x2 <= 1e6
    // x1 + x2 <= 2
    // Maximize 1e6 x1 + x2
    indus::Model m;
    m.sense = indus::ObjSense::kMaximize;
    m.num_rows = 2;
    m.num_cols = 2;
    m.c = {1e6, 1.0};
    m.col_lower = {0.0, 0.0};
    m.col_upper = {10.0, 10.0};
    m.row_lower = {-1e20, -1e20};
    m.row_upper = {1e6, 2.0};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1e6}, {0, 1, 1e-5},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    m.A.set_from_triplets(2, 2, tri);

    indus::Options opts;
    opts.enable_scaling = true; // Ruiz equilibration
    indus::Solution sol = indus::solve(m, opts);

    ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal);
    ASSERT_NEAR(sol.col_value[0], 1.0, 1e-4);
    ASSERT_NEAR(sol.col_value[1], 1.0, 1e-4);
    std::cout << "  Passed: Ill-conditioned coefficient ratios handled safely with Ruiz scaling.\n";
}

// 8. Time Limit and Iteration Limit
void test_limits() {
    std::cout << "[Reliability 8] Time limit and iteration limit handling...\n";

    // Non-trivial 3x3 LP
    indus::Model m;
    m.sense = indus::ObjSense::kMaximize;
    m.num_rows = 3;
    m.num_cols = 3;
    m.c = {2.40, 1.60, 1.64};
    m.col_lower = {10.0, 0.0, 0.0};
    m.col_upper = {1e20, 45.0, 60.0};
    m.row_lower = {90.0, 40.0, -1e20};
    m.row_upper = {120.0, 40.0, 0.0};
    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0},
        {1, 0, 0.30}, {1, 1, 0.45}, {1, 2, 0.38},
        {2, 0, 0.80}, {2, 1, -0.86}, {2, 2, -0.22}
    };
    m.A.set_from_triplets(3, 3, tri);

    // Iteration limit = 0
    {
        indus::Options opts;
        opts.iteration_limit = 0;
        opts.enable_presolve = false;
        indus::Solution sol = indus::solve(m, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kIterationLimit);
    }

    // Time limit = 0.0
    {
        indus::Options opts;
        opts.time_limit = 0.0;
        indus::Solution sol = indus::solve(m, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kTimeLimit);
    }

    std::cout << "  Passed: Solver limits strictly honored with correct status codes.\n";
}

// 9. Repeated Solver Use (Idempotency and No Memory/State Leak)
void test_repeated_solver_use() {
    std::cout << "[Reliability 9] Repeated solver invocations (idempotency & state isolation)...\n";
    indus::Model m;
    m.sense = indus::ObjSense::kMaximize;
    m.num_rows = 2;
    m.num_cols = 2;
    m.c = {3.0, 2.0};
    m.col_lower = {0.0, 0.0};
    m.col_upper = {1e20, 1e20};
    m.row_lower = {-1e20, -1e20};
    m.row_upper = {4.0, 2.0};
    m.A.set_from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}});

    indus::Options opts;
    double first_obj = 0.0;

    for (int iter = 0; iter < 10; ++iter) {
        indus::Solution sol = indus::solve(m, opts);
        ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal);
        if (iter == 0) {
            first_obj = sol.objective_value;
        } else {
            ASSERT_NEAR(sol.objective_value, first_obj, 1e-12);
        }
        ASSERT_NEAR(sol.col_value[0], 3.0, 1e-9);
        ASSERT_NEAR(sol.col_value[1], 1.0, 1e-9);
    }

    std::cout << "  Passed: 10 consecutive solves yielded identical, deterministic solutions.\n";
}

// 10. CPU Fallback Without CUDA
void test_cpu_fallback_without_cuda() {
    std::cout << "[Reliability 10] Seamless CPU fallback for GPU PDHG...\n";
    indus::Model m;
    m.sense = indus::ObjSense::kMinimize;
    m.num_rows = 2;
    m.num_cols = 2;
    m.c = {1.0, 2.0};
    m.col_lower = {0.0, 0.0};
    m.col_upper = {10.0, 10.0};
    m.row_lower = {3.0, -1e20};
    m.row_upper = {1e20, 5.0};
    m.A.set_from_triplets(2, 2, {{0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, -1.0}});

    indus::Options opts;
    opts.use_gpu = true;
    opts.set("algorithm", "pdhg_cuda");

    indus::Solution sol = indus::solve(m, opts);
    ASSERT_TRUE(sol.status == indus::SolveStatus::kOptimal || sol.status == indus::SolveStatus::kFeasible);
    ASSERT_NEAR(sol.objective_value, 3.0, 1e-4);
    std::cout << "  Passed: GPU solve cleanly fell back to CPU reference with accurate optimum.\n";
}

} // namespace

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  INDUS-OPT PHASE 6: RELIABILITY, HARDENING & BOUNDARY AUDIT SUITE      \n";
    std::cout << "========================================================================\n\n";

    test_empty_model();
    test_zero_variables_with_constraints();
    test_zero_constraints_with_variables();
    test_invalid_bounds_and_validation();
    test_infeasible_and_unbounded();
    test_duplicate_constraints();
    test_extreme_coefficients();
    test_limits();
    test_repeated_solver_use();
    test_cpu_fallback_without_cuda();

    std::cout << "\n========================================================================\n";
    std::cout << "  ALL 10 PHASE 6 RELIABILITY & HARDENING TESTS PASSED (100% GREEN)       \n";
    std::cout << "========================================================================\n";
    return 0;
}
