#include "indus/model.hpp"
#include "indus/options.hpp"
#include "indus/pdhg.hpp"
#include "indus/gpu.hpp"
#include "indus/verifier.hpp"
#include "indus/io.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <filesystem>

namespace {

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion FAILED: " #cond << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define ASSERT_NEAR(val, expected, tol) \
    do { \
        double diff = std::abs((val) - (expected)); \
        if (diff > (tol)) { \
            std::cerr << "Assertion FAILED: " #val " (" << (val) << ") != " #expected " (" << (expected) \
                      << ") [diff=" << diff << " > tol=" << (tol) << "] at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#define ASSERT_EQ(val, expected) \
    do { \
        if ((val) != (expected)) { \
            std::cerr << "Assertion FAILED: " #val << " != " #expected << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1); \
        } \
    } while (0)

#ifndef INDUS_SOURCE_DIR
#define INDUS_SOURCE_DIR "."
#endif

std::string find_model_path(const std::string& rel) {
    std::filesystem::path p_src = std::filesystem::path(INDUS_SOURCE_DIR) / rel;
    if (std::filesystem::exists(p_src)) return p_src.string();
    if (std::filesystem::exists(rel)) return rel;
    if (std::filesystem::exists("../" + rel)) return "../" + rel;
    if (std::filesystem::exists("../../" + rel)) return "../../" + rel;
    return rel;
}

// 1. CPU PDHG on a 2-variable LP with known optimum
void test_2var_known_optimum() {
    std::cout << "[TEST 1] CPU PDHG 2-variable LP with known optimum...\n";
    // min -x1 - 2*x2
    // s.t.
    // x1 + x2 <= 4
    // x1 <= 3
    // x2 <= 3
    // x1 >= 0, x2 >= 0
    // Optimum: x1 = 1, x2 = 3, obj = -7.0
    indus::Model model;
    model.name = "test_2var";
    model.num_rows = 3;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMinimize;
    model.c = {-1.0, -2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1e20, 1e20};
    model.row_lower = {-1e20, -1e20, -1e20};
    model.row_upper = {4.0, 3.0, 3.0};
    model.row_names = {"r1", "r2", "r3"};
    model.col_names = {"x1", "x2"};

    std::vector<indus::la::Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0},
        {2, 1, 1.0}
    };
    model.A.set_from_triplets(3, 2, trips);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.set("tolerance", 1e-5);
    opts.iteration_limit = 20000;

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_EQ(sol.status, indus::SolveStatus::kOptimal);
    ASSERT_NEAR(sol.objective_value, -7.0, 1e-3);
    ASSERT_NEAR(sol.col_value[0], 1.0, 1e-3);
    ASSERT_NEAR(sol.col_value[1], 3.0, 1e-3);

    auto vres = indus::verifier::verify_solution(model, sol, 1e-3);
    ASSERT_TRUE(vres.passed);
    std::cout << "  Passed. Obj=" << sol.objective_value << " (expected -7.0)\n";
}

// 2. CPU PDHG on an equality-constrained LP
void test_equality_constrained_lp() {
    std::cout << "[TEST 2] CPU PDHG on equality-constrained LP...\n";
    // min x1 + 2*x2 + 3*x3
    // s.t.
    // x1 + x2 + x3 = 6
    // 2*x1 + x2 = 5
    // x1, x2, x3 >= 0
    // x1 = 1, x2 = 3, x3 = 2 => obj = 1 + 6 + 6 = 13.0
    indus::Model model;
    model.num_rows = 2;
    model.num_cols = 3;
    model.c = {1.0, 2.0, 3.0};
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {1e20, 1e20, 1e20};
    model.row_lower = {6.0, 5.0};
    model.row_upper = {6.0, 5.0};
    model.row_names = {"eq1", "eq2"};
    model.col_names = {"x1", "x2", "x3"};

    std::vector<indus::la::Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}, {0, 2, 1.0},
        {1, 0, 2.0}, {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 3, trips);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.set("tolerance", 1e-5);
    opts.iteration_limit = 20000;

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_EQ(sol.status, indus::SolveStatus::kOptimal);
    ASSERT_NEAR(sol.objective_value, 13.0, 1e-3);
    ASSERT_NEAR(sol.col_value[0] + sol.col_value[1] + sol.col_value[2], 6.0, 1e-3);
    ASSERT_NEAR(2.0 * sol.col_value[0] + sol.col_value[1], 5.0, 1e-3);

    auto vres = indus::verifier::verify_solution(model, sol, 1e-3);
    ASSERT_TRUE(vres.passed);
    std::cout << "  Passed. Obj=" << sol.objective_value << " (expected 13.0)\n";
}

// 3. CPU PDHG on <=, >=, and ranged rows
void test_row_senses_and_ranged() {
    std::cout << "[TEST 3] CPU PDHG with mixed row senses (<=, >=, ==, ranged)...\n";
    // min 2*x1 + 3*x2
    // s.t.
    // row1: x1 + x2 >= 2  (>= row)
    // row2: x1 - x2 <= 4  (<= row)
    // row3: 1 <= 2*x1 + x2 <= 6 (ranged row)
    // x1, x2 >= 0
    indus::Model model;
    model.num_rows = 3;
    model.num_cols = 2;
    model.c = {2.0, 3.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.row_lower = {2.0, -1e20, 1.0};
    model.row_upper = {1e20, 4.0, 6.0};
    model.row_names = {"ge_row", "le_row", "ranged_row"};
    model.col_names = {"x1", "x2"};

    std::vector<indus::la::Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, -1.0},
        {2, 0, 2.0}, {2, 1, 1.0}
    };
    model.A.set_from_triplets(3, 2, trips);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.set("tolerance", 1e-5);
    opts.iteration_limit = 20000;

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_EQ(sol.status, indus::SolveStatus::kOptimal);
    // x1 = 2, x2 = 0 => 2*x1 + 3*x2 = 4.0. Checks:
    // ge: 2 >= 2 (ok)
    // le: 2 - 0 <= 4 (ok)
    // ranged: 1 <= 4 <= 6 (ok)
    ASSERT_NEAR(sol.objective_value, 4.0, 1e-3);

    auto vres = indus::verifier::verify_solution(model, sol, 1e-3);
    ASSERT_TRUE(vres.passed);
    std::cout << "  Passed. Mixed row senses satisfied cleanly. Obj=" << sol.objective_value << "\n";
}

// 4. CPU PDHG with finite and free variables
void test_free_and_finite_bounds() {
    std::cout << "[TEST 4] CPU PDHG with free variables and two-sided box bounds...\n";
    // min 2*x1 + x2
    // s.t.
    // x1 + x2 = 5
    // x1 in [-10, 10]
    // x2 free in [-inf, +inf]
    // Clearly x1 will go as small as possible to minimize 2*x1 + x2 = x1 + (x1+x2) = x1 + 5.
    // Since x2 is free, x1 = -10, x2 = 15 => obj = 2*(-10) + 15 = -5.
    indus::Model model;
    model.num_rows = 1;
    model.num_cols = 2;
    model.c = {2.0, 1.0};
    model.col_lower = {-10.0, -1e20};
    model.col_upper = {10.0, 1e20};
    model.row_lower = {5.0};
    model.row_upper = {5.0};
    model.row_names = {"sum_eq"};
    model.col_names = {"x1_bounded", "x2_free"};

    std::vector<indus::la::Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0}
    };
    model.A.set_from_triplets(1, 2, trips);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.set("tolerance", 1e-5);
    opts.iteration_limit = 20000;

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_EQ(sol.status, indus::SolveStatus::kOptimal);
    ASSERT_NEAR(sol.objective_value, -5.0, 1e-3);
    ASSERT_NEAR(sol.col_value[0], -10.0, 1e-3);
    ASSERT_NEAR(sol.col_value[1], 15.0, 1e-3);

    auto vres = indus::verifier::verify_solution(model, sol, 1e-3);
    ASSERT_TRUE(vres.passed);
    std::cout << "  Passed. Free variable handled without divergence.\n";
}

// 5. Minimization and maximization equivalence
void test_minimization_and_maximization() {
    std::cout << "[TEST 5] CPU PDHG minimization vs maximization symmetry...\n";
    // min -2*x1 - 3*x2 s.t. x1 + x2 <= 5, x >= 0 => obj = -15 (at x1=0, x2=5)
    // max 2*x1 + 3*x2 s.t. x1 + x2 <= 5, x >= 0 => obj = 15 (at x1=0, x2=5)
    indus::Model min_model;
    min_model.num_rows = 1;
    min_model.num_cols = 2;
    min_model.sense = indus::ObjSense::kMinimize;
    min_model.c = {-2.0, -3.0};
    min_model.col_lower = {0.0, 0.0};
    min_model.col_upper = {1e20, 1e20};
    min_model.row_lower = {-1e20};
    min_model.row_upper = {5.0};
    min_model.row_names = {"limit"};
    min_model.col_names = {"x1", "x2"};
    std::vector<indus::la::Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    min_model.A.set_from_triplets(1, 2, trips);

    indus::Model max_model = min_model;
    max_model.sense = indus::ObjSense::kMaximize;
    max_model.c = {2.0, 3.0};

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.set("tolerance", 1e-5);
    opts.iteration_limit = 20000;

    indus::Solution min_sol = indus::solve(min_model, opts);
    indus::Solution max_sol = indus::solve(max_model, opts);

    ASSERT_EQ(min_sol.status, indus::SolveStatus::kOptimal);
    ASSERT_EQ(max_sol.status, indus::SolveStatus::kOptimal);
    ASSERT_NEAR(min_sol.objective_value, -15.0, 1e-3);
    ASSERT_NEAR(max_sol.objective_value, 15.0, 1e-3);
    ASSERT_NEAR(min_sol.col_value[1], max_sol.col_value[1], 1e-3);

    std::cout << "  Passed. Min obj = " << min_sol.objective_value << ", Max obj = " << max_sol.objective_value << "\n";
}

// 6. Presolve + PDHG + Postsolve Integration
void test_presolve_pdhg_postsolve() {
    std::cout << "[TEST 6] Presolve + PDHG + Postsolve integration...\n";
    // Model with a fixed variable and redundant singleton row
    // x1 + x2 <= 4
    // x3 = 5 (fixed var)
    // x2 <= 10 (redundant)
    // min -x1 - x2 + 2*x3
    indus::Model model;
    model.num_rows = 2;
    model.num_cols = 3;
    model.c = {-1.0, -1.0, 2.0};
    model.col_lower = {0.0, 0.0, 5.0};
    model.col_upper = {1e20, 1e20, 5.0};
    model.row_lower = {-1e20, -1e20};
    model.row_upper = {4.0, 10.0};
    model.row_names = {"r1", "r2"};
    model.col_names = {"x1", "x2", "x3"};

    std::vector<indus::la::Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 3, trips);

    indus::Options opts;
    opts.enable_presolve = true;
    opts.set("algorithm", "pdhg_cpu");
    opts.set("tolerance", 1e-5);
    opts.iteration_limit = 20000;

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_EQ(sol.status, indus::SolveStatus::kOptimal);
    // At optimum: x1 + x2 = 4, x3 = 5 => obj = -4 + 10 = 6.0
    ASSERT_NEAR(sol.objective_value, 6.0, 1e-3);
    ASSERT_NEAR(sol.col_value[2], 5.0, 1e-5);

    auto vres = indus::verifier::verify_solution(model, sol, 1e-3);
    ASSERT_TRUE(vres.passed);
    std::cout << "  Passed. Presolve reduced model, PDHG solved, postsolve validated cleanly.\n";
}

// 7. Infeasible model behavior
void test_infeasible_model() {
    std::cout << "[TEST 7] CPU PDHG on infeasible model...\n";
    // x1 + x2 <= 1
    // x1 + x2 >= 3
    // x1, x2 >= 0
    indus::Model model;
    model.num_rows = 2;
    model.num_cols = 2;
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.row_lower = {-1e20, 3.0};
    model.row_upper = {1.0, 1e20};
    model.row_names = {"r1", "r2"};
    model.col_names = {"x1", "x2"};

    std::vector<indus::la::Triplet> trips = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 2, trips);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.iteration_limit = 500;

    indus::Solution sol = indus::solve(model, opts);
    // PDHG must not claim OPTIMAL on infeasible models!
    ASSERT_TRUE(sol.status != indus::SolveStatus::kOptimal);
    std::cout << "  Passed. Solver reported non-optimal status: " << indus::to_string(sol.status) << "\n";
}

// 8. Iteration limit behavior
void test_iteration_limit() {
    std::cout << "[TEST 8] CPU PDHG iteration limit behavior...\n";
    std::string path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps");
    indus::Model model = indus::io::read_mps(path);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cpu");
    opts.enable_presolve = false;
    opts.iteration_limit = 10; // Very small limit

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_TRUE(sol.status == indus::SolveStatus::kIterationLimit || sol.status == indus::SolveStatus::kFeasible);
    ASSERT_TRUE(sol.status != indus::SolveStatus::kOptimal);
    std::cout << "  Passed. Correctly stopped at iteration limit without claiming optimality.\n";
}

// 9. Simplex vs PDHG Objective Comparison on Benchmark Models
void test_simplex_vs_pdhg_comparison() {
    std::cout << "[TEST 9] Simplex vs CPU PDHG objective comparison on Netlib instances...\n";
    std::vector<std::string> test_files = {
        "SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps",
        "SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"
    };

    for (const auto& rel_path : test_files) {
        std::string full_path = find_model_path(rel_path);
        indus::Model model = indus::io::read_mps(full_path);

        indus::Options simplex_opts;
        simplex_opts.set("algorithm", "dual_simplex");
        simplex_opts.enable_presolve = true;
        indus::Solution simplex_sol = indus::solve(model, simplex_opts);

        indus::Options pdhg_opts;
        pdhg_opts.set("algorithm", "pdhg_cpu");
        pdhg_opts.enable_presolve = true;
        pdhg_opts.set("tolerance", 1e-4);
        pdhg_opts.iteration_limit = 50000;
        indus::Solution pdhg_sol = indus::solve(model, pdhg_opts);

        ASSERT_EQ(simplex_sol.status, indus::SolveStatus::kOptimal);
        ASSERT_EQ(pdhg_sol.status, indus::SolveStatus::kOptimal);

        const double rel_diff = std::abs(simplex_sol.objective_value - pdhg_sol.objective_value) /
                                (1.0 + std::abs(simplex_sol.objective_value));
        std::cout << "  Model " << model.name << ": Simplex obj = " << simplex_sol.objective_value
                  << ", PDHG obj = " << pdhg_sol.objective_value
                  << ", RelDiff = " << rel_diff << "\n";
        ASSERT_TRUE(rel_diff < 1e-3);
    }
    std::cout << "  Passed. Objectives match within tolerance.\n";
}

// 10. Hardware Abstraction & CPU Fallback Test
void test_hardware_probe_and_cpu_fallback() {
    std::cout << "[TEST 10] Hardware probe and seamless CPU fallback...\n";
    auto dev_info = indus::gpu::probe_device();
    std::cout << "  Detected device: " << dev_info.device_name
              << " | Available: " << (dev_info.available ? "YES" : "NO") << "\n";

    // Requesting GPU mode on any machine must succeed (running either real GPU or clean CPU fallback)
    indus::Model model;
    model.num_rows = 1;
    model.num_cols = 2;
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.row_lower = {3.0};
    model.row_upper = {10.0};
    model.row_names = {"r1"};
    model.col_names = {"x1", "x2"};
    std::vector<indus::la::Triplet> trips = {{0, 0, 1.0}, {0, 1, 1.0}};
    model.A.set_from_triplets(1, 2, trips);

    indus::Options opts;
    opts.set("algorithm", "pdhg_cuda"); // Explicitly request GPU mode
    opts.use_gpu = true;
    opts.enable_presolve = false;
    opts.iteration_limit = 5000;

    indus::Solution sol = indus::solve(model, opts);
    ASSERT_EQ(sol.status, indus::SolveStatus::kOptimal);
    ASSERT_NEAR(sol.objective_value, 3.0, 1e-3);
    std::cout << "  Passed. GPU mode requested and executed successfully with status="
              << indus::to_string(sol.status) << ", obj=" << sol.objective_value << "\n";
}

} // namespace

int main() {
    std::cout << "================================================================\n";
    std::cout << "  INDUS-OPT PHASE 5: PDHG & GPU HARDWARE TEST SUITE\n";
    std::cout << "================================================================\n\n";

    test_2var_known_optimum();
    test_equality_constrained_lp();
    test_row_senses_and_ranged();
    test_free_and_finite_bounds();
    test_minimization_and_maximization();
    test_presolve_pdhg_postsolve();
    test_infeasible_model();
    test_iteration_limit();
    test_simplex_vs_pdhg_comparison();
    test_hardware_probe_and_cpu_fallback();

    std::cout << "\n================================================================\n";
    std::cout << "  ALL 10 PHASE 5 PDHG & GPU TESTS PASSED SUCCESSFULLY (100% GREEN)\n";
    std::cout << "================================================================\n";
    return 0;
}
