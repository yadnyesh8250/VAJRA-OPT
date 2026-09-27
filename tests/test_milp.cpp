#include "indus/milp.hpp"
#include "indus/model.hpp"
#include "indus/options.hpp"
#include "indus/verifier.hpp"
#include "indus/io.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
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

// Helper: build standard CSC sparse matrix from triplets
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
// Test 1: Binary 0-1 Knapsack (Known Integer Optimum != LP Relaxation)
// Maximize 5*x1 + 6*x2 + 7*x3 + 8*x4
// Subject to:
//   2*x1 + 3*x2 + 4*x3 + 5*x4 <= 7
//   x1, x2, x3, x4 in {0, 1}
//
// LP relaxation: x1=1, x2=1, x3=0.5, x4=0 -> obj = 14.5
// Rounding x3 up: weight = 2+3+4 = 9 > 7 (INFEASIBLE)
// True integer optimum: x1=1, x4=1 (weight 7, val 13) or x2=1, x3=1 (weight 7, val 13)
// -----------------------------------------------------------------------------
void test_binary_knapsack() {
    std::cout << "\n[TEST] test_binary_knapsack\n";
    indus::Model model;
    model.name = "knapsack_01";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 1;
    model.num_cols = 4;

    model.c = {5.0, 6.0, 7.0, 8.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {1.0, 1.0, 1.0, 1.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger,
                      indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20};
    model.row_upper = {7.0};

    std::vector<int> rows = {0, 0, 0, 0};
    std::vector<int> cols = {0, 1, 2, 3};
    std::vector<double> vals = {2.0, 3.0, 4.0, 5.0};
    model.A = make_csc(1, 4, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.objective_value - 13.0) < 1e-4, "Integer optimum objective must be 13.0");
    TEST_ASSERT(sol.nodes > 1, "Branch-and-bound must explore more than 1 node");

    // Independent verification
    auto vres = indus::verifier::verify_solution(model, sol, 1e-5);
    TEST_ASSERT(vres.passed, "Solution verifier must pass");
    TEST_ASSERT(vres.integer_feasible, "All integer variables must be integral");
    TEST_ASSERT(vres.optimality_proven, "Global bound proof must be verified");
    TEST_ASSERT(vres.is_milp, "Model must be identified as MILP");

    TEST_PASS("Binary knapsack solved to verified optimum 13.0 (LP relaxation was 14.5)");
}

// -----------------------------------------------------------------------------
// Test 2: Mandatory Correction #6 Model
// - The LP relaxation is fractional.
// - Rounding the LP result is infeasible.
// - Branching is required.
// - The node limit returns NODE_LIMIT.
// - A feasible incumbent exists but optimality is NOT proven.
// -----------------------------------------------------------------------------
void test_node_limit_and_unproven_optimality() {
    std::cout << "\n[TEST] test_node_limit_and_unproven_optimality (Correction #6)\n";
    indus::Model model;
    model.name = "correction_6_model";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 2;
    model.num_cols = 6;

    // Multidimensional knapsack with 6 binary variables
    model.c = {10.0, 10.0, 12.0, 12.0, 14.0, 14.0};
    model.col_lower.assign(6, 0.0);
    model.col_upper.assign(6, 1.0);
    model.col_type.assign(6, indus::VarType::kInteger);

    // Row 0: 3*x1 + 3*x2 + 4*x3 + 4*x4 + 5*x5 + 5*x6 <= 11
    // Row 1: 2*x1 + 2*x2 + 3*x3 + 3*x4 + 4*x5 + 4*x6 <= 8
    model.row_lower = {-1e20, -1e20};
    model.row_upper = {11.0, 8.0};

    std::vector<int> rows = {
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1
    };
    std::vector<int> cols = {
        0, 1, 2, 3, 4, 5,
        0, 1, 2, 3, 4, 5
    };
    std::vector<double> vals = {
        3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
        2.0, 2.0, 3.0, 3.0, 4.0, 4.0
    };
    model.A = make_csc(2, 6, rows, cols, vals);

    // First solve with node_limit = 2 (triggers NODE_LIMIT while exploring)
    indus::Options limit_opts;
    limit_opts.algorithm = "milp";
    limit_opts.node_limit = 2; // Strict node limit!

    indus::Solution limited_sol = indus::solve(model, limit_opts);

    TEST_ASSERT(limited_sol.status == indus::SolveStatus::kNodeLimit,
                "Solver must return kNodeLimit when node limit is reached");
    TEST_ASSERT(limited_sol.nodes <= 2, "Explored nodes must respect node limit");

    // Now test full solve to confirm LP relaxation is fractional & branching works
    indus::Options full_opts;
    full_opts.algorithm = "milp";
    full_opts.node_limit = 50000;

    indus::Solution full_sol = indus::solve(model, full_opts);
    TEST_ASSERT(full_sol.status == indus::SolveStatus::kOptimal, "Full solve must reach kOptimal");
    TEST_ASSERT(full_sol.nodes > 1, "Branching must be required");

    auto full_vres = indus::verifier::verify_solution(model, full_sol, 1e-5);
    TEST_ASSERT(full_vres.passed, "Full solution must verify");
    TEST_ASSERT(full_vres.optimality_proven, "Full solution optimality must be proven");

    // Also verify that a solution with node limit that has an incumbent does NOT claim optimality
    if (limited_sol.has_incumbent) {
        auto lim_vres = indus::verifier::verify_solution(model, limited_sol, 1e-5);
        TEST_ASSERT(lim_vres.passed, "Feasible incumbent from node limit must pass feasibility verification");
        TEST_ASSERT(!lim_vres.optimality_proven, "Optimality MUST NOT be proven when terminated at node limit");
        TEST_ASSERT(lim_vres.summary.find("UNPROVEN") != std::string::npos,
                    "Verifier summary must explicitly state bound is unproven");
    }

    TEST_PASS("Model with fractional LP relaxation, branching, and node limit verified per Correction #6");
}

// -----------------------------------------------------------------------------
// Test 3: General Integer Production Planning (Maximization)
// Maximize 4*x1 + 5*x2
// Subject to:
//   2*x1 + 3*x2 <= 16
//   3*x1 + 2*x2 <= 16
//   x1, x2 in {0, 1, 2, ..., 10}
//
// LP relaxation: x1 = 3.2, x2 = 3.2 -> obj = 28.8
// Integer optimum: x1 = 2, x2 = 4 (or x1 = 4, x2 = 2 has obj 26) -> obj = 4*2 + 5*4 = 28.0
// -----------------------------------------------------------------------------
void test_integer_production_planning() {
    std::cout << "\n[TEST] test_integer_production_planning\n";
    indus::Model model;
    model.name = "production_milp";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 2;
    model.num_cols = 2;

    model.c = {4.0, 5.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20, -1e20};
    model.row_upper = {16.0, 16.0};

    std::vector<int> rows = {0, 0, 1, 1};
    std::vector<int> cols = {0, 1, 0, 1};
    std::vector<double> vals = {2.0, 3.0, 3.0, 2.0};
    model.A = make_csc(2, 2, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(std::abs(sol.objective_value - 28.0) < 1e-4, "Integer optimum objective must be 28.0");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) < 1e-4, "x1 must be 2.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 4.0) < 1e-4, "x2 must be 4.0");

    auto vres = indus::verifier::verify_solution(model, sol, 1e-5);
    TEST_ASSERT(vres.passed, "Verifier must pass");
    TEST_ASSERT(vres.optimality_proven, "Optimality proof must be verified");

    TEST_PASS("Integer production planning solved to verified optimum (2, 4) with obj 28.0");
}

// -----------------------------------------------------------------------------
// Test 4: Mixed-Integer Minimization (Continuous + Integer Variables)
// Minimize 3*x1 + 2*x2 + 1.5*y1
// Subject to:
//   x1 + x2 + y1 >= 4.5
//   2*x1 + y1 >= 3.0
//   x1, x2 in Z+ (integer), y1 in R+ (continuous)
// -----------------------------------------------------------------------------
void test_mixed_integer_minimization() {
    std::cout << "\n[TEST] test_mixed_integer_minimization\n";
    indus::Model model;
    model.name = "mixed_minimization";
    model.sense = indus::ObjSense::kMinimize;
    model.num_rows = 2;
    model.num_cols = 3;

    model.c = {3.0, 2.0, 1.5};
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {10.0, 10.0, 10.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger, indus::VarType::kContinuous};

    model.row_lower = {4.5, 3.0};
    model.row_upper = {1e20, 1e20};

    std::vector<int> rows = {0, 0, 0, 1, 1};
    std::vector<int> cols = {0, 1, 2, 0, 2};
    std::vector<double> vals = {1.0, 1.0, 1.0, 2.0, 1.0};
    model.A = make_csc(2, 3, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(sol.quality.is_primal_feasible, "Primal feasibility must pass");
    TEST_ASSERT(sol.quality.is_integer_feasible, "Integrality feasibility must pass");

    // Check that x1 and x2 are integers, but y1 can be fractional
    const double x1 = sol.col_value[0];
    const double x2 = sol.col_value[1];
    TEST_ASSERT(std::abs(x1 - std::round(x1)) < 1e-5, "x1 must be integer");
    TEST_ASSERT(std::abs(x2 - std::round(x2)) < 1e-5, "x2 must be integer");

    auto vres = indus::verifier::verify_solution(model, sol, 1e-5);
    TEST_ASSERT(vres.passed, "Verifier must pass");
    TEST_ASSERT(vres.optimality_proven, "Optimality proof must be verified");

    TEST_PASS("Mixed continuous-integer minimization solved and verified");
}

// -----------------------------------------------------------------------------
// Test 5: Infeasible MILP (LP relaxation feasible, but no integer solution)
// 2*x1 + 4*x2 = 7
// x1, x2 in Z+
// 2*(x1 + 2*x2) is always even, 7 is odd -> integer infeasible!
// -----------------------------------------------------------------------------
void test_infeasible_milp() {
    std::cout << "\n[TEST] test_infeasible_milp\n";
    indus::Model model;
    model.name = "infeasible_integer";
    model.sense = indus::ObjSense::kMinimize;
    model.num_rows = 1;
    model.num_cols = 2;

    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {5.0, 5.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {7.0};
    model.row_upper = {7.0};

    std::vector<int> rows = {0, 0};
    std::vector<int> cols = {0, 1};
    std::vector<double> vals = {2.0, 4.0};
    model.A = make_csc(1, 2, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kInfeasible,
                "Solver must prove integer infeasibility");
    TEST_ASSERT(!sol.has_incumbent, "Infeasible model must have no incumbent");

    TEST_PASS("Branch-and-bound proved integer infeasibility on 2*x1 + 4*x2 = 7");
}

// -----------------------------------------------------------------------------
// Test 6: Verifier Integrity Audit (Rejects Integrality Violations)
// -----------------------------------------------------------------------------
void test_verifier_integrality_rejection() {
    std::cout << "\n[TEST] test_verifier_integrality_rejection\n";
    indus::Model model;
    model.name = "verifier_audit";
    model.sense = indus::ObjSense::kMinimize;
    model.num_rows = 1;
    model.num_cols = 2;

    model.c = {1.0, 2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {5.0, 5.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kContinuous};

    model.row_lower = {3.0};
    model.row_upper = {3.0};

    std::vector<int> rows = {0, 0};
    std::vector<int> cols = {0, 1};
    std::vector<double> vals = {1.0, 1.0};
    model.A = make_csc(1, 2, rows, cols, vals);

    // Create a bogus solution that satisfies linear constraints but violates integrality on x1
    indus::Solution bad_sol;
    bad_sol.status = indus::SolveStatus::kOptimal;
    bad_sol.col_value = {1.5, 1.5}; // x1 is 1.5, which is fractional!
    bad_sol.row_value = {3.0};
    bad_sol.objective_value = 1.0 * 1.5 + 2.0 * 1.5; // 4.5
    bad_sol.has_incumbent = true;

    auto vres = indus::verifier::verify_solution(model, bad_sol, 1e-5);
    TEST_ASSERT(!vres.passed, "Verifier MUST REJECT fractional integer variable");
    TEST_ASSERT(!vres.integer_feasible, "integer_feasible must be false");
    TEST_ASSERT(std::abs(vres.max_integrality_violation - 0.5) < 1e-4, "Integrality violation must be 0.5");
    TEST_ASSERT(!vres.violations.empty(), "Violations list must be populated");

    TEST_PASS("Verifier strictly caught fractional value on integer variable (violation: 0.5)");
}

// -----------------------------------------------------------------------------
// Test 7: Solved to Optimality at Root Node
// -----------------------------------------------------------------------------
void test_root_node_optimality() {
    std::cout << "\n[TEST] test_root_node_optimality\n";
    indus::Model model;
    model.name = "root_optimal";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 1;
    model.num_cols = 2;

    // Maximize x1 + x2 s.t. x1 <= 2, x2 <= 3
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {2.0, 3.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20};
    model.row_upper = {10.0};

    std::vector<int> rows = {0, 0};
    std::vector<int> cols = {0, 1};
    std::vector<double> vals = {1.0, 1.0};
    model.A = make_csc(1, 2, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(sol.nodes == 1, "Must be solved to optimality at root node (1 node)");
    TEST_ASSERT(std::abs(sol.objective_value - 5.0) < 1e-4, "Objective must be 5.0");

    auto vres = indus::verifier::verify_solution(model, sol, 1e-5);
    TEST_ASSERT(vres.passed, "Verifier must pass");
    TEST_ASSERT(vres.optimality_proven, "Optimality proof must be verified");

    TEST_PASS("Root node integer optimality correctly recognized in 1 node");
}

// -----------------------------------------------------------------------------
// Test 8: LP Relaxation Iteration Limit Handling (Requirement 1 & 6)
// Proves that when a node relaxation hits iteration limit:
// - do not prune the node
// - do not update the global bound from that result
// - terminate the MILP with an honest unverified/limit status (never OPTIMAL)
// -----------------------------------------------------------------------------
void test_lp_relaxation_iteration_limit() {
    std::cout << "\n[TEST] test_lp_relaxation_iteration_limit\n";
    indus::Model model;
    model.name = "iter_limit_model";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 2;
    model.num_cols = 4;

    model.c = {5.0, 6.0, 7.0, 8.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {1.0, 1.0, 1.0, 1.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger,
                      indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20, -1e20};
    model.row_upper = {7.0, 5.0};

    std::vector<int> rows = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int> cols = {0, 1, 2, 3, 0, 1, 2, 3};
    std::vector<double> vals = {2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 2.0, 3.0};
    model.A = make_csc(2, 4, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";
    opts.iteration_limit = 1; // Strict limit: simplex cannot solve in 1 iteration!

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kIterationLimit,
                "Solver must return kIterationLimit when LP relaxation hits iteration limit");
    TEST_ASSERT(sol.status != indus::SolveStatus::kOptimal,
                "Solver MUST NEVER claim kOptimal when LP relaxation is non-optimal");
    TEST_ASSERT(!sol.search_completed, "Search must be marked as not completed");

    auto vres = indus::verifier::verify_solution(model, sol, 1e-5);
    TEST_ASSERT(!vres.optimality_proven, "Optimality proof must be rejected when iteration limit hit");

    TEST_PASS("LP relaxation iteration limit handled safely without claiming false optimality");
}

// -----------------------------------------------------------------------------
// Test 9: Node Limit Without Incumbent (Requirement 6)
// -----------------------------------------------------------------------------
void test_node_limit_without_incumbent() {
    std::cout << "\n[TEST] test_node_limit_without_incumbent\n";
    indus::Model model;
    model.name = "node_limit_no_incumbent";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 1;
    model.num_cols = 4;

    model.c = {5.0, 6.0, 7.0, 8.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {1.0, 1.0, 1.0, 1.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger,
                      indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20};
    model.row_upper = {7.0};

    std::vector<int> rows = {0, 0, 0, 0};
    std::vector<int> cols = {0, 1, 2, 3};
    std::vector<double> vals = {2.0, 3.0, 4.0, 5.0};
    model.A = make_csc(1, 4, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";
    opts.node_limit = 1; // Strict node limit of 1: only root node explored, which is fractional

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kNodeLimit, "Solver must return kNodeLimit");
    TEST_ASSERT(!sol.has_incumbent, "Must not have found an incumbent at root node");
    TEST_ASSERT(!sol.search_completed, "Search must not be marked completed");
    TEST_ASSERT(!indus::verifier::verify_milp_optimality(sol), "Optimality proof must fail without incumbent");

    TEST_PASS("Node limit without incumbent correctly returns kNodeLimit with has_incumbent=false");
}

// -----------------------------------------------------------------------------
// Test 10: Time Limit Handling (Requirement 6)
// -----------------------------------------------------------------------------
void test_time_limit_handling() {
    std::cout << "\n[TEST] test_time_limit_handling\n";
    indus::Model model;
    model.name = "time_limit_model";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 1;
    model.num_cols = 4;

    model.c = {5.0, 6.0, 7.0, 8.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {1.0, 1.0, 1.0, 1.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger,
                      indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20};
    model.row_upper = {7.0};

    std::vector<int> rows = {0, 0, 0, 0};
    std::vector<int> cols = {0, 1, 2, 3};
    std::vector<double> vals = {2.0, 3.0, 4.0, 5.0};
    model.A = make_csc(1, 4, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";
    opts.time_limit = 0.0; // 0 seconds time limit

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kTimeLimit, "Solver must return kTimeLimit");
    TEST_ASSERT(!sol.search_completed, "Search must not be marked completed");

    TEST_PASS("Time limit handling correctly returns kTimeLimit");
}

// -----------------------------------------------------------------------------
// Test 11: Invalid Model Metadata Validation (Requirement 5)
// -----------------------------------------------------------------------------
void test_invalid_model_metadata() {
    std::cout << "\n[TEST] test_invalid_model_metadata\n";

    indus::Options opts;
    opts.algorithm = "milp";

    // 1. col_type size mismatch
    {
        indus::Model m;
        m.num_rows = 1;
        m.num_cols = 2;
        m.c = {1.0, 2.0};
        m.col_lower = {0.0, 0.0};
        m.col_upper = {1.0, 1.0};
        m.col_type = {indus::VarType::kInteger}; // Size 1 != num_cols 2
        m.row_lower = {-1e20};
        m.row_upper = {1.0};
        std::vector<int> rows = {0, 0};
        std::vector<int> cols = {0, 1};
        std::vector<double> vals = {1.0, 1.0};
        m.A = make_csc(1, 2, rows, cols, vals);

        indus::Solution sol = indus::solve(m, opts);
        TEST_ASSERT(sol.status == indus::SolveStatus::kModelError,
                    "col_type size mismatch must return kModelError");
    }

    // 2. NaN bound
    {
        indus::Model m;
        m.num_rows = 1;
        m.num_cols = 2;
        m.c = {1.0, 2.0};
        m.col_lower = {std::numeric_limits<double>::quiet_NaN(), 0.0};
        m.col_upper = {1.0, 1.0};
        m.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};
        m.row_lower = {-1e20};
        m.row_upper = {1.0};
        std::vector<int> rows = {0, 0};
        std::vector<int> cols = {0, 1};
        std::vector<double> vals = {1.0, 1.0};
        m.A = make_csc(1, 2, rows, cols, vals);

        indus::Solution sol = indus::solve(m, opts);
        TEST_ASSERT(sol.status == indus::SolveStatus::kModelError,
                    "NaN column bound must return kModelError");
    }

    // 3. Lower > Upper bound
    {
        indus::Model m;
        m.num_rows = 1;
        m.num_cols = 2;
        m.c = {1.0, 2.0};
        m.col_lower = {5.0, 0.0};
        m.col_upper = {2.0, 1.0};
        m.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};
        m.row_lower = {-1e20};
        m.row_upper = {1.0};
        std::vector<int> rows = {0, 0};
        std::vector<int> cols = {0, 1};
        std::vector<double> vals = {1.0, 1.0};
        m.A = make_csc(1, 2, rows, cols, vals);

        indus::Solution sol = indus::solve(m, opts);
        TEST_ASSERT(sol.status == indus::SolveStatus::kModelError,
                    "Lower > upper bound must return kModelError");
    }

    // 4. Infeasible fractional bounds for integer variable [0.2, 0.8]
    {
        indus::Model m;
        m.num_rows = 1;
        m.num_cols = 2;
        m.c = {1.0, 2.0};
        m.col_lower = {0.2, 0.0};
        m.col_upper = {0.8, 1.0};
        m.col_type = {indus::VarType::kInteger, indus::VarType::kInteger};
        m.row_lower = {-1e20};
        m.row_upper = {1.0};
        std::vector<int> rows = {0, 0};
        std::vector<int> cols = {0, 1};
        std::vector<double> vals = {1.0, 1.0};
        m.A = make_csc(1, 2, rows, cols, vals);

        indus::Solution sol = indus::solve(m, opts);
        TEST_ASSERT(sol.status == indus::SolveStatus::kModelError,
                    "Empty integer range [0.2, 0.8] must return kModelError");
    }

    TEST_PASS("Model validation strictly caught invalid metadata and returned kModelError");
}

// -----------------------------------------------------------------------------
// Test 12: Unnamed Model Export and Re-Verification (Requirements 2, 3, 4)
// -----------------------------------------------------------------------------
void test_unnamed_model_export_and_reverification() {
    std::cout << "\n[TEST] test_unnamed_model_export_and_reverification\n";
    indus::Model model;
    model.name = ""; // Unnamed
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 1;
    model.num_cols = 4;

    model.c = {5.0, 6.0, 7.0, 8.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {1.0, 1.0, 1.0, 1.0};
    model.col_type = {indus::VarType::kInteger, indus::VarType::kInteger,
                      indus::VarType::kInteger, indus::VarType::kInteger};

    model.row_lower = {-1e20};
    model.row_upper = {7.0};

    std::vector<int> rows = {0, 0, 0, 0};
    std::vector<int> cols = {0, 1, 2, 3};
    std::vector<double> vals = {2.0, 3.0, 4.0, 5.0};
    model.A = make_csc(1, 4, rows, cols, vals);

    // Explicitly empty col_names and row_names
    model.col_names.clear();
    model.row_names.clear();

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);
    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(sol.search_completed, "Search must be completed");

    // Write solution to disk
    const std::string tmp_sol = "/tmp/test_unnamed_milp.sol";
    indus::io::write_solution(sol, model, tmp_sol);

    // Verify written solution file using verify_solution_file
    auto file_res = indus::verifier::verify_solution_file(model, tmp_sol, 1e-5);
    TEST_ASSERT(file_res.passed, "Exported unnamed model solution file must pass verification");
    TEST_ASSERT(file_res.optimality_proven, "Exported solution file must have proven optimality");
    TEST_ASSERT(file_res.integer_feasible, "Exported solution must be integer feasible");

    std::filesystem::remove(tmp_sol);
    TEST_PASS("Exported and re-verified .sol file for unnamed model with proof metadata");
}

// -----------------------------------------------------------------------------
// Test 13: Multi-Node Branching Tree (Requirement 6)
// Model requiring several branch nodes (depth > 3, nodes > 5)
// -----------------------------------------------------------------------------
void test_multi_node_branching_tree() {
    std::cout << "\n[TEST] test_multi_node_branching_tree\n";
    indus::Model model;
    model.name = "multi_branch";
    model.sense = indus::ObjSense::kMaximize;
    model.num_rows = 2;
    model.num_cols = 5;

    // Multidimensional knapsack with 5 binary variables
    model.c = {8.0, 11.0, 6.0, 15.0, 9.0};
    model.col_lower.assign(5, 0.0);
    model.col_upper.assign(5, 1.0);
    model.col_type.assign(5, indus::VarType::kInteger);

    // Row 0: 4*x0 + 5*x1 + 3*x2 + 7*x3 + 4*x4 <= 12
    // Row 1: 3*x0 + 4*x1 + 2*x2 + 5*x3 + 3*x4 <= 9
    model.row_lower = {-1e20, -1e20};
    model.row_upper = {12.0, 9.0};

    std::vector<int> rows = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    std::vector<int> cols = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
    std::vector<double> vals = {4.0, 5.0, 3.0, 7.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0};
    model.A = make_csc(2, 5, rows, cols, vals);

    indus::Options opts;
    opts.algorithm = "milp";

    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be kOptimal");
    TEST_ASSERT(sol.nodes >= 3, "Model must require several branch nodes (explored >= 3 nodes)");
    TEST_ASSERT(sol.search_completed, "Search must be completed");

    auto vres = indus::verifier::verify_solution(model, sol, 1e-5);
    TEST_ASSERT(vres.passed, "Solution must pass verification");
    TEST_ASSERT(vres.optimality_proven, "Optimality proof must be verified");

    TEST_PASS("Multi-node branching tree successfully solved and verified (nodes: " + std::to_string(sol.nodes) + ")");
}

} // namespace

int main() {
    std::cout << "========================================================================\n";
    std::cout << "  RUNNING NATIVE MILP BRANCH-AND-BOUND TEST SUITE (PHASE 7)             \n";
    std::cout << "========================================================================\n";

    test_binary_knapsack();
    test_node_limit_and_unproven_optimality();
    test_integer_production_planning();
    test_mixed_integer_minimization();
    test_infeasible_milp();
    test_verifier_integrality_rejection();
    test_root_node_optimality();
    test_lp_relaxation_iteration_limit();
    test_node_limit_without_incumbent();
    test_time_limit_handling();
    test_invalid_model_metadata();
    test_unnamed_model_export_and_reverification();
    test_multi_node_branching_tree();

    std::cout << "\n========================================================================\n";
    std::cout << "  MILP TEST RESULTS: " << g_passed << " PASSED, " << g_failed << " FAILED\n";
    std::cout << "========================================================================\n";

    return (g_failed == 0) ? 0 : 1;
}
