#include "indus/presolve.hpp"
#include "indus/model.hpp"
#include "indus/verifier.hpp"
#include "indus/tolerances.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "ASSERTION FAILED [" << __FILE__ << ":" << __LINE__ << "]: " << (msg) << std::endl; \
            std::exit(1); \
        } \
    } while(0)

namespace {

// 1. Empty Redundant Row: 0 nonzeros, bounds [-10, 10]
void test_empty_redundant_row() {
    std::cout << "[Test 1] Empty Redundant Row..." << std::endl;
    indus::Model model;
    model.name = "empty_redundant_row";
    model.num_rows = 2;
    model.num_cols = 2;
    model.c = {1.0, 2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_names = {"x1", "x2"};

    // Row 0 is empty with bounds [0, 5] (redundant because 0 in [0, 5])
    // Row 1 is x1 + x2 >= 2
    model.row_lower = {0.0, 2.0};
    model.row_upper = {5.0, 1e20};
    model.row_names = {"empty_r", "valid_r"};

    std::vector<indus::la::Triplet> tri = {
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(!pres_res.is_infeasible, "Empty redundant row should not be infeasible");
    TEST_ASSERT(pres_res.num_empty_rows == 1, "Expected 1 empty row reduction");
    TEST_ASSERT(pres_res.presolved_model.num_rows == 1, "Presolved model must have 1 row");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    TEST_ASSERT(std::abs(sol.objective_value - 2.0) <= 1e-6, "Objective value must be 2.0");
    TEST_ASSERT(std::abs(sol.row_value[0]) <= 1e-9, "Empty row activity must be 0");
    TEST_ASSERT(std::abs(sol.row_dual[0]) <= 1e-9, "Empty row dual must be 0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 2. Empty Infeasible Row: 0 nonzeros, bounds [2, 10] (0 not in [2, 10])
void test_empty_infeasible_row() {
    std::cout << "[Test 2] Empty Infeasible Row..." << std::endl;
    indus::Model model;
    model.name = "empty_infeasible_row";
    model.num_rows = 1;
    model.num_cols = 1;
    model.c = {1.0};
    model.col_lower = {0.0};
    model.col_upper = {10.0};
    model.col_names = {"x1"};

    // Empty row requiring activity >= 2
    model.row_lower = {2.0};
    model.row_upper = {10.0};
    model.row_names = {"empty_infeas"};
    model.A.set_from_triplets(1, 1, {}, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.is_infeasible, "Empty row with positive lower bound must be flagged infeasible");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kInfeasible, "Solver must report INFEASIBLE");
    std::cout << "  -> PASSED" << std::endl;
}

// 3. Fixed Variable Substitution: x2 fixed to 3.0
void test_fixed_variable_substitution() {
    std::cout << "[Test 3] Fixed Variable Substitution..." << std::endl;
    indus::Model model;
    model.name = "fixed_variable";
    model.num_rows = 1;
    model.num_cols = 2;
    model.c = {2.0, 5.0}; // min 2 x1 + 5 x2
    model.col_lower = {0.0, 3.0};
    model.col_upper = {10.0, 3.0}; // x2 is fixed to 3.0
    model.col_names = {"x1", "x2"};

    // x1 + 2 x2 >= 10  =>  x1 >= 10 - 2(3) = 4
    model.row_lower = {10.0};
    model.row_upper = {1e20};
    model.row_names = {"c1"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 2.0}
    };
    model.A.set_from_triplets(1, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.num_fixed_cols >= 1, "Expected fixed column reduction");
    TEST_ASSERT(pres_res.presolved_model.num_cols <= 1, "Reduced model should eliminate x2");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    // At optimal: x1 = 4, x2 = 3 => obj = 2(4) + 5(3) = 8 + 15 = 23
    TEST_ASSERT(std::abs(sol.objective_value - 23.0) <= 1e-6, "Objective must be 23.0");
    TEST_ASSERT(std::abs(sol.col_value[0] - 4.0) <= 1e-6, "x1 must be 4.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 3.0) <= 1e-6, "x2 must be 3.0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 4. Singleton Row Bound Tightening: 2 x1 <= 8 implies x1 <= 4
void test_singleton_row_tightening() {
    std::cout << "[Test 4] Singleton Row Bound Tightening..." << std::endl;
    indus::Model model;
    model.name = "singleton_row";
    model.num_rows = 2;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMaximize;
    model.c = {3.0, 1.0}; // max 3 x1 + x2
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_names = {"x1", "x2"};

    // Row 0: 2 x1 <= 8  (singleton row tightening x1 upper bound to 4)
    // Row 1: x1 + x2 <= 6
    model.row_lower = {-1e20, -1e20};
    model.row_upper = {8.0, 6.0};
    model.row_names = {"sing_row", "joint_row"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 2.0},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.num_singleton_rows == 1, "Expected 1 singleton row reduction");
    TEST_ASSERT(pres_res.presolved_model.num_rows == 1, "Singleton row should be dropped from model");
    TEST_ASSERT(std::abs(pres_res.presolved_model.col_upper[0] - 4.0) <= 1e-6, "x1 upper bound tightened to 4.0");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    // At optimal: x1 = 4, x2 = 2 => max 3(4) + 2 = 14
    TEST_ASSERT(std::abs(sol.objective_value - 14.0) <= 1e-6, "Objective must be 14.0");
    TEST_ASSERT(std::abs(sol.col_value[0] - 4.0) <= 1e-6, "x1 must be 4.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 2.0) <= 1e-6, "x2 must be 2.0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 5. Forcing Row Detection: x1 + x2 <= 0 with x1 >= 0, x2 >= 0 forces x1 = 0, x2 = 0
void test_forcing_row() {
    std::cout << "[Test 5] Forcing Row Detection..." << std::endl;
    indus::Model model;
    model.name = "forcing_row";
    model.num_rows = 2;
    model.num_cols = 3;
    model.c = {2.0, 3.0, 1.0}; // min 2 x1 + 3 x2 + x3
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {10.0, 10.0, 10.0};
    model.col_names = {"x1", "x2", "x3"};

    // Row 0: x1 + x2 <= 0  (forces x1 = 0, x2 = 0)
    // Row 1: x1 + x2 + x3 >= 5
    model.row_lower = {-1e20, 5.0};
    model.row_upper = {0.0, 1e20};
    model.row_names = {"forcing_r", "activity_r"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}
    };
    model.A.set_from_triplets(2, 3, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.num_forcing_rows >= 1, "Expected forcing row reduction");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    // At optimal: x1 = 0, x2 = 0, x3 = 5 => obj = 5
    TEST_ASSERT(std::abs(sol.objective_value - 5.0) <= 1e-6, "Objective must be 5.0");
    TEST_ASSERT(std::abs(sol.col_value[0]) <= 1e-6, "x1 must be 0");
    TEST_ASSERT(std::abs(sol.col_value[1]) <= 1e-6, "x2 must be 0");
    TEST_ASSERT(std::abs(sol.col_value[2] - 5.0) <= 1e-6, "x3 must be 5.0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 6. Redundant Row Removal: x1 + x2 <= 100 with x1 in [0, 2], x2 in [0, 3]
void test_redundant_row() {
    std::cout << "[Test 6] Redundant Row Removal..." << std::endl;
    indus::Model model;
    model.name = "redundant_row";
    model.num_rows = 2;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMaximize;
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {2.0, 3.0};
    model.col_names = {"x1", "x2"};

    // Row 0: x1 + x2 <= 100 (max activity is 5, bound is 100 => strictly redundant)
    // Row 1: x1 + x2 <= 4
    model.row_lower = {-1e20, -1e20};
    model.row_upper = {100.0, 4.0};
    model.row_names = {"redundant_r", "active_r"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.num_redundant_rows >= 1, "Expected redundant row reduction");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    TEST_ASSERT(std::abs(sol.objective_value - 4.0) <= 1e-6, "Objective must be 4.0");
    TEST_ASSERT(std::abs(sol.row_dual[0]) <= 1e-9, "Redundant row shadow price must be 0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 7. Free Variable Column Singleton Elimination
void test_free_variable_singleton() {
    std::cout << "[Test 7] Free Variable Column Singleton Elimination..." << std::endl;
    indus::Model model;
    model.name = "free_col_singleton";
    model.num_rows = 2;
    model.num_cols = 2;
    model.c = {3.0, 2.0}; // min 3 x1 + 2 x2
    // x1 is free (-1e20, 1e20), x2 in [0, 10]
    model.col_lower = {-1e20, 0.0};
    model.col_upper = {1e20, 10.0};
    model.col_names = {"x1_free", "x2"};

    // Row 0: x1 + 2 x2 = 10  (x1 appears ONLY in row 0)
    // Row 1: x2 >= 2
    model.row_lower = {10.0, 2.0};
    model.row_upper = {10.0, 1e20};
    model.row_names = {"eq_r", "bound_r"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 2.0},
        {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.num_free_col_singletons == 1, "Expected 1 free column singleton reduction");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    // Since x1 = 10 - 2 x2, obj = 3(10 - 2 x2) + 2 x2 = 30 - 4 x2.
    // To minimize, x2 should be as large as possible: x2 = 10 => x1 = -10.
    // Obj = 3(-10) + 2(10) = -10.
    TEST_ASSERT(std::abs(sol.objective_value - (-10.0)) <= 1e-6, "Objective must be -10.0");
    TEST_ASSERT(std::abs(sol.col_value[0] - (-10.0)) <= 1e-6, "x1 must be -10.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 10.0) <= 1e-6, "x2 must be 10.0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 8. Chained Multi-Pass Fixed-Point Reductions
void test_chained_fixed_point_reductions() {
    std::cout << "[Test 8] Chained Multi-Pass Fixed-Point Reductions..." << std::endl;
    indus::Model model;
    model.name = "chained_reductions";
    // 3 variables, 3 constraints
    // x1 in [2, 2] (fixed)
    // r0: x1 + x2 = 5  => once x1=2 is substituted, r0 becomes singleton 1.0 x2 = 3 => x2 fixed to 3
    // r1: x2 + x3 = 7  => once x2=3 is substituted, r1 becomes singleton 1.0 x3 = 4 => x3 fixed to 4
    // r2: x1 + x2 + x3 <= 100 (becomes redundant once all are fixed)
    model.num_rows = 3;
    model.num_cols = 3;
    model.c = {1.0, 2.0, 3.0};
    model.col_lower = {2.0, 0.0, 0.0};
    model.col_upper = {2.0, 10.0, 10.0};
    model.col_names = {"x1", "x2", "x3"};

    model.row_lower = {5.0, 7.0, -1e20};
    model.row_upper = {5.0, 7.0, 100.0};
    model.row_names = {"r0", "r1", "r2"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 1, 1.0}, {2, 2, 1.0}
    };
    model.A.set_from_triplets(3, 3, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.total_reductions() >= 3, "Chained reductions should cascade across multiple passes");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    // x1 = 2, x2 = 3, x3 = 4 => obj = 1(2) + 2(3) + 3(4) = 2 + 6 + 12 = 20
    TEST_ASSERT(std::abs(sol.objective_value - 20.0) <= 1e-6, "Objective must be 20.0");
    TEST_ASSERT(std::abs(sol.col_value[0] - 2.0) <= 1e-6, "x1 must be 2.0");
    TEST_ASSERT(std::abs(sol.col_value[1] - 3.0) <= 1e-6, "x2 must be 3.0");
    TEST_ASSERT(std::abs(sol.col_value[2] - 4.0) <= 1e-6, "x3 must be 4.0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 9. Numerical Safety on Ill-Conditioned Input
void test_ill_conditioned_numerical_safety() {
    std::cout << "[Test 9] Numerical Safety on Ill-Conditioned Input..." << std::endl;
    indus::Model model;
    model.name = "ill_conditioned";
    model.num_rows = 2;
    model.num_cols = 2;
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1e20, 1e20};
    model.col_names = {"x1", "x2"};

    // Row 0 has a tiny coefficient (1e-16) that should be dropped by zero tolerance
    // Row 1 is a valid constraint
    model.row_lower = {0.0, 1.0};
    model.row_upper = {10.0, 1e20};
    model.row_names = {"r_tiny", "r_normal"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1e-16},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    model.A.set_from_triplets(2, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(!pres_res.is_infeasible, "Tiny coefficients must not trigger false infeasibility");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    TEST_ASSERT(!std::isnan(sol.objective_value) && !std::isinf(sol.objective_value), "Objective must be finite");
    for (double v : sol.col_value) {
        TEST_ASSERT(!std::isnan(v) && !std::isinf(v), "Primal values must be finite");
    }
    for (double y : sol.row_dual) {
        TEST_ASSERT(!std::isnan(y) && !std::isinf(y), "Row duals must be finite");
    }

    std::cout << "  -> PASSED" << std::endl;
}

// 10. Basic Doubleton Equality Substitution: a1*x1 + a2*x2 = b
void test_doubleton_basic_equality() {
    std::cout << "[Test 10] Basic Doubleton Equality Substitution..." << std::endl;
    indus::Model model;
    model.name = "doubleton_basic";
    model.num_rows = 2;
    model.num_cols = 3;
    model.c = {2.0, 3.0, 1.0};
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {10.0, 10.0, 10.0};
    model.col_names = {"x1", "x2", "x3"};

    // Row 0 is doubleton equality: 2*x1 + 3*x2 = 6
    // Row 1 is x2 + x3 >= 1
    model.row_lower = {6.0, 1.0};
    model.row_upper = {6.0, 1e20};
    model.row_names = {"eq_dbltn", "ineq_r1"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 2.0}, {0, 1, 3.0},
        {1, 1, 1.0}, {1, 2, 1.0}
    };
    model.A.set_from_triplets(2, 3, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(!pres_res.is_infeasible, "Model must not be infeasible");
    TEST_ASSERT(pres_res.num_doubleton_rows == 1, "Expected exactly 1 doubleton reduction");
    TEST_ASSERT(pres_res.presolved_model.num_rows == 1, "Presolved model must have 1 row");
    TEST_ASSERT(pres_res.presolved_model.num_cols == 2, "Presolved model must have 2 cols");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve status must be OPTIMAL");
    // Verify primal equality row activity matches b=6.0
    TEST_ASSERT(std::abs(2.0 * sol.col_value[0] + 3.0 * sol.col_value[1] - 6.0) <= 1e-6,
                "Equality row 2*x1 + 3*x2 = 6 must be satisfied exactly");
    TEST_ASSERT(sol.col_value[1] + sol.col_value[2] >= 1.0 - 1e-6,
                "Inequality x2 + x3 >= 1 must be satisfied");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    TEST_ASSERT(vres.primal_feasible, "Solution must be primal feasible");
    TEST_ASSERT(vres.dual_feasible, "Solution must be dual feasible");
    std::cout << "  -> PASSED" << std::endl;
}

// 11. Doubleton Substitution with Eliminated Variable in Another Row
void test_doubleton_substitution_in_another_row() {
    std::cout << "[Test 11] Doubleton Substitution in Another Row..." << std::endl;
    indus::Model model;
    model.name = "doubleton_other_row";
    model.num_rows = 3;
    model.num_cols = 3;
    model.c = {1.0, 2.0, 4.0};
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {10.0, 10.0, 10.0};
    model.col_names = {"x1", "x2", "x3"};

    // Row 0 is doubleton equality: x1 + x2 = 5
    // Row 1 is 2*x1 + x2 + x3 >= 6 (contains eliminated variable x1)
    // Row 2 is x2 + 2*x3 >= 4
    model.row_lower = {5.0, 6.0, 4.0};
    model.row_upper = {5.0, 1e20, 1e20};
    model.row_names = {"r0_dbltn", "r1_coupl", "r2_other"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 2.0}, {1, 1, 1.0}, {1, 2, 1.0},
        {2, 1, 1.0}, {2, 2, 2.0}
    };
    model.A.set_from_triplets(3, 3, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(!pres_res.is_infeasible, "Model must not be infeasible");
    TEST_ASSERT(pres_res.num_doubleton_rows >= 1, "Expected doubleton reduction");

    indus::Options opts_presolve;
    opts_presolve.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts_presolve);

    indus::Options opts_nopresolve;
    opts_nopresolve.enable_presolve = false;
    indus::Solution sol_raw = indus::solve(model, opts_nopresolve);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Presolved status must be OPTIMAL");
    TEST_ASSERT(sol_raw.status == indus::SolveStatus::kOptimal, "Raw status must be OPTIMAL");
    TEST_ASSERT(std::abs(sol.objective_value - sol_raw.objective_value) <= 1e-6,
                "Presolved and raw objective values must match");

    // Check primal validity
    TEST_ASSERT(std::abs(sol.col_value[0] + sol.col_value[1] - 5.0) <= 1e-6, "x1 + x2 = 5 must hold");
    TEST_ASSERT(2.0 * sol.col_value[0] + sol.col_value[1] + sol.col_value[2] >= 6.0 - 1e-6, "Row 1 must hold");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Original model verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 12. Objective Preservation Before and After Presolve
void test_doubleton_objective_preservation() {
    std::cout << "[Test 12] Doubleton Objective Preservation..." << std::endl;
    indus::Model model;
    model.name = "doubleton_obj_preserv";
    model.num_rows = 2;
    model.num_cols = 3;
    model.c = {3.0, 5.0, 2.0};
    model.objective_offset = 12.5;
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {20.0, 20.0, 20.0};
    model.col_names = {"x1", "x2", "x3"};

    // Row 0: x1 + 2*x2 = 8
    // Row 1: x2 + x3 >= 2
    model.row_lower = {8.0, 2.0};
    model.row_upper = {8.0, 1e20};
    model.row_names = {"eq_r0", "ineq_r1"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 2.0},
        {1, 1, 1.0}, {1, 2, 1.0}
    };
    model.A.set_from_triplets(2, 3, tri, true);

    indus::Options opts_on;
    opts_on.enable_presolve = true;
    indus::Solution sol_on = indus::solve(model, opts_on);

    indus::Options opts_off;
    opts_off.enable_presolve = false;
    indus::Solution sol_off = indus::solve(model, opts_off);

    TEST_ASSERT(sol_on.status == indus::SolveStatus::kOptimal, "Presolved status must be OPTIMAL");
    TEST_ASSERT(sol_off.status == indus::SolveStatus::kOptimal, "Unpresolved status must be OPTIMAL");
    TEST_ASSERT(std::abs(sol_on.objective_value - sol_off.objective_value) <= 1e-8,
                "Objective discrepancy must be < 1e-8");

    auto vres = indus::verifier::verify_solution(model, sol_on);
    TEST_ASSERT(vres.passed, "Verifier must pass on reconstructed solution");
    std::cout << "  -> PASSED" << std::endl;
}

// 13. Bound Propagation Through Doubleton Equality
void test_doubleton_bound_propagation() {
    std::cout << "[Test 13] Bound Propagation Through Doubleton Equality..." << std::endl;
    indus::Model model;
    model.name = "doubleton_bound_prop";
    model.num_rows = 1;
    model.num_cols = 2;
    model.c = {0.0, -1.0}; // min -x2 (maximize x2)
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 100.0}; // x1 in [0, 10], x2 in [0, 100]
    model.col_names = {"x1", "x2"};

    // x1 - 2*x2 = 0 => x1 = 2*x2 => x2 = x1 / 2 <= 10 / 2 = 5!
    model.row_lower = {0.0};
    model.row_upper = {0.0};
    model.row_names = {"ratio_eq"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, -2.0}
    };
    model.A.set_from_triplets(1, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(!pres_res.is_infeasible, "Model must be feasible");
    TEST_ASSERT(pres_res.num_doubleton_rows == 1, "Must reduce doubleton row");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve must be OPTIMAL");
    TEST_ASSERT(std::abs(sol.col_value[1] - 5.0) <= 1e-6, "Propagated bound on x2 must be active: x2=5");
    TEST_ASSERT(std::abs(sol.col_value[0] - 10.0) <= 1e-6, "Recovered x1 must equal 10");
    TEST_ASSERT(std::abs(sol.objective_value - (-5.0)) <= 1e-6, "Objective value must be -5.0");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verifier must pass");
    std::cout << "  -> PASSED" << std::endl;
}

// 14. Infeasible Implied Bounds Detection
void test_doubleton_infeasible_implied_bounds() {
    std::cout << "[Test 14] Infeasible Implied Bounds Detection..." << std::endl;
    indus::Model model;
    model.name = "doubleton_infeasible";
    model.num_rows = 1;
    model.num_cols = 2;
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {2.0, 3.0}; // Maximum possible sum is 2 + 3 = 5
    model.col_names = {"x1", "x2"};

    // x1 + x2 = 10 (Impossible since max sum is 5)
    model.row_lower = {10.0};
    model.row_upper = {10.0};
    model.row_names = {"eq_impossible"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0}
    };
    model.A.set_from_triplets(1, 2, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(pres_res.is_infeasible, "Presolve must detect crossed bounds / infeasibility");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kInfeasible, "Solver status must be INFEASIBLE");
    std::cout << "  -> PASSED" << std::endl;
}

// 15. Chained Doubleton / Fixed / Singleton Cascading Reduction
void test_doubleton_chained_cascade() {
    std::cout << "[Test 15] Chained Doubleton/Fixed/Singleton Reduction..." << std::endl;
    indus::Model model;
    model.name = "doubleton_chained";
    model.num_rows = 3;
    model.num_cols = 4;
    model.c = {1.0, 2.0, 3.0, 4.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {10.0, 10.0, 10.0, 10.0};
    model.col_names = {"x1", "x2", "x3", "x4"};

    // Row 0: x1 + x2 = 4 (Doubleton)
    // Row 1: x2 = 3 (Fixed Column via singleton row)
    // Row 2: x3 + x4 = 6 (Doubleton)
    model.row_lower = {4.0, 3.0, 6.0};
    model.row_upper = {4.0, 3.0, 6.0};
    model.row_names = {"r0_dbl", "r1_fix", "r2_dbl"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0},
        {2, 2, 1.0}, {2, 3, 1.0}
    };
    model.A.set_from_triplets(3, 4, tri, true);

    indus::presolve::PresolveEngine engine;
    auto pres_res = engine.apply(model);

    TEST_ASSERT(!pres_res.is_infeasible, "Cascade model must be feasible");
    TEST_ASSERT(pres_res.total_reductions() >= 3, "Cascade must produce multiple reductions");

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Solve must be OPTIMAL");
    TEST_ASSERT(std::abs(sol.col_value[1] - 3.0) <= 1e-6, "x2 must be 3");
    TEST_ASSERT(std::abs(sol.col_value[0] - 1.0) <= 1e-6, "x1 must be 4 - 3 = 1");
    TEST_ASSERT(std::abs(sol.col_value[2] + sol.col_value[3] - 6.0) <= 1e-6, "x3 + x4 must be 6");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verifier must pass on cascaded solution");
    std::cout << "  -> PASSED" << std::endl;
}

// 16. Dual Feasibility and Complementarity Validation
void test_doubleton_dual_feasibility_complementarity() {
    std::cout << "[Test 16] Dual Feasibility & Complementarity Validation..." << std::endl;
    indus::Model model;
    model.name = "doubleton_kkt";
    model.num_rows = 1;
    model.num_cols = 2;
    model.c = {1.0, 4.0}; // min x1 + 4*x2
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_names = {"x1", "x2"};

    // x1 + x2 = 3
    model.row_lower = {3.0};
    model.row_upper = {3.0};
    model.row_names = {"eq3"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0}
    };
    model.A.set_from_triplets(1, 2, tri, true);

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    TEST_ASSERT(sol.status == indus::SolveStatus::kOptimal, "Status must be OPTIMAL");
    // Primal: x1 = 3, x2 = 0
    TEST_ASSERT(std::abs(sol.col_value[0] - 3.0) <= 1e-6, "x1 must be 3");
    TEST_ASSERT(std::abs(sol.col_value[1] - 0.0) <= 1e-6, "x2 must be 0");
    // Dual: row dual y0 = 1.0
    // d1 = 1 - 1*y0 = 0.0 (interior, complementary)
    // d2 = 4 - 1*y0 = 3.0 >= 0 (at lower bound, dual feasible)
    TEST_ASSERT(std::abs(sol.row_dual[0] - 1.0) <= 1e-6, "Row dual must be 1.0");
    TEST_ASSERT(std::abs(sol.col_dual[0]) <= 1e-6, "Reduced cost d1 must be 0 for interior variable");
    TEST_ASSERT(sol.col_dual[1] >= -1e-6, "Reduced cost d2 must be non-negative at lower bound");

    auto vres = indus::verifier::verify_solution(model, sol);
    TEST_ASSERT(vres.passed, "Verifier must pass");
    TEST_ASSERT(vres.dual_feasible, "Dual feasibility must be true");
    TEST_ASSERT(vres.max_dual_violation <= 1e-9, "Dual violation must be zero");
    TEST_ASSERT(vres.max_complementarity_violation <= 1e-9, "Complementarity violation must be zero");
    std::cout << "  -> PASSED" << std::endl;
}

} // namespace

int main() {
    std::cout << "========================================================================\n"
              << "  SIDDHANTA (INDUS-OPT): PHASE 4 PRESOLVE/POSTSOLVE TEST SUITE          \n"
              << "  Unit & Regression Tests for Reversible LP Reductions                 \n"
              << "========================================================================\n";

    test_empty_redundant_row();
    test_empty_infeasible_row();
    test_fixed_variable_substitution();
    test_singleton_row_tightening();
    test_forcing_row();
    test_redundant_row();
    test_free_variable_singleton();
    test_chained_fixed_point_reductions();
    test_ill_conditioned_numerical_safety();
    test_doubleton_basic_equality();
    test_doubleton_substitution_in_another_row();
    test_doubleton_objective_preservation();
    test_doubleton_bound_propagation();
    test_doubleton_infeasible_implied_bounds();
    test_doubleton_chained_cascade();
    test_doubleton_dual_feasibility_complementarity();

    std::cout << "\nALL 16 PHASE 4 PRESOLVE TESTS PASSED CLEANLY.\n";
    return 0;
}
