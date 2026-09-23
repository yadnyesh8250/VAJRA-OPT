#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>

#include "indus/types.hpp"
#include "indus/tolerances.hpp"
#include "indus/sparse.hpp"
#include "src/solvers/simplex/simplex_core.hpp"

using namespace indus;
using namespace indus::simplex;

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

// Helper to construct a SimplexModel from standard parameters
SimplexModel make_model(
    int m, int n,
    const std::vector<la::Triplet>& A_triplets,
    const std::vector<double>& c,
    const std::vector<double>& row_lb,
    const std::vector<double>& row_ub,
    const std::vector<double>& col_lb,
    const std::vector<double>& col_ub,
    ObjSense sense = ObjSense::kMinimize
) {
    SimplexModel model;
    model.num_rows = m;
    model.num_cols = n;
    model.num_vars = n + m;
    model.sense = sense;
    model.A = la::SparseMatrixCSC::from_triplets(m, n, A_triplets);

    model.lower.resize(static_cast<size_t>(n + m));
    model.upper.resize(static_cast<size_t>(n + m));
    model.cost.resize(static_cast<size_t>(n + m));

    for (int j = 0; j < n; ++j) {
        model.lower[static_cast<size_t>(j)] = col_lb[static_cast<size_t>(j)];
        model.upper[static_cast<size_t>(j)] = col_ub[static_cast<size_t>(j)];
        model.cost[static_cast<size_t>(j)] = (sense == ObjSense::kMaximize) ? -c[static_cast<size_t>(j)] : c[static_cast<size_t>(j)];
    }

    for (int i = 0; i < m; ++i) {
        model.lower[static_cast<size_t>(n + i)] = -row_ub[static_cast<size_t>(i)];
        model.upper[static_cast<size_t>(n + i)] = -row_lb[static_cast<size_t>(i)];
        model.cost[static_cast<size_t>(n + i)] = 0.0;
    }

    return model;
}

void test_textbook_2x2_lp() {
    // min -x1 - 2*x2
    // s.t.  x1 + x2 <= 4
    //       x1 + 3*x2 <= 9
    //       x1 >= 0, x2 >= 0
    // Optimal: x1 = 1.5, x2 = 2.5, obj = -6.5
    std::vector<la::Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 3.0}
    };
    std::vector<double> c = {-1.0, -2.0};
    std::vector<double> row_lb = {-1e20, -1e20};
    std::vector<double> row_ub = {4.0, 9.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1e20, 1e20};

    // 1. Dual Simplex solve
    SimplexModel model1 = make_model(2, 2, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core1(model1);
    SimplexResult res_dual = core1.solve_dual();

    TEST_ASSERT(res_dual.status == SolveStatus::kOptimal, "Dual Simplex status == Optimal");
    TEST_ASSERT(std::abs(res_dual.objective_value - (-6.5)) < 1e-6, "Dual Simplex objective == -6.5");
    TEST_ASSERT(std::abs(res_dual.col_value[0] - 1.5) < 1e-6, "Dual Simplex x1 == 1.5");
    TEST_ASSERT(std::abs(res_dual.col_value[1] - 2.5) < 1e-6, "Dual Simplex x2 == 2.5");

    // 2. Primal Simplex solve
    SimplexModel model2 = make_model(2, 2, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core2(model2);
    SimplexResult res_primal = core2.solve_primal();

    TEST_ASSERT(res_primal.status == SolveStatus::kOptimal, "Primal Simplex status == Optimal");
    TEST_ASSERT(std::abs(res_primal.objective_value - (-6.5)) < 1e-6, "Primal Simplex objective == -6.5");
    TEST_ASSERT(std::abs(res_primal.col_value[0] - 1.5) < 1e-6, "Primal Simplex x1 == 1.5");
    TEST_ASSERT(std::abs(res_primal.col_value[1] - 2.5) < 1e-6, "Primal Simplex x2 == 2.5");

}

void test_bounded_3x3_lp() {
    // min 2*x1 + 3*x2 - x3
    // s.t. 1 <= x1 + x2 <= 5
    //      2 <= x2 + x3 <= 6
    //      0 <= x1 + x3 <= 4
    //      0 <= x1 <= 3, 0 <= x2 <= 4, 0 <= x3 <= 2
    std::vector<la::Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 1, 1.0}, {1, 2, 1.0},
        {2, 0, 1.0}, {2, 2, 1.0}
    };
    std::vector<double> c = {2.0, 3.0, -1.0};
    std::vector<double> row_lb = {1.0, 2.0, 0.0};
    std::vector<double> row_ub = {5.0, 6.0, 4.0};
    std::vector<double> col_lb = {0.0, 0.0, 0.0};
    std::vector<double> col_ub = {3.0, 4.0, 2.0};

    SimplexModel model = make_model(3, 3, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core(model);
    SimplexResult res = core.solve_dual();

    TEST_ASSERT(res.status == SolveStatus::kOptimal, "Bounded LP status == Optimal");

    // Primal feasibility check
    for (int j = 0; j < 3; ++j) {
        TEST_ASSERT(res.col_value[j] >= col_lb[j] - 1e-6, "Primal col lb");
        TEST_ASSERT(res.col_value[j] <= col_ub[j] + 1e-6, "Primal col ub");
    }
    for (int i = 0; i < 3; ++i) {
        TEST_ASSERT(res.row_value[i] >= row_lb[i] - 1e-6, "Primal row lb");
        TEST_ASSERT(res.row_value[i] <= row_ub[i] + 1e-6, "Primal row ub");
    }
}

void test_infeasible_lp_farkas() {
    // Infeasible system:
    // x1 + x2 <= 1
    // x1 + x2 >= 2
    // x1, x2 >= 0
    std::vector<la::Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, 1.0}
    };
    std::vector<double> c = {1.0, 1.0};
    std::vector<double> row_lb = {-1e20, 2.0};
    std::vector<double> row_ub = {1.0, 1e20};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1e20, 1e20};

    SimplexModel model = make_model(2, 2, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core(model);
    SimplexResult res = core.solve_dual();

    TEST_ASSERT(res.status == SolveStatus::kInfeasible, "Infeasible LP detected by dual simplex");
    TEST_ASSERT(res.certificate_type == "farkas", "Farkas certificate reported");
    TEST_ASSERT(!res.certificate_vector.empty(), "Farkas certificate vector not empty");

    // Primal simplex should also detect infeasibility
    SimplexModel model2 = make_model(2, 2, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core2(model2);
    SimplexResult res2 = core2.solve_primal();
    TEST_ASSERT(res2.status == SolveStatus::kInfeasible, "Infeasible LP detected by primal simplex");
}

void test_unbounded_lp_ray() {
    // min -2*x1 - x2
    // s.t. x1 - x2 <= 2
    //      x1, x2 >= 0
    std::vector<la::Triplet> triplets = {
        {0, 0, 1.0}, {0, 1, -1.0}
    };
    std::vector<double> c = {-2.0, -1.0};
    std::vector<double> row_lb = {-1e20};
    std::vector<double> row_ub = {2.0};
    std::vector<double> col_lb = {0.0, 0.0};
    std::vector<double> col_ub = {1e20, 1e20};

    SimplexModel model = make_model(1, 2, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core(model);
    SimplexResult res = core.solve_primal();

    TEST_ASSERT(res.status == SolveStatus::kUnbounded, "Unbounded LP detected by primal simplex");
    TEST_ASSERT(res.certificate_type == "ray", "Ray certificate reported");
    TEST_ASSERT(!res.certificate_vector.empty(), "Ray vector not empty");

    // Verify ray direction d: d >= 0 and cᵀ d < 0
    const auto& d = res.certificate_vector;
    TEST_ASSERT(d[0] >= -1e-7 && d[1] >= -1e-7, "Ray direction d >= 0");
    double c_dot_d = c[0] * d[0] + c[1] * d[1];
    TEST_ASSERT(c_dot_d < -1e-6, "Ray has negative cost cᵀ d < 0");
}

void test_degenerate_cycling_lp() {
    // Classic Beale cycling example:
    // min -0.75*x1 + 20*x2 - 0.5*x3 + 6*x4
    // s.t. 0.25*x1 - 8*x2 - x3 + 9*x4 <= 0
    //      0.5*x1 - 12*x2 - 0.5*x3 + 3*x4 <= 0
    //      x3 <= 1
    //      x1..x4 >= 0
    // Solves to optimal without cycling thanks to Harris ratio test.
    std::vector<la::Triplet> triplets = {
        {0, 0, 0.25}, {0, 1, -8.0}, {0, 2, -1.0}, {0, 3, 9.0},
        {1, 0, 0.5},  {1, 1, -12.0}, {1, 2, -0.5}, {1, 3, 3.0},
        {2, 2, 1.0}
    };
    std::vector<double> c = {-0.75, 20.0, -0.5, 6.0};
    std::vector<double> row_lb = {-1e20, -1e20, -1e20};
    std::vector<double> row_ub = {0.0, 0.0, 1.0};
    std::vector<double> col_lb = {0.0, 0.0, 0.0, 0.0};
    std::vector<double> col_ub = {1e20, 1e20, 1e20, 1e20};

    SimplexModel model = make_model(3, 4, triplets, c, row_lb, row_ub, col_lb, col_ub);
    SimplexCore core(model);
    SimplexResult res = core.solve_primal(500);

    TEST_ASSERT(res.status == SolveStatus::kOptimal, "Degenerate Beale LP status == Optimal (no cycling)");
    TEST_ASSERT(res.iterations < 20, "Solved in few iterations without cycling");
    TEST_ASSERT(std::abs(res.objective_value - (-1.25)) < 1e-4, "Optimal Beale objective == -1.25");
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "  INDUS-OPT / SIDDHANTA: Phase 2 Simplex Engine Test\n";
    std::cout << "  Bounded Revised Dual & Primal Simplex Verification\n";
    std::cout << "=========================================================\n";

    RUN_TEST(test_textbook_2x2_lp);
    RUN_TEST(test_bounded_3x3_lp);
    RUN_TEST(test_infeasible_lp_farkas);
    RUN_TEST(test_unbounded_lp_ray);
    RUN_TEST(test_degenerate_cycling_lp);

    std::cout << "---------------------------------------------------------\n";
    std::cout << "Total Tests Passed: " << g_tests_passed << "\n";
    std::cout << "Total Tests Failed: " << g_tests_failed << "\n";
    std::cout << "---------------------------------------------------------\n";

    if (g_tests_failed == 0) {
        std::cout << ">>> PHASE 2 GATE CHECK: ALL SIMPLEX TESTS PASSED! <<<\n";
        return 0;
    } else {
        std::cout << ">>> PHASE 2 GATE CHECK: FAILED <<<\n";
        return 1;
    }
}
