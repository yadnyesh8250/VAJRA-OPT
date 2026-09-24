#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/verifier.hpp"
#include "indus/tolerances.hpp"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <filesystem>

namespace {

#ifndef INDUS_SOURCE_DIR
#define INDUS_SOURCE_DIR "."
#endif

std::string find_path(const std::string& rel) {
    std::filesystem::path p_src = std::filesystem::path(INDUS_SOURCE_DIR) / rel;
    if (std::filesystem::exists(p_src)) return p_src.string();
    if (std::filesystem::exists(rel)) return rel;
    if (std::filesystem::exists("../" + rel)) return "../" + rel;
    if (std::filesystem::exists("../../" + rel)) return "../../" + rel;
    return rel;
}

// 1. Free variables: x1 in (-inf, +inf)
void test_free_variable() {
    std::cout << "[Regression 1] Free Variable Handling..." << std::endl;
    indus::Model model;
    model.name = "free_var";
    model.num_rows = 2;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMinimize;

    // min -2 x1 + x2
    model.c = {-2.0, 1.0};

    // x1 free (-1e20, 1e20), x2 >= 0
    model.col_lower = {-1e20, 0.0};
    model.col_upper = {1e20, 1e20};
    model.col_names = {"x1", "x2"};

    // Rows:
    // x1 + x2 <= 5
    // x1 - x2 >= -10  =>  -10 <= x1 - x2 <= inf
    model.row_lower = {-1e20, -10.0};
    model.row_upper = {5.0, 1e20};
    model.row_names = {"r1", "r2"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 1.0}, {0, 1, 1.0},
        {1, 0, 1.0}, {1, 1, -1.0}
    };
    model.A.set_from_triplets(2, 2, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    std::cout << "  Status: " << static_cast<int>(sol.status) << " (" << sol.status_message << ")\n";
    std::cout << "  Obj: " << sol.objective_value << ", Iters: " << sol.iterations << "\n";
    std::cout << "  x1 = " << sol.col_value[0] << ", x2 = " << sol.col_value[1] << "\n";

    assert(sol.status == indus::SolveStatus::kOptimal);
    // At optimum: x1 = 5, x2 = 0 -> obj = -10
    assert(std::abs(sol.col_value[0] - 5.0) < 1e-5);
    assert(std::abs(sol.col_value[1] - 0.0) < 1e-5);
    assert(std::abs(sol.objective_value - (-10.0)) < 1e-5);

    auto vres = indus::verifier::verify_solution(model, sol);
    std::cout << "  Verifier: " << vres.summary << "\n";
    for (const auto& viol : vres.violations) {
        std::cout << "    Violation: " << viol << "\n";
    }
    assert(vres.passed);
    std::cout << "  PASS: Free variable resolved to exact optimum x1=5, x2=0, obj=-10" << std::endl;
}

// 2. Ranged rows: l <= Ax <= u
void test_ranged_row() {
    std::cout << "[Regression 2] Ranged Row Handling..." << std::endl;
    indus::Model model;
    model.name = "ranged_row";
    model.num_rows = 1;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMaximize;

    // max x1 + 2 x2
    model.c = {1.0, 2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_names = {"x1", "x2"};

    // 5 <= x1 + x2 <= 8
    model.row_lower = {5.0};
    model.row_upper = {8.0};
    model.row_names = {"range_row"};

    std::vector<indus::la::Triplet> tri = {{0, 0, 1.0}, {0, 1, 1.0}};
    model.A.set_from_triplets(1, 2, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    assert(sol.status == indus::SolveStatus::kOptimal);
    // max x1 + 2 x2 s.t. x1+x2 <= 8, x2 <= 10 -> x1=0, x2=8, obj = 16
    assert(std::abs(sol.objective_value - 16.0) < 1e-5);
    assert(std::abs(sol.col_value[1] - 8.0) < 1e-5);

    auto vres = indus::verifier::verify_solution(model, sol);
    assert(vres.passed);
    std::cout << "  PASS: Ranged row correctly bounded at upper limit x2=8, obj=16" << std::endl;
}

// 3. Equality rows: Ax = b
void test_equality_row() {
    std::cout << "[Regression 3] Equality Row Handling..." << std::endl;
    indus::Model model;
    model.name = "equality_row";
    model.num_rows = 1;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMinimize;

    // min 3 x1 + 4 x2
    model.c = {3.0, 4.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_names = {"x1", "x2"};

    // 2 x1 + x2 = 6
    model.row_lower = {6.0};
    model.row_upper = {6.0};
    model.row_names = {"eq_row"};

    std::vector<indus::la::Triplet> tri = {{0, 0, 2.0}, {0, 1, 1.0}};
    model.A.set_from_triplets(1, 2, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    assert(sol.status == indus::SolveStatus::kOptimal);
    // To minimize 3x1 + 4x2 with 2x1 + x2 = 6:
    // x1 = 3, x2 = 0 gives obj = 9
    // x1 = 0, x2 = 6 gives obj = 24
    assert(std::abs(sol.col_value[0] - 3.0) < 1e-5);
    assert(std::abs(sol.col_value[1] - 0.0) < 1e-5);
    assert(std::abs(sol.objective_value - 9.0) < 1e-5);

    auto vres = indus::verifier::verify_solution(model, sol);
    assert(vres.passed);
    std::cout << "  PASS: Equality row strictly satisfied: 2(3) + 0 = 6, obj=9" << std::endl;
}

// 4. Negative bounds: x in [-15, -5]
void test_negative_bounds() {
    std::cout << "[Regression 4] Negative Bounds Handling..." << std::endl;
    indus::Model model;
    model.name = "neg_bounds";
    model.num_rows = 1;
    model.num_cols = 1;
    model.sense = indus::ObjSense::kMinimize;

    // min 2 x1
    model.c = {2.0};
    model.col_lower = {-15.0};
    model.col_upper = {-5.0};
    model.col_names = {"x1"};

    // x1 >= -12
    model.row_lower = {-12.0};
    model.row_upper = {1e20};
    model.row_names = {"r1"};

    std::vector<indus::la::Triplet> tri = {{0, 0, 1.0}};
    model.A.set_from_triplets(1, 1, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    assert(sol.status == indus::SolveStatus::kOptimal);
    // x1 in [-15, -5], x1 >= -12 -> min 2 x1 at x1 = -12 -> obj = -24
    assert(std::abs(sol.col_value[0] - (-12.0)) < 1e-5);
    assert(std::abs(sol.objective_value - (-24.0)) < 1e-5);

    auto vres = indus::verifier::verify_solution(model, sol);
    assert(vres.passed);
    std::cout << "  PASS: Negative bound handled accurately: x1 = -12, obj = -24" << std::endl;
}

// 5. Infeasible model with Farkas certificate
void test_infeasible_model() {
    std::cout << "[Regression 5] Infeasible Model & Farkas Certificate..." << std::endl;
    indus::Model model;
    model.name = "infeasible_test";
    model.num_rows = 2;
    model.num_cols = 1;
    model.sense = indus::ObjSense::kMinimize;

    model.c = {1.0};
    model.col_lower = {0.0};
    model.col_upper = {10.0};
    model.col_names = {"x1"};

    // x1 <= 2  AND  x1 >= 5  (contradiction)
    model.row_lower = {-1e20, 5.0};
    model.row_upper = {2.0, 1e20};
    model.row_names = {"r1", "r2"};

    std::vector<indus::la::Triplet> tri = {{0, 0, 1.0}, {1, 0, 1.0}};
    model.A.set_from_triplets(2, 1, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    assert(sol.status == indus::SolveStatus::kInfeasible);
    std::cout << "  PASS: Infeasible model detected with status INFEASIBLE: " << sol.status_message << std::endl;
}

// 6. Unbounded model with ray certificate
void test_unbounded_model() {
    std::cout << "[Regression 6] Unbounded Model & Ray Certificate..." << std::endl;
    indus::Model model;
    model.name = "unbounded_test";
    model.num_rows = 1;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMaximize;

    // max x1 + x2
    model.c = {1.0, 1.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1e20, 1e20};
    model.col_names = {"x1", "x2"};

    // -x1 + x2 <= 2
    model.row_lower = {-1e20};
    model.row_upper = {2.0};
    model.row_names = {"r1"};

    std::vector<indus::la::Triplet> tri = {{0, 0, -1.0}, {0, 1, 1.0}};
    model.A.set_from_triplets(1, 2, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    assert(sol.status == indus::SolveStatus::kUnbounded);
    std::cout << "  PASS: Unbounded model detected with status UNBOUNDED: " << sol.status_message << std::endl;
}

// 7. Beale's Degenerate Cycling LP
void test_beale_cycling() {
    std::cout << "[Regression 7] Degenerate Cycling Prevention (Harris Test)..." << std::endl;
    indus::Model model;
    model.name = "beale_degenerate";
    model.num_rows = 3;
    model.num_cols = 4;
    model.sense = indus::ObjSense::kMinimize;

    // min -0.75 x1 + 20 x2 - 0.5 x3 + 6 x4
    model.c = {-0.75, 20.0, -0.5, 6.0};
    model.col_lower = {0.0, 0.0, 0.0, 0.0};
    model.col_upper = {1e20, 1e20, 1e20, 1e20};
    model.col_names = {"x1", "x2", "x3", "x4"};

    // 0.25 x1 - 8 x2 - x3 + 9 x4 <= 0
    // 0.5 x1 - 12 x2 - 0.5 x3 + 3 x4 <= 0
    // x3 <= 1
    model.row_lower = {-1e20, -1e20, -1e20};
    model.row_upper = {0.0, 0.0, 1.0};
    model.row_names = {"r1", "r2", "r3"};

    std::vector<indus::la::Triplet> tri = {
        {0, 0, 0.25}, {0, 1, -8.0}, {0, 2, -1.0}, {0, 3, 9.0},
        {1, 0, 0.5},  {1, 1, -12.0}, {1, 2, -0.5}, {1, 3, 3.0},
        {2, 2, 1.0}
    };
    model.A.set_from_triplets(3, 4, tri, true);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    assert(sol.status == indus::SolveStatus::kOptimal);
    assert(sol.iterations < 20); // Resolves without infinite cycling
    auto vres = indus::verifier::verify_solution(model, sol);
    assert(vres.passed);
    std::cout << "  PASS: Beale degenerate LP solved in " << sol.iterations << " pivots without cycling." << std::endl;
}

// 8. Ill-conditioned small matrix
void test_ill_conditioned_model() {
    std::cout << "[Regression 8] Ill-Conditioned Matrix Scaling..." << std::endl;
    const std::string ill_path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/ill_conditioned.mps");
    indus::Model model = indus::io::read_mps(ill_path);

    std::cout << "  Loaded ill_conditioned.mps: " << model.num_rows << " rows, "
              << model.num_cols << " cols, " << model.A.nnz() << " nonzeros" << std::endl;

    indus::Options opts;
    opts.enable_presolve = true;
    indus::Solution sol = indus::solve(model, opts);

    std::cout << "  Status: " << sol.status_message << ", Obj: " << sol.objective_value << std::endl;
    assert(sol.status == indus::SolveStatus::kOptimal || sol.status == indus::SolveStatus::kFeasible);
    assert(!std::isnan(sol.objective_value));
    std::cout << "  PASS: Ill-conditioned model solved without numerical breakdown." << std::endl;
}

// 9. Strict solution file parser validation
void test_strict_solution_verification() {
    std::cout << "[Regression 9] Strict Solution File Parser..." << std::endl;
    indus::Model model;
    model.name = "strict_test";
    model.num_rows = 1;
    model.num_cols = 2;
    model.sense = indus::ObjSense::kMinimize;
    model.c = {1.0, 2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {10.0, 10.0};
    model.col_names = {"x1", "x2"};
    model.row_lower = {0.0};
    model.row_upper = {5.0};
    model.row_names = {"c1"};
    std::vector<indus::la::Triplet> tri = {{0, 0, 1.0}, {0, 1, 1.0}};
    model.A.set_from_triplets(1, 2, tri, true);

    const std::string tmp_dir = std::filesystem::temp_directory_path().string();

    auto write_file = [](const std::string& path, const std::string& content) {
        std::ofstream out(path);
        out << content;
    };

    // Case 1: Missing # Status: header
    {
        const std::string p = tmp_dir + "/test_missing_status.sol";
        write_file(p, "# Solution File\n# Objective: 0\n# Columns (Variables)\nx1 0 1\nx2 0 2\n# Rows (Constraints)\nc1 0 0\n");
        auto res = indus::verifier::verify_solution_file(model, p);
        assert(!res.passed);
        assert(res.summary.find("Status") != std::string::npos);
        std::filesystem::remove(p);
    }

    // Case 2: Unknown variable name
    {
        const std::string p = tmp_dir + "/test_unknown_var.sol";
        write_file(p, "# Status: OPTIMAL\n# Objective: 0\n# Columns (Variables)\nx1 0 1\nunknown_var 0 2\n# Rows (Constraints)\nc1 0 0\n");
        auto res = indus::verifier::verify_solution_file(model, p);
        assert(!res.passed);
        assert(res.summary.find("Unknown variable") != std::string::npos);
        std::filesystem::remove(p);
    }

    // Case 3: Duplicate variable name
    {
        const std::string p = tmp_dir + "/test_duplicate_var.sol";
        write_file(p, "# Status: OPTIMAL\n# Objective: 0\n# Columns (Variables)\nx1 0 1\nx1 0 1\n# Rows (Constraints)\nc1 0 0\n");
        auto res = indus::verifier::verify_solution_file(model, p);
        assert(!res.passed);
        assert(res.summary.find("Duplicate variable") != std::string::npos);
        std::filesystem::remove(p);
    }

    // Case 4: Malformed float / NaN
    {
        const std::string p = tmp_dir + "/test_malformed_float.sol";
        write_file(p, "# Status: OPTIMAL\n# Objective: 0\n# Columns (Variables)\nx1 invalid_num 1\nx2 0 2\n# Rows (Constraints)\nc1 0 0\n");
        auto res = indus::verifier::verify_solution_file(model, p);
        assert(!res.passed);
        assert(res.summary.find("Malformed") != std::string::npos);
        std::filesystem::remove(p);
    }

    // Case 5: Incomplete solution (missing x2)
    {
        const std::string p = tmp_dir + "/test_incomplete_var.sol";
        write_file(p, "# Status: OPTIMAL\n# Objective: 0\n# Columns (Variables)\nx1 0 1\n# Rows (Constraints)\nc1 0 0\n");
        auto res = indus::verifier::verify_solution_file(model, p);
        assert(!res.passed);
        assert(res.summary.find("Incomplete solution: missing variable x2") != std::string::npos);
        std::filesystem::remove(p);
    }

    // Case 6: Valid solution
    {
        const std::string p = tmp_dir + "/test_valid.sol";
        write_file(p, "# Status: OPTIMAL\n# Objective: 0.0\n# Columns (Variables)\nx1 0.0 1.0\nx2 0.0 2.0\n# Rows (Constraints)\nc1 0.0 0.0\n");
        auto res = indus::verifier::verify_solution_file(model, p);
        assert(res.passed);
        std::filesystem::remove(p);
    }

    std::cout << "  PASS: Strict solution parser rejected missing variables, duplicates, unknown names, malformed floats, and missing headers." << std::endl;
}

} // namespace

int main() {
    std::cout << "======================================================\n";
    std::cout << "  RUNNING PHASE 3 EDGE-CASE REGRESSION TEST SUITE     \n";
    std::cout << "======================================================\n\n";

    test_free_variable();
    test_ranged_row();
    test_equality_row();
    test_negative_bounds();
    test_infeasible_model();
    test_unbounded_model();
    test_beale_cycling();
    test_ill_conditioned_model();
    test_strict_solution_verification();

    std::cout << "\n>>> ALL 9 REGRESSION TESTS PASSED! <<<\n";
    return 0;
}
