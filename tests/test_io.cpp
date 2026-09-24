#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/tolerances.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <filesystem>

namespace {

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

void test_mps_parser_crude_blend() {
    std::cout << "--- [Test 1] MPS Reader: crude_blend.mps ---" << std::endl;
    const std::string mps_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps");
    indus::Model model = indus::io::read_mps(mps_path);

    std::cout << "Model Name: " << model.name << std::endl;
    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    assert(model.num_rows == 3);
    assert(model.num_cols == 3);
    assert(model.sense == indus::ObjSense::kMaximize);

    // Columns: AL, BN, MU
    assert(model.col_names.size() == 3);
    assert(model.col_names[0] == "AL");
    assert(model.col_names[1] == "BN");
    assert(model.col_names[2] == "MU");

    // Objective: 2.40 AL + 1.60 BN + 1.64 MU
    assert(std::abs(model.c[0] - 2.40) < 1e-6);
    assert(std::abs(model.c[1] - 1.60) < 1e-6);
    assert(std::abs(model.c[2] - 1.64) < 1e-6);

    // Bounds:
    // AL: lower 10.0, upper inf
    assert(std::abs(model.col_lower[0] - 10.0) < 1e-6);
    assert(model.col_upper[0] > 1e19);

    // BN: lower 0.0, upper 45.0
    assert(std::abs(model.col_lower[1] - 0.0) < 1e-6);
    assert(std::abs(model.col_upper[1] - 45.0) < 1e-6);

    // MU: lower 0.0, upper 60.0
    assert(std::abs(model.col_lower[2] - 0.0) < 1e-6);
    assert(std::abs(model.col_upper[2] - 60.0) < 1e-6);

    // Rows: THRUPUT (G with RANGES 30 -> 90 <= ... <= 120), DIESEL (E 40), SULPHUR (L 0 -> <= 0)
    assert(model.row_names.size() == 3);
    assert(model.row_names[0] == "THRUPUT");
    assert(std::abs(model.row_lower[0] - 90.0) < 1e-6);
    assert(std::abs(model.row_upper[0] - 120.0) < 1e-6);

    assert(model.row_names[1] == "DIESEL");
    assert(std::abs(model.row_lower[1] - 40.0) < 1e-6);
    assert(std::abs(model.row_upper[1] - 40.0) < 1e-6);

    assert(model.row_names[2] == "SULPHUR");
    assert(model.row_lower[2] < -1e19);
    assert(std::abs(model.row_upper[2] - 0.0) < 1e-6);

    std::cout << "PASS: crude_blend.mps parsed correctly with RANGES, BOUNDS, and OBJSENSE." << std::endl;
}

void test_lp_parser_crude_blend() {
    std::cout << "--- [Test 2] LP Reader: crude_blend.lp ---" << std::endl;
    const std::string lp_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp");
    indus::Model model = indus::io::read_lp(lp_path);

    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    assert(model.num_rows == 3);
    assert(model.num_cols == 3);
    assert(model.sense == indus::ObjSense::kMaximize);

    // Compare with MPS model
    indus::Model mps_model = indus::io::read_mps(find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"));
    assert(model.num_rows == mps_model.num_rows);
    assert(model.num_cols == mps_model.num_cols);

    // Check bounds
    for (int j = 0; j < model.num_cols; ++j) {
        assert(std::abs(model.c[j] - mps_model.c[j]) < 1e-6);
        assert(std::abs(model.col_lower[j] - mps_model.col_lower[j]) < 1e-6);
        if (mps_model.col_upper[j] < 1e19) {
            assert(std::abs(model.col_upper[j] - mps_model.col_upper[j]) < 1e-6);
        } else {
            assert(model.col_upper[j] > 1e19);
        }
    }

    for (int i = 0; i < model.num_rows; ++i) {
        if (mps_model.row_lower[i] > -1e19) {
            assert(std::abs(model.row_lower[i] - mps_model.row_lower[i]) < 1e-6);
        } else {
            assert(model.row_lower[i] < -1e19);
        }

        if (mps_model.row_upper[i] < 1e19) {
            assert(std::abs(model.row_upper[i] - mps_model.row_upper[i]) < 1e-6);
        } else {
            assert(model.row_upper[i] > 1e19);
        }
    }

    std::cout << "PASS: crude_blend.lp matches crude_blend.mps exactly." << std::endl;
}

void test_solve_crude_blend_and_export() {
    std::cout << "--- [Test 3] End-to-End Solve: crude_blend.mps & IO Writers ---" << std::endl;
    indus::Model model = indus::io::read_mps(find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"));

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    std::cout << "Solve Status: " << static_cast<int>(sol.status) << " (" << sol.status_message << ")" << std::endl;
    std::cout << "Iterations: " << sol.iterations << std::endl;
    std::cout << "Optimal Objective: " << sol.objective_value << std::endl;
    for (size_t j = 0; j < model.col_names.size(); ++j) {
        std::cout << "  " << model.col_names[j] << " = " << sol.col_value[j]
                  << " (dual: " << sol.col_dual[j] << ")" << std::endl;
    }
    for (size_t i = 0; i < model.row_names.size(); ++i) {
        std::cout << "  Row " << model.row_names[i] << " = " << sol.row_value[i]
                  << " (shadow price: " << sol.row_dual[i] << ")" << std::endl;
    }

    assert(sol.status == indus::SolveStatus::kOptimal);
    assert(sol.quality.is_primal_feasible);
    assert(sol.quality.is_dual_feasible);

    // Test export
    std::filesystem::create_directories("tmp");
    const std::string sol_path = "tmp/crude_blend.sol";
    const std::string json_path = "tmp/crude_blend.json";

    indus::io::write_solution(sol, model, sol_path);
    indus::io::write_json(sol, model, json_path);

    assert(std::filesystem::exists(sol_path));
    assert(std::filesystem::file_size(sol_path) > 0);
    assert(std::filesystem::exists(json_path));
    assert(std::filesystem::file_size(json_path) > 0);

    std::cout << "PASS: crude_blend solved to optimality and exported to .sol and .json." << std::endl;
}

void test_netlib_afiro() {
    std::cout << "--- [Test 4] Netlib Benchmark: afiro.mps ---" << std::endl;
    const std::string afiro_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps");
    indus::Model model = indus::io::read_mps(afiro_path);

    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    assert(model.num_rows == 27); // 27 constraints (excluding N row)
    assert(model.num_cols == 32);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    std::cout << "Solve Status: " << static_cast<int>(sol.status) << " (" << sol.status_message << ")" << std::endl;
    std::cout << "Iterations: " << sol.iterations << std::endl;
    std::cout << "Computed Objective: " << sol.objective_value << std::endl;

    const double published_optimal = -464.75314286;
    const double rel_err = std::abs(sol.objective_value - published_optimal) / std::abs(published_optimal);
    std::cout << "Published Optimal:  " << published_optimal << std::endl;
    std::cout << "Relative Error:     " << rel_err << std::endl;

    assert(sol.status == indus::SolveStatus::kOptimal);
    assert(rel_err < 1e-5);
    assert(sol.quality.is_primal_feasible);

    std::cout << "PASS: Netlib AFIRO solved and matches published benchmark (-464.75314286)." << std::endl;
}

void test_netlib_blend_parse() {
    std::cout << "--- [Test 5] Netlib Benchmark Parsing: blend.mps ---" << std::endl;
    const std::string blend_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend.mps");
    indus::Model model = indus::io::read_mps(blend_path);

    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    int obj_nnz = 0;
    for (double cval : model.c) {
        if (std::abs(cval) > 1e-12) ++obj_nnz;
    }
    std::cout << "A nnz: " << model.A.nnz() << ", obj nnz: " << obj_nnz
              << ", total MPS entries: " << (model.A.nnz() + obj_nnz) << std::endl;

    assert(model.num_rows == 74); // 74 constraints (excluding N row)
    assert(model.num_cols == 83);
    assert(model.A.nnz() + obj_nnz == 521);

    std::cout << "PASS: Netlib BLEND parsed correctly (521 total matrix + objective entries)." << std::endl;
}

} // namespace

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RUNNING PHASE 3 IO & INTEGRATION TESTS  " << std::endl;
    std::cout << "========================================" << std::endl;

    test_mps_parser_crude_blend();
    test_lp_parser_crude_blend();
    test_solve_crude_blend_and_export();
    test_netlib_afiro();
    test_netlib_blend_parse();

    std::cout << "\nALL PHASE 3 TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
