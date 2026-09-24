#include "indus/verifier.hpp"
#include "indus/io.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: indus_verifier <model.mps|model.lp> <solution.sol> [--tol <tol>] [--json <report.json>]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const std::string sol_path = argv[2];
    double tol = 1e-4;
    std::string json_report_path;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tol" && i + 1 < argc) {
            tol = std::stod(argv[++i]);
        } else if (arg == "--json" && i + 1 < argc) {
            json_report_path = argv[++i];
        }
    }

    std::cout << "========================================================================\n";
    std::cout << "  SIDDHANTA (INDUS-OPT): INDEPENDENT SOLUTION VERIFIER (AUDIT TOOL)     \n";
    std::cout << "========================================================================\n";
    std::cout << "Model File:    " << model_path << "\n";
    std::cout << "Solution File: " << sol_path << "\n";
    std::cout << "Tolerance:     " << tol << "\n\n";

    indus::verifier::VerificationResult res;
    try {
        res = indus::verifier::verify_files(model_path, sol_path, tol);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Verification aborted due to exception: " << e.what() << "\n";
        return 1;
    }

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << std::left
              << std::setw(30) << "Reported Status:"
              << res.reported_status << "\n"
              << std::setw(30) << "Reported Objective:"
              << std::setprecision(12) << res.reported_objective << "\n"
              << std::setw(30) << "Recomputed Objective:"
              << std::setprecision(12) << res.recomputed_objective << "\n"
              << std::setw(30) << "Objective Discrepancy:"
              << std::scientific << res.objective_discrepancy << std::defaultfloat << " ("
              << (res.objective_matches ? "PASSED" : "FAILED") << ")\n"
              << std::setw(30) << "Variable Bounds Feasibility:"
              << (res.bounds_feasible ? "PASSED" : "FAILED")
              << " (max viol: " << std::scientific << res.max_bound_violation << std::defaultfloat << ")\n"
              << std::setw(30) << "Row Activity Feasibility:"
              << (res.primal_feasible ? "PASSED" : "FAILED")
              << " (max viol: " << std::scientific << res.max_row_violation << std::defaultfloat << ")\n"
              << std::setw(30) << "Dual / Reduced Cost Feasibility:"
              << (res.dual_feasible ? "PASSED" : "FAILED")
              << " (max viol: " << std::scientific << res.max_dual_violation << std::defaultfloat << ")\n"
              << std::setw(30) << "Complementary Slackness:"
              << (res.max_complementarity_violation <= tol ? "PASSED" : "FAILED")
              << " (max viol: " << std::scientific << res.max_complementarity_violation << std::defaultfloat << ")\n";
    std::cout << "------------------------------------------------------------------------\n";

    if (!res.violations.empty()) {
        std::cout << "Violations (" << res.violations.size() << "):\n";
        size_t shown = std::min(res.violations.size(), static_cast<size_t>(10));
        for (size_t i = 0; i < shown; ++i) {
            std::cout << "  - " << res.violations[i] << "\n";
        }
        if (res.violations.size() > shown) {
            std::cout << "  ... and " << (res.violations.size() - shown) << " more.\n";
        }
    }

    if (!json_report_path.empty()) {
        std::ofstream jf(json_report_path);
        jf << "{\n"
           << "  \"model\": \"" << model_path << "\",\n"
           << "  \"solution\": \"" << sol_path << "\",\n"
           << "  \"verified\": " << (res.passed ? "true" : "false") << ",\n"
           << "  \"primal_feasible\": " << (res.primal_feasible ? "true" : "false") << ",\n"
           << "  \"dual_feasible\": " << (res.dual_feasible ? "true" : "false") << ",\n"
           << "  \"objective_matches\": " << (res.objective_matches ? "true" : "false") << ",\n"
           << "  \"reported_objective\": " << std::setprecision(17) << res.reported_objective << ",\n"
           << "  \"recomputed_objective\": " << std::setprecision(17) << res.recomputed_objective << ",\n"
           << "  \"max_primal_violation\": " << res.max_primal_violation << ",\n"
           << "  \"max_dual_violation\": " << res.max_dual_violation << ",\n"
           << "  \"max_complementarity_violation\": " << res.max_complementarity_violation << "\n"
           << "}\n";
    }

    if (res.passed) {
        std::cout << "\n>>> [VERIFICATION SUCCESS] Solution satisfies all mathematical optimality certificates! <<<\n\n";
        return 0;
    } else {
        std::cout << "\n>>> [VERIFICATION FAILED] Solution violated mathematical optimality conditions! <<<\n\n";
        return 1;
    }
}
