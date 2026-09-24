#pragma once

#include <string>
#include <vector>
#include "indus/model.hpp"

namespace indus::verifier {

struct VerificationResult {
    bool passed = false;
    bool primal_feasible = false;
    bool dual_feasible = false;
    bool bounds_feasible = false;
    bool objective_matches = false;

    double max_primal_violation = 0.0;
    double max_bound_violation = 0.0;
    double max_row_violation = 0.0;
    double max_dual_violation = 0.0;
    double max_complementarity_violation = 0.0;
    double objective_discrepancy = 0.0;

    double reported_objective = 0.0;
    double recomputed_objective = 0.0;
    std::string reported_status;

    std::vector<std::string> violations;
    std::string summary;
};

// Verifies a loaded Model against a .sol solution file
VerificationResult verify_solution_file(const Model& model, const std::string& sol_filepath, double tol = 1e-6);

// Verifies model file (.mps or .lp) and .sol file completely independently
VerificationResult verify_files(const std::string& model_filepath, const std::string& sol_filepath, double tol = 1e-6);

// Verifies in-memory Model and Solution
VerificationResult verify_solution(const Model& model, const Solution& solution, double tol = 1e-6);

} // namespace indus::verifier
