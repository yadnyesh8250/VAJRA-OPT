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

    bool is_milp = false;
    bool integer_feasible = true;
    double max_integrality_violation = 0.0;
    bool optimality_proven = false;
    double mip_gap = 0.0;

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

// Verifies in-memory Model and Solution (audits integrality if model.has_integers())
VerificationResult verify_solution(const Model& model, const Solution& solution, double tol = 1e-6);

// Verifies MILP global optimality proof (incumbent exists, gap within tolerance, search completed)
bool verify_milp_optimality(const Solution& solution, double mip_tolerance = 1e-4);

} // namespace indus::verifier
