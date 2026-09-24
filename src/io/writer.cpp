#include "indus/io.hpp"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace indus::io {

namespace {

std::string status_to_str(SolveStatus status) {
    switch (status) {
        case SolveStatus::kOptimal: return "OPTIMAL";
        case SolveStatus::kInfeasible: return "INFEASIBLE";
        case SolveStatus::kUnbounded: return "UNBOUNDED";
        case SolveStatus::kInfeasibleOrUnbounded: return "INFEASIBLE_OR_UNBOUNDED";
        case SolveStatus::kFeasible: return "FEASIBLE";
        case SolveStatus::kIterationLimit: return "ITERATION_LIMIT";
        case SolveStatus::kTimeLimit: return "TIME_LIMIT";
        case SolveStatus::kNodeLimit: return "NODE_LIMIT";
        case SolveStatus::kNumericalError: return "NUMERICAL_ERROR";
        case SolveStatus::kModelError: return "MODEL_ERROR";
        case SolveStatus::kNotSolved: return "NOT_SOLVED";
        default: return "UNKNOWN";
    }
}

std::string basis_status_to_str(BasisStatus bs) {
    switch (bs) {
        case BasisStatus::kBasic: return "BASIC";
        case BasisStatus::kAtLower: return "AT_LOWER";
        case BasisStatus::kAtUpper: return "AT_UPPER";
        case BasisStatus::kFixed: return "FIXED";
        case BasisStatus::kNonbasicFree: return "FREE";
        default: return "UNKNOWN";
    }
}

} // namespace

#ifndef INDUS_GIT_COMMIT
#define INDUS_GIT_COMMIT "clean-room-v1.0"
#endif

void write_solution(const Solution& solution, const Model& model, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open solution file for writing: " + filepath);
    }

    file << std::setprecision(17);

    file << "# Solution produced by SIDDHANTA (INDUS-OPT)\n";
    file << "# Model: " << (model.name.empty() ? "unnamed" : model.name) << "\n";
    file << "# Rows: " << model.num_rows << " Cols: " << model.num_cols << " Nonzeros: " << model.A.nnz() << "\n";
    file << "# Algorithm: " << solution.algorithm_used << "\n";
    file << "# Git Commit: " << INDUS_GIT_COMMIT << "\n";
    file << "# Status: " << status_to_str(solution.status) << "\n";
    file << "# Message: " << solution.status_message << "\n";
    file << "# Objective: " << solution.objective_value << "\n";
    file << "# Best Dual Bound: " << solution.best_dual_bound << "\n";
    file << "# Iterations: " << solution.iterations << "\n";
    file << "# Nodes: " << solution.nodes << "\n";
    file << "# Solve Time: " << solution.solve_time_seconds << " s\n";
    file << "# Primal Feasible: " << (solution.quality.is_primal_feasible ? "true" : "false") << "\n";
    file << "# Dual Feasible: " << (solution.quality.is_dual_feasible ? "true" : "false") << "\n";
    file << "# Max Primal Violation: " << solution.quality.max_primal_violation << "\n";
    file << "# Max Dual Violation: " << solution.quality.max_dual_violation << "\n";
    file << "\n";

    file << "# Columns (Variables)\n";
    file << "# Name  Value  ReducedCost  Status\n";
    const size_t num_cols = model.col_names.size();
    for (size_t j = 0; j < num_cols; ++j) {
        const std::string& name = model.col_names[j];
        const double val = (j < solution.col_value.size()) ? solution.col_value[j] : 0.0;
        const double red_cost = (j < solution.col_dual.size()) ? solution.col_dual[j] : 0.0;
        const std::string bstat = (j < solution.col_basis_status.size())
                                      ? basis_status_to_str(solution.col_basis_status[j])
                                      : "UNKNOWN";
        file << name << " " << val << " " << red_cost << " " << bstat << "\n";
    }

    file << "\n# Rows (Constraints)\n";
    file << "# Name  Activity  DualMultiplier  Status\n";
    const size_t num_rows = model.row_names.size();
    for (size_t i = 0; i < num_rows; ++i) {
        const std::string& name = model.row_names[i];
        const double val = (i < solution.row_value.size()) ? solution.row_value[i] : 0.0;
        const double dual = (i < solution.row_dual.size()) ? solution.row_dual[i] : 0.0;
        const std::string bstat = (i < solution.row_basis_status.size())
                                      ? basis_status_to_str(solution.row_basis_status[i])
                                      : "UNKNOWN";
        file << name << " " << val << " " << dual << " " << bstat << "\n";
    }

    if (solution.certificate_type != "none" && !solution.certificate_vector.empty()) {
        file << "\n# Certificate: " << solution.certificate_type << "\n";
        for (size_t k = 0; k < solution.certificate_vector.size(); ++k) {
            file << solution.certificate_vector[k] << " ";
        }
        file << "\n";
    }
}

void write_json(const Solution& solution, const Model& model, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON telemetry file for writing: " + filepath);
    }

    file << std::setprecision(17);

    file << "{\n";
    file << "  \"model_name\": \"" << (model.name.empty() ? "unnamed" : model.name) << "\",\n";
    file << "  \"num_rows\": " << model.num_rows << ",\n";
    file << "  \"num_cols\": " << model.num_cols << ",\n";
    file << "  \"num_nonzeros\": " << model.A.nnz() << ",\n";
    file << "  \"algorithm\": \"" << solution.algorithm_used << "\",\n";
    file << "  \"git_commit\": \"" << INDUS_GIT_COMMIT << "\",\n";
    file << "  \"status\": \"" << status_to_str(solution.status) << "\",\n";
    file << "  \"status_message\": \"" << solution.status_message << "\",\n";
    file << "  \"objective_value\": " << solution.objective_value << ",\n";
    file << "  \"best_dual_bound\": " << solution.best_dual_bound << ",\n";
    file << "  \"relative_gap\": " << solution.relative_gap << ",\n";
    file << "  \"iterations\": " << solution.iterations << ",\n";
    file << "  \"nodes\": " << solution.nodes << ",\n";
    file << "  \"solve_time_seconds\": " << solution.solve_time_seconds << ",\n";

    file << "  \"quality\": {\n";
    file << "    \"is_primal_feasible\": " << (solution.quality.is_primal_feasible ? "true" : "false") << ",\n";
    file << "    \"is_dual_feasible\": " << (solution.quality.is_dual_feasible ? "true" : "false") << ",\n";
    file << "    \"max_primal_violation\": " << solution.quality.max_primal_violation << ",\n";
    file << "    \"max_dual_violation\": " << solution.quality.max_dual_violation << ",\n";
    file << "    \"max_complementarity_violation\": " << solution.quality.max_complementarity_violation << ",\n";
    file << "    \"duality_gap\": " << solution.quality.duality_gap << "\n";
    file << "  },\n";

    file << "  \"presolve\": {\n";
    file << "    \"presolved_rows\": " << solution.presolve_num_rows << ",\n";
    file << "    \"presolved_cols\": " << solution.presolve_num_cols << ",\n";
    file << "    \"total_reductions\": " << solution.presolve_total_reductions << ",\n";
    file << "    \"doubleton_reductions\": " << solution.presolve_doubleton_reductions << "\n";
    file << "  },\n";

    file << "  \"certificate\": {\n";
    file << "    \"type\": \"" << solution.certificate_type << "\",\n";
    file << "    \"values\": [";
    for (size_t k = 0; k < solution.certificate_vector.size(); ++k) {
        file << solution.certificate_vector[k];
        if (k + 1 < solution.certificate_vector.size()) file << ", ";
    }
    file << "]\n";
    file << "  },\n";

    file << "  \"variables\": [\n";
    const size_t num_cols = model.col_names.size();
    for (size_t j = 0; j < num_cols; ++j) {
        const std::string& name = model.col_names[j];
        const double val = (j < solution.col_value.size()) ? solution.col_value[j] : 0.0;
        const double red_cost = (j < solution.col_dual.size()) ? solution.col_dual[j] : 0.0;
        const std::string bstat = (j < solution.col_basis_status.size())
                                      ? basis_status_to_str(solution.col_basis_status[j])
                                      : "UNKNOWN";
        file << "    {\"name\": \"" << name << "\", \"value\": " << val
             << ", \"reduced_cost\": " << red_cost << ", \"status\": \"" << bstat << "\"}";
        if (j + 1 < num_cols) file << ",";
        file << "\n";
    }
    file << "  ],\n";

    file << "  \"constraints\": [\n";
    const size_t num_rows = model.row_names.size();
    for (size_t i = 0; i < num_rows; ++i) {
        const std::string& name = model.row_names[i];
        const double val = (i < solution.row_value.size()) ? solution.row_value[i] : 0.0;
        const double dual = (i < solution.row_dual.size()) ? solution.row_dual[i] : 0.0;
        const std::string bstat = (i < solution.row_basis_status.size())
                                      ? basis_status_to_str(solution.row_basis_status[i])
                                      : "UNKNOWN";
        file << "    {\"name\": \"" << name << "\", \"activity\": " << val
             << ", \"dual\": " << dual << ", \"status\": \"" << bstat << "\"}";
        if (i + 1 < num_rows) file << ",";
        file << "\n";
    }
    file << "  ]\n";

    file << "}\n";
}

} // namespace indus::io
