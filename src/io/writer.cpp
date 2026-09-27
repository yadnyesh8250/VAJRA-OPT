#include "indus/io.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
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

// Produce a safe JSON string (no surrounding quotes added here).
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20u) {
                    // other control characters as \uXXXX
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
                break;
        }
    }
    return out;
}

// Get column name with automatic fallback for unnamed models.
std::string col_name(const Model& model, int j) {
    const auto idx = static_cast<size_t>(j);
    if (idx < model.col_names.size() && !model.col_names[idx].empty()) return model.col_names[idx];
    return "c" + std::to_string(j);
}

// Get row name with automatic fallback for unnamed models.
std::string row_name(const Model& model, int i) {
    const auto idx = static_cast<size_t>(i);
    if (idx < model.row_names.size() && !model.row_names[idx].empty()) return model.row_names[idx];
    return "r" + std::to_string(i);
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
    file << "# Termination Reason: " << (solution.termination_reason.empty() ? solution.status_message : solution.termination_reason) << "\n";
    file << "# Objective: " << solution.objective_value << "\n";
    file << "# Best Dual Bound: " << solution.best_dual_bound << "\n";
    file << "# Has Incumbent: " << (solution.has_incumbent ? "true" : "false") << "\n";
    file << "# Absolute Gap: " << solution.absolute_gap << "\n";
    file << "# Relative Gap: " << solution.relative_gap << "\n";
    file << "# Iterations: " << solution.iterations << "\n";
    file << "# Nodes: " << solution.nodes << "\n";
    file << "# Open Nodes: " << solution.open_nodes << "\n";
    file << "# Search Completed: " << (solution.search_completed ? "true" : "false") << "\n";
    file << "# Solve Time: " << solution.solve_time_seconds << " s\n";
    file << "# Primal Feasible: " << (solution.quality.is_primal_feasible ? "true" : "false") << "\n";
    file << "# Dual Feasible: " << (solution.quality.is_dual_feasible ? "true" : "false") << "\n";
    file << "# Max Primal Violation: " << solution.quality.max_primal_violation << "\n";
    file << "# Max Dual Violation: " << solution.quality.max_dual_violation << "\n";
    file << "\n";

    file << "# Columns (Variables)\n";
    file << "# Name  Value  ReducedCost  Status\n";
    for (int j = 0; j < model.num_cols; ++j) {
        const std::string name = col_name(model, j);
        const size_t idx = static_cast<size_t>(j);
        const double val = (idx < solution.col_value.size()) ? solution.col_value[idx] : 0.0;
        const double red_cost = (idx < solution.col_dual.size()) ? solution.col_dual[idx] : 0.0;
        const std::string bstat = (idx < solution.col_basis_status.size())
                                      ? basis_status_to_str(solution.col_basis_status[idx])
                                      : "UNKNOWN";
        file << name << " " << val << " " << red_cost << " " << bstat << "\n";
    }

    file << "\n# Rows (Constraints)\n";
    file << "# Name  Activity  DualMultiplier  Status\n";
    for (int i = 0; i < model.num_rows; ++i) {
        const std::string name = row_name(model, i);
        const size_t idx = static_cast<size_t>(i);
        const double val = (idx < solution.row_value.size()) ? solution.row_value[idx] : 0.0;
        const double dual = (idx < solution.row_dual.size()) ? solution.row_dual[idx] : 0.0;
        const std::string bstat = (idx < solution.row_basis_status.size())
                                      ? basis_status_to_str(solution.row_basis_status[idx])
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
    file << "  \"model_name\": \"" << json_escape(model.name.empty() ? "unnamed" : model.name) << "\",\n";
    file << "  \"num_rows\": " << model.num_rows << ",\n";
    file << "  \"num_cols\": " << model.num_cols << ",\n";
    file << "  \"num_nonzeros\": " << model.A.nnz() << ",\n";
    file << "  \"algorithm\": \"" << json_escape(solution.algorithm_used) << "\",\n";
    file << "  \"git_commit\": \"" << json_escape(INDUS_GIT_COMMIT) << "\",\n";
    file << "  \"status\": \"" << status_to_str(solution.status) << "\",\n";
    file << "  \"status_message\": \"" << json_escape(solution.status_message) << "\",\n";
    file << "  \"termination_reason\": \"" << json_escape(solution.termination_reason.empty() ? solution.status_message : solution.termination_reason) << "\",\n";
    file << "  \"objective_value\": " << solution.objective_value << ",\n";
    file << "  \"best_dual_bound\": " << solution.best_dual_bound << ",\n";
    file << "  \"has_incumbent\": " << (solution.has_incumbent ? "true" : "false") << ",\n";
    file << "  \"absolute_gap\": " << solution.absolute_gap << ",\n";
    file << "  \"relative_gap\": " << solution.relative_gap << ",\n";
    file << "  \"iterations\": " << solution.iterations << ",\n";
    file << "  \"nodes\": " << solution.nodes << ",\n";
    file << "  \"open_nodes\": " << solution.open_nodes << ",\n";
    file << "  \"search_completed\": " << (solution.search_completed ? "true" : "false") << ",\n";
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
    file << "    \"type\": \"" << json_escape(solution.certificate_type) << "\",\n";
    file << "    \"values\": [";
    for (size_t k = 0; k < solution.certificate_vector.size(); ++k) {
        file << solution.certificate_vector[k];
        if (k + 1 < solution.certificate_vector.size()) file << ", ";
    }
    file << "]\n";
    file << "  },\n";

    file << "  \"variables\": [\n";
    for (int j = 0; j < model.num_cols; ++j) {
        const std::string name = col_name(model, j);
        const size_t idx = static_cast<size_t>(j);
        const double val = (idx < solution.col_value.size()) ? solution.col_value[idx] : 0.0;
        const double red_cost = (idx < solution.col_dual.size()) ? solution.col_dual[idx] : 0.0;
        const std::string bstat = (idx < solution.col_basis_status.size())
                                      ? basis_status_to_str(solution.col_basis_status[idx])
                                      : "UNKNOWN";
        file << "    {\"name\": \"" << json_escape(name) << "\", \"value\": " << val
             << ", \"reduced_cost\": " << red_cost << ", \"status\": \"" << bstat << "\"}";
        if (j + 1 < model.num_cols) file << ",";
        file << "\n";
    }
    file << "  ],\n";

    file << "  \"constraints\": [\n";
    for (int i = 0; i < model.num_rows; ++i) {
        const std::string name = row_name(model, i);
        const size_t idx = static_cast<size_t>(i);
        const double val = (idx < solution.row_value.size()) ? solution.row_value[idx] : 0.0;
        const double dual = (idx < solution.row_dual.size()) ? solution.row_dual[idx] : 0.0;
        const std::string bstat = (idx < solution.row_basis_status.size())
                                      ? basis_status_to_str(solution.row_basis_status[idx])
                                      : "UNKNOWN";
        file << "    {\"name\": \"" << json_escape(name) << "\", \"activity\": " << val
             << ", \"dual\": " << dual << ", \"status\": \"" << bstat << "\"}";
        if (i + 1 < model.num_rows) file << ",";
        file << "\n";
    }
    file << "  ]\n";

    file << "}\n";
}

} // namespace indus::io
