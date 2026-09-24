#include "indus/verifier.hpp"
#include "indus/io.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iomanip>

namespace indus::verifier {

namespace {

std::vector<std::string> split_tokens(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string t;
    while (iss >> t) tokens.push_back(t);
    return tokens;
}

} // namespace

VerificationResult verify_solution(const Model& model, const Solution& solution, double tol) {
    VerificationResult res;
    res.reported_status = (solution.status == SolveStatus::kOptimal) ? "OPTIMAL" :
                          (solution.status == SolveStatus::kFeasible) ? "FEASIBLE" :
                          (solution.status == SolveStatus::kInfeasible) ? "INFEASIBLE" :
                          (solution.status == SolveStatus::kUnbounded) ? "UNBOUNDED" : "OTHER";
    res.reported_objective = solution.objective_value;

    const int n = model.num_cols;
    const int m = model.num_rows;

    if (static_cast<int>(solution.col_value.size()) != n) {
        res.violations.push_back("Primal solution vector size (" + std::to_string(solution.col_value.size()) +
                                 ") does not match model num_cols (" + std::to_string(n) + ")");
        res.passed = false;
        res.summary = "FAIL: Dimension mismatch";
        return res;
    }

    // 1. Check for NaN / Inf
    for (int j = 0; j < n; ++j) {
        const double xj = solution.col_value[static_cast<size_t>(j)];
        if (std::isnan(xj) || std::isinf(xj)) {
            res.violations.push_back("Variable " + model.col_names[static_cast<size_t>(j)] + " is NaN or Inf");
        }
    }
    if (std::isnan(solution.objective_value) || std::isinf(solution.objective_value)) {
        res.violations.push_back("Reported objective is NaN or Inf");
    }

    // 2. Bound Feasibility
    for (int j = 0; j < n; ++j) {
        const double xj = solution.col_value[static_cast<size_t>(j)];
        const double lj = model.col_lower[static_cast<size_t>(j)];
        const double uj = model.col_upper[static_cast<size_t>(j)];

        if (xj < lj - tol) {
            const double viol = lj - xj;
            res.max_bound_violation = std::max(res.max_bound_violation, viol);
            res.violations.push_back("Variable " + model.col_names[static_cast<size_t>(j)] +
                                     " below lower bound: " + std::to_string(xj) + " < " + std::to_string(lj));
        }
        if (xj > uj + tol) {
            const double viol = xj - uj;
            res.max_bound_violation = std::max(res.max_bound_violation, viol);
            res.violations.push_back("Variable " + model.col_names[static_cast<size_t>(j)] +
                                     " above upper bound: " + std::to_string(xj) + " > " + std::to_string(uj));
        }
    }
    res.bounds_feasible = (res.max_bound_violation <= tol);

    // 3. Row Activity & Constraint Feasibility
    std::vector<double> Ax(static_cast<size_t>(m), 0.0);
    model.A.multiply(solution.col_value, Ax);

    for (int i = 0; i < m; ++i) {
        const double ax = Ax[static_cast<size_t>(i)];
        const double li = model.row_lower[static_cast<size_t>(i)];
        const double ui = model.row_upper[static_cast<size_t>(i)];

        if (ax < li - tol) {
            const double viol = li - ax;
            res.max_row_violation = std::max(res.max_row_violation, viol);
            res.violations.push_back("Row " + model.row_names[static_cast<size_t>(i)] +
                                     " below lower bound: " + std::to_string(ax) + " < " + std::to_string(li));
        }
        if (ax > ui + tol) {
            const double viol = ax - ui;
            res.max_row_violation = std::max(res.max_row_violation, viol);
            res.violations.push_back("Row " + model.row_names[static_cast<size_t>(i)] +
                                     " above upper bound: " + std::to_string(ax) + " > " + std::to_string(ui));
        }
    }
    res.max_primal_violation = std::max(res.max_bound_violation, res.max_row_violation);
    res.primal_feasible = (res.max_primal_violation <= tol);

    // 4. Objective Recomputation
    double recomputed_cTx = model.objective_offset;
    for (int j = 0; j < n; ++j) {
        recomputed_cTx += model.c[static_cast<size_t>(j)] * solution.col_value[static_cast<size_t>(j)];
    }
    res.recomputed_objective = recomputed_cTx;
    res.objective_discrepancy = std::abs(res.recomputed_objective - res.reported_objective);
    const double obj_denom = std::max(1.0, std::abs(res.reported_objective));
    res.objective_matches = (res.objective_discrepancy / obj_denom <= tol);
    if (!res.objective_matches) {
        res.violations.push_back("Objective mismatch: reported " + std::to_string(res.reported_objective) +
                                 " vs recomputed " + std::to_string(res.recomputed_objective));
    }

    // 5. Dual Feasibility & Complementary Slackness (if duals provided)
    if (static_cast<int>(solution.row_dual.size()) == m) {
        std::vector<double> Aty(static_cast<size_t>(n), 0.0);
        model.A.multiply_transpose(solution.row_dual, Aty);

        const double sense_factor = (model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;

        for (int j = 0; j < n; ++j) {
            const double c_eff = sense_factor * model.c[static_cast<size_t>(j)];
            const double dj = c_eff - Aty[static_cast<size_t>(j)];

            const double xj = solution.col_value[static_cast<size_t>(j)];
            const double lj = model.col_lower[static_cast<size_t>(j)];
            const double uj = model.col_upper[static_cast<size_t>(j)];

            // In canonical minimization:
            // If variable can increase (xj < uj), reduced cost must be >= 0 (violation if dj < -tol)
            if (xj < uj - tol && dj < -tol) {
                res.max_dual_violation = std::max(res.max_dual_violation, -dj);
                res.violations.push_back("Variable " + model.col_names[static_cast<size_t>(j)] +
                                         " can increase but dj < 0: " + std::to_string(dj));
            }
            // If variable can decrease (xj > lj), reduced cost must be <= 0 (violation if dj > tol)
            if (xj > lj + tol && dj > tol) {
                res.max_dual_violation = std::max(res.max_dual_violation, dj);
                res.violations.push_back("Variable " + model.col_names[static_cast<size_t>(j)] +
                                         " can decrease but dj > 0: " + std::to_string(dj));
            }

            // Complementarity
            if (dj > tol) {
                const double dist = std::abs(xj - lj);
                res.max_complementarity_violation = std::max(res.max_complementarity_violation, dj * dist);
            } else if (dj < -tol) {
                const double dist = std::abs(xj - uj);
                res.max_complementarity_violation = std::max(res.max_complementarity_violation, (-dj) * dist);
            }
        }

        for (int i = 0; i < m; ++i) {
            const double yi = solution.row_dual[static_cast<size_t>(i)];
            const double li = model.row_lower[static_cast<size_t>(i)];
            const double ui = model.row_upper[static_cast<size_t>(i)];

            if (li <= -1e20 && ui < 1e20) {
                // <= row: yi <= 0
                if (yi > tol) {
                    res.max_dual_violation = std::max(res.max_dual_violation, yi);
                    res.violations.push_back("Row " + model.row_names[static_cast<size_t>(i)] +
                                             " (<= row) has positive dual multiplier yi: " + std::to_string(yi));
                }
            } else if (li > -1e20 && ui >= 1e20) {
                // >= row: yi >= 0
                if (yi < -tol) {
                    res.max_dual_violation = std::max(res.max_dual_violation, -yi);
                    res.violations.push_back("Row " + model.row_names[static_cast<size_t>(i)] +
                                             " (>= row) has negative dual multiplier yi: " + std::to_string(yi));
                }
            }
        }
        res.dual_feasible = (res.max_dual_violation <= tol && res.max_complementarity_violation <= tol);
    } else {
        res.dual_feasible = true; // No dual multipliers to verify
    }

    res.passed = res.primal_feasible && res.objective_matches && res.dual_feasible && res.violations.empty();

    std::ostringstream ss;
    ss << (res.passed ? "[VERIFIED OPTIMAL]" : "[VERIFICATION FAILED]")
       << " Primal viol: " << res.max_primal_violation
       << ", Dual viol: " << res.max_dual_violation
       << ", Obj discrepancy: " << res.objective_discrepancy;
    res.summary = ss.str();

    return res;
}

VerificationResult verify_solution_file(const Model& model, const std::string& sol_filepath, double tol) {
    std::ifstream file(sol_filepath);
    if (!file.is_open()) {
        VerificationResult res;
        res.passed = false;
        res.summary = "Could not open solution file: " + sol_filepath;
        res.violations.push_back(res.summary);
        return res;
    }

    std::unordered_map<std::string, int> col_map;
    for (size_t j = 0; j < model.col_names.size(); ++j) {
        col_map[model.col_names[j]] = static_cast<int>(j);
    }

    std::unordered_map<std::string, int> row_map;
    for (size_t i = 0; i < model.row_names.size(); ++i) {
        row_map[model.row_names[i]] = static_cast<int>(i);
    }

    Solution sol;
    sol.col_value.assign(static_cast<size_t>(model.num_cols), 0.0);
    sol.row_value.assign(static_cast<size_t>(model.num_rows), 0.0);
    sol.row_dual.assign(static_cast<size_t>(model.num_rows), 0.0);
    sol.col_dual.assign(static_cast<size_t>(model.num_cols), 0.0);

    std::vector<bool> col_seen(static_cast<size_t>(model.num_cols), false);
    std::vector<bool> row_seen(static_cast<size_t>(model.num_rows), false);
    bool status_found = false;
    bool in_columns = false;
    bool in_rows = false;

    std::string line;
    int line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        if (line.empty()) continue;

        // Header comments
        if (line[0] == '#') {
            if (line.find("Columns (Variables)") != std::string::npos) {
                in_columns = true;
                in_rows = false;
            } else if (line.find("Rows (Constraints)") != std::string::npos) {
                in_columns = false;
                in_rows = true;
            } else if (line.find("Status:") != std::string::npos) {
                auto tokens = split_tokens(line);
                if (tokens.size() >= 3) {
                    status_found = true;
                    if (tokens[2] == "OPTIMAL") sol.status = SolveStatus::kOptimal;
                    else if (tokens[2] == "FEASIBLE") sol.status = SolveStatus::kFeasible;
                    else if (tokens[2] == "INFEASIBLE") sol.status = SolveStatus::kInfeasible;
                    else if (tokens[2] == "UNBOUNDED") sol.status = SolveStatus::kUnbounded;
                    else {
                        VerificationResult res;
                        res.passed = false;
                        res.summary = "Unrecognized status token at line " + std::to_string(line_number) + ": " + tokens[2];
                        res.violations.push_back(res.summary);
                        return res;
                    }
                }
            } else if (line.find("Objective:") != std::string::npos) {
                auto tokens = split_tokens(line);
                if (tokens.size() >= 3) {
                    try {
                        size_t pos = 0;
                        double obj_val = std::stod(tokens[2], &pos);
                        if (pos != tokens[2].size() || std::isnan(obj_val) || std::isinf(obj_val)) {
                            throw std::invalid_argument("Malformed objective");
                        }
                        sol.objective_value = obj_val;
                    } catch (...) {
                        VerificationResult res;
                        res.passed = false;
                        res.summary = "Malformed objective float at line " + std::to_string(line_number) + ": " + tokens[2];
                        res.violations.push_back(res.summary);
                        return res;
                    }
                }
            }
            continue;
        }

        auto tokens = split_tokens(line);
        if (tokens.empty()) continue;

        if (in_columns) {
            if (tokens.size() < 2) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Malformed column record at line " + std::to_string(line_number) + ": expected at least 2 tokens";
                res.violations.push_back(res.summary);
                return res;
            }

            const std::string name = tokens[0];
            auto it = col_map.find(name);
            if (it == col_map.end()) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Unknown variable in solution file at line " + std::to_string(line_number) + ": " + name;
                res.violations.push_back(res.summary);
                return res;
            }

            const int c_idx = it->second;
            if (col_seen[static_cast<size_t>(c_idx)]) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Duplicate variable in solution file at line " + std::to_string(line_number) + ": " + name;
                res.violations.push_back(res.summary);
                return res;
            }

            try {
                size_t pos = 0;
                double val = std::stod(tokens[1], &pos);
                if (pos != tokens[1].size() || std::isnan(val) || std::isinf(val)) {
                    throw std::invalid_argument("Malformed float value");
                }
                sol.col_value[static_cast<size_t>(c_idx)] = val;
            } catch (...) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Malformed numeric value for variable " + name + " at line " + std::to_string(line_number) + ": " + tokens[1];
                res.violations.push_back(res.summary);
                return res;
            }

            if (tokens.size() >= 3) {
                try {
                    size_t pos = 0;
                    double dj = std::stod(tokens[2], &pos);
                    if (pos != tokens[2].size() || std::isnan(dj) || std::isinf(dj)) {
                        throw std::invalid_argument("Malformed reduced cost");
                    }
                    sol.col_dual[static_cast<size_t>(c_idx)] = dj;
                } catch (...) {
                    VerificationResult res;
                    res.passed = false;
                    res.summary = "Malformed reduced cost for variable " + name + " at line " + std::to_string(line_number) + ": " + tokens[2];
                    res.violations.push_back(res.summary);
                    return res;
                }
            }
            col_seen[static_cast<size_t>(c_idx)] = true;
        } else if (in_rows) {
            if (tokens.size() < 2) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Malformed row record at line " + std::to_string(line_number) + ": expected at least 2 tokens";
                res.violations.push_back(res.summary);
                return res;
            }

            const std::string name = tokens[0];
            auto it = row_map.find(name);
            if (it == row_map.end()) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Unknown row in solution file at line " + std::to_string(line_number) + ": " + name;
                res.violations.push_back(res.summary);
                return res;
            }

            const int r_idx = it->second;
            if (row_seen[static_cast<size_t>(r_idx)]) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Duplicate row in solution file at line " + std::to_string(line_number) + ": " + name;
                res.violations.push_back(res.summary);
                return res;
            }

            try {
                size_t pos = 0;
                double val = std::stod(tokens[1], &pos);
                if (pos != tokens[1].size() || std::isnan(val) || std::isinf(val)) {
                    throw std::invalid_argument("Malformed row activity");
                }
                sol.row_value[static_cast<size_t>(r_idx)] = val;
            } catch (...) {
                VerificationResult res;
                res.passed = false;
                res.summary = "Malformed activity value for row " + name + " at line " + std::to_string(line_number) + ": " + tokens[1];
                res.violations.push_back(res.summary);
                return res;
            }

            if (tokens.size() >= 3) {
                try {
                    size_t pos = 0;
                    double yi = std::stod(tokens[2], &pos);
                    if (pos != tokens[2].size() || std::isnan(yi) || std::isinf(yi)) {
                        throw std::invalid_argument("Malformed shadow price");
                    }
                    sol.row_dual[static_cast<size_t>(r_idx)] = yi;
                } catch (...) {
                    VerificationResult res;
                    res.passed = false;
                    res.summary = "Malformed shadow price for row " + name + " at line " + std::to_string(line_number) + ": " + tokens[2];
                    res.violations.push_back(res.summary);
                    return res;
                }
            }
            row_seen[static_cast<size_t>(r_idx)] = true;
        }
    }

    if (!status_found) {
        VerificationResult res;
        res.passed = false;
        res.summary = "Solution file missing '# Status:' header declaration";
        res.violations.push_back(res.summary);
        return res;
    }

    // Completeness check: all variables declared in model MUST be present in solution file
    for (int j = 0; j < model.num_cols; ++j) {
        if (!col_seen[static_cast<size_t>(j)]) {
            VerificationResult res;
            res.passed = false;
            res.summary = "Incomplete solution: missing variable " + model.col_names[static_cast<size_t>(j)];
            res.violations.push_back(res.summary);
            return res;
        }
    }

    return verify_solution(model, sol, tol);
}

VerificationResult verify_files(const std::string& model_filepath, const std::string& sol_filepath, double tol) {
    Model model;
    if (model_filepath.ends_with(".lp") || model_filepath.ends_with(".LP")) {
        model = io::read_lp(model_filepath);
    } else {
        model = io::read_mps(model_filepath);
    }
    return verify_solution_file(model, sol_filepath, tol);
}

} // namespace indus::verifier
