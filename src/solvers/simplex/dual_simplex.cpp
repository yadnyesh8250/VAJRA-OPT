#include "src/solvers/simplex/simplex_core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace indus::simplex {

SimplexCore::SimplexCore(SimplexModel model) : model_(std::move(model)) {
    basis_.initialize(model_.num_rows, model_.num_vars);
}

void SimplexCore::init_slack_basis() {
    const int m = model_.num_rows;
    const int n = model_.num_cols;
    const int num_vars = model_.num_vars;

    basis_.initialize(m, num_vars);

    // Slack variables are at indices n ... n + m - 1
    for (int i = 0; i < m; ++i) {
        const int slack_idx = n + i;
        basis_.basic_vars[static_cast<size_t>(i)] = slack_idx;
        basis_.basic_index[static_cast<size_t>(slack_idx)] = i;
        basis_.status[static_cast<size_t>(slack_idx)] = BasisStatus::kBasic;
        basis_.devex_weights[static_cast<size_t>(i)] = 1.0;
    }

    // Structural variables are non-basic
    for (int j = 0; j < n; ++j) {
        basis_.basic_index[static_cast<size_t>(j)] = -1;
        const double l = model_.lower[static_cast<size_t>(j)];
        const double u = model_.upper[static_cast<size_t>(j)];

        if (std::abs(u - l) <= tol::kZeroDrop) {
            basis_.status[static_cast<size_t>(j)] = BasisStatus::kFixed;
            basis_.primal[static_cast<size_t>(j)] = l;
        } else if (l > -1e20) {
            basis_.status[static_cast<size_t>(j)] = BasisStatus::kAtLower;
            basis_.primal[static_cast<size_t>(j)] = l;
        } else if (u < 1e20) {
            basis_.status[static_cast<size_t>(j)] = BasisStatus::kAtUpper;
            basis_.primal[static_cast<size_t>(j)] = u;
        } else {
            basis_.status[static_cast<size_t>(j)] = BasisStatus::kNonbasicFree;
            basis_.primal[static_cast<size_t>(j)] = 0.0;
        }
    }
}

bool SimplexCore::refactorize_basis() {
    const int m = model_.num_rows;
    if (m <= 0) return true;

    // Construct basis matrix B from columns of basic_vars
    std::vector<la::Triplet> B_triplets;
    std::vector<int> col_rows;
    std::vector<double> col_vals;

    for (int k = 0; k < m; ++k) {
        const int var = basis_.basic_vars[static_cast<size_t>(k)];
        model_.get_column(var, col_rows, col_vals);
        const size_t sz = col_rows.size();
        for (size_t idx = 0; idx < sz; ++idx) {
            B_triplets.push_back({col_rows[idx], k, col_vals[idx]});
        }
    }

    la::SparseMatrixCSC B = la::SparseMatrixCSC::from_triplets(m, m, B_triplets);
    la::FactorizationStatus status = lu_.factorize(B, tol::kMarkowitzThreshold);
    return (status != la::FactorizationStatus::kFailed);
}

void SimplexCore::compute_primal_values() {
    const int m = model_.num_rows;
    const int num_vars = model_.num_vars;

    // b_N = - sum_{j nonbasic} a_j * x_j
    std::vector<double> rhs(static_cast<size_t>(m), 0.0);
    std::vector<int> col_rows;
    std::vector<double> col_vals;

    for (int j = 0; j < num_vars; ++j) {
        if (basis_.basic_index[static_cast<size_t>(j)] == -1) {
            const double xj = basis_.primal[static_cast<size_t>(j)];
            if (std::abs(xj) > tol::kZeroDrop) {
                model_.get_column(j, col_rows, col_vals);
                const size_t sz = col_rows.size();
                for (size_t idx = 0; idx < sz; ++idx) {
                    rhs[static_cast<size_t>(col_rows[idx])] -= col_vals[idx] * xj;
                }
            }
        }
    }

    lu_.ftran(rhs);

    for (int i = 0; i < m; ++i) {
        const int var = basis_.basic_vars[static_cast<size_t>(i)];
        basis_.primal[static_cast<size_t>(var)] = rhs[static_cast<size_t>(i)];
    }
}

void SimplexCore::compute_dual_values() {
    const int m = model_.num_rows;
    std::vector<double> c_B(static_cast<size_t>(m), 0.0);
    for (int i = 0; i < m; ++i) {
        const int var = basis_.basic_vars[static_cast<size_t>(i)];
        c_B[static_cast<size_t>(i)] = model_.cost[static_cast<size_t>(var)];
    }

    lu_.btran(c_B);
    basis_.dual = std::move(c_B);
}

void SimplexCore::compute_reduced_costs() {
    const int n = model_.num_cols;
    const int m = model_.num_rows;

    // For structural j: d_j = c_j - A_{\cdot j}ᵀ * y
    for (int j = 0; j < n; ++j) {
        const auto rows = model_.A.col_rows(j);
        const auto vals = model_.A.col_vals(j);
        double dot = 0.0;
        const size_t sz = rows.size();
        for (size_t idx = 0; idx < sz; ++idx) {
            dot += vals[idx] * basis_.dual[static_cast<size_t>(rows[idx])];
        }
        basis_.reduced_cost[static_cast<size_t>(j)] = model_.cost[static_cast<size_t>(j)] - dot;
    }

    // For slacks j = n + i: d_{n+i} = 0 - (e_i)ᵀ * y = -y_i
    for (int i = 0; i < m; ++i) {
        basis_.reduced_cost[static_cast<size_t>(n + i)] = -basis_.dual[static_cast<size_t>(i)];
    }

    // For basic variables, explicitly clamp to zero
    for (int i = 0; i < m; ++i) {
        const int var = basis_.basic_vars[static_cast<size_t>(i)];
        basis_.reduced_cost[static_cast<size_t>(var)] = 0.0;
    }
}

double SimplexCore::compute_objective_value() const {
    double obj = model_.objective_offset;
    const size_t n_vars = static_cast<size_t>(model_.num_vars);
    for (size_t j = 0; j < n_vars; ++j) {
        obj += model_.cost[j] * basis_.primal[j];
    }
    if (model_.sense == ObjSense::kMaximize) {
        obj = -obj;
    }
    return obj;
}

void SimplexCore::ftran(int enter_var, std::vector<double>& alpha) const {
    const int m = model_.num_rows;
    alpha.assign(static_cast<size_t>(m), 0.0);

    std::vector<int> col_rows;
    std::vector<double> col_vals;
    model_.get_column(enter_var, col_rows, col_vals);
    const size_t sz = col_rows.size();
    for (size_t idx = 0; idx < sz; ++idx) {
        alpha[static_cast<size_t>(col_rows[idx])] = col_vals[idx];
    }

    lu_.ftran(alpha);
}

void SimplexCore::btran(int row_p, std::vector<double>& v, double sign) const {
    const int m = model_.num_rows;
    v.assign(static_cast<size_t>(m), 0.0);
    if (row_p >= 0 && row_p < m) {
        v[static_cast<size_t>(row_p)] = sign;
    }
    lu_.btran(v);
}

double SimplexCore::compute_tableau_entry(int j, const std::vector<double>& v) const {
    if (j < model_.num_cols) {
        const auto rows = model_.A.col_rows(j);
        const auto vals = model_.A.col_vals(j);
        double dot = 0.0;
        const size_t sz = rows.size();
        for (size_t idx = 0; idx < sz; ++idx) {
            dot += vals[idx] * v[static_cast<size_t>(rows[idx])];
        }
        return dot;
    } else {
        const int r = j - model_.num_cols;
        return v[static_cast<size_t>(r)];
    }
}

SimplexResult SimplexCore::solve(int64_t max_iterations) {
    // Check initial basis status
    init_slack_basis();
    if (!refactorize_basis()) {
        SimplexResult res;
        res.status = SolveStatus::kNumericalError;
        res.status_message = "Initial basis factorization failed";
        return res;
    }

    compute_primal_values();
    compute_dual_values();
    compute_reduced_costs();

    SimplexResult res = solve_dual(max_iterations);
    if (res.status == SolveStatus::kOptimal || res.status == SolveStatus::kInfeasible || res.status == SolveStatus::kUnbounded) {
        return res;
    }
    return solve_primal(max_iterations);
}

SimplexResult SimplexCore::solve_dual(int64_t max_iterations) {
    SimplexResult result;
    const int m = model_.num_rows;
    const int n = model_.num_cols;
    const int num_vars = model_.num_vars;

    init_slack_basis();
    if (!refactorize_basis()) {
        result.status = SolveStatus::kNumericalError;
        result.status_message = "Initial basis factorization failed";
        return result;
    }

    compute_dual_values();
    compute_reduced_costs();

    // Dual feasibility setup: flip non-basic variables if needed to establish dual feasibility
    for (int j = 0; j < num_vars; ++j) {
        if (basis_.basic_index[static_cast<size_t>(j)] == -1) {
            const double l = model_.lower[static_cast<size_t>(j)];
            const double u = model_.upper[static_cast<size_t>(j)];
            const double dj = basis_.reduced_cost[static_cast<size_t>(j)];

            if (basis_.status[static_cast<size_t>(j)] == BasisStatus::kAtLower) {
                if (dj < -tol::kDualFeasibility && u < 1e20) {
                    basis_.status[static_cast<size_t>(j)] = BasisStatus::kAtUpper;
                    basis_.primal[static_cast<size_t>(j)] = u;
                }
            } else if (basis_.status[static_cast<size_t>(j)] == BasisStatus::kAtUpper) {
                if (dj > tol::kDualFeasibility && l > -1e20) {
                    basis_.status[static_cast<size_t>(j)] = BasisStatus::kAtLower;
                    basis_.primal[static_cast<size_t>(j)] = l;
                }
            }
        }
    }

    compute_primal_values();

    int64_t iter = 0;
    while (iter < max_iterations) {
        iter++;

        // 1. Dual Pricing: select leaving variable p using Devex weights
        int leave_row = -1;
        double sigma = 0.0; // +1 if x < l (needs to increase), -1 if x > u (needs to decrease)
        double max_score = 0.0;

        for (int i = 0; i < m; ++i) {
            const int var = basis_.basic_vars[static_cast<size_t>(i)];
            const double x = basis_.primal[static_cast<size_t>(var)];
            const double l = model_.lower[static_cast<size_t>(var)];
            const double u = model_.upper[static_cast<size_t>(var)];

            double viol = 0.0;
            double s = 0.0;

            if (x < l - tol::kPrimalFeasibility) {
                viol = l - x;
                s = 1.0;  // Needs to increase
            } else if (x > u + tol::kPrimalFeasibility) {
                viol = x - u;
                s = -1.0; // Needs to decrease
            }

            if (viol > tol::kPrimalFeasibility) {
                const double weight = std::max(basis_.devex_weights[static_cast<size_t>(i)], 1e-4);
                const double score = (viol * viol) / weight;
                if (score > max_score) {
                    max_score = score;
                    leave_row = i;
                    sigma = s;
                }
            }
        }

        // If no basic variable violates bounds:
        if (leave_row == -1) {
            // Check dual feasibility
            bool dual_feas = true;
            for (int j = 0; j < num_vars; ++j) {
                if (basis_.basic_index[static_cast<size_t>(j)] == -1) {
                    const double dj = basis_.reduced_cost[static_cast<size_t>(j)];
                    const auto st = basis_.status[static_cast<size_t>(j)];
                    if (st == BasisStatus::kAtLower && dj < -tol::kDualFeasibility) { dual_feas = false; break; }
                    if (st == BasisStatus::kAtUpper && dj > tol::kDualFeasibility) { dual_feas = false; break; }
                    if (st == BasisStatus::kNonbasicFree && std::abs(dj) > tol::kDualFeasibility) { dual_feas = false; break; }
                }
            }

            if (dual_feas) {
                result.status = SolveStatus::kOptimal;
                result.status_message = "Optimal solution found";
                break;
            } else {
                // Primal feasible, but not dual feasible: finish with primal simplex
                return solve_primal(max_iterations - iter);
            }
        }

        // 2. Generate pivot row via BTRAN: Bᵀ v = e_p
        std::vector<double> v;
        btran(leave_row, v, 1.0);

        // 3. Bound-Flipping Ratio Test: select entering variable
        struct Candidate {
            int var = -1;
            double theta = 0.0;
            double a_pj = 0.0;
        };
        std::vector<Candidate> candidates;

        for (int j = 0; j < num_vars; ++j) {
            if (basis_.basic_index[static_cast<size_t>(j)] != -1) continue;
            if (basis_.status[static_cast<size_t>(j)] == BasisStatus::kFixed) continue;

            const double a_pj = compute_tableau_entry(j, v);
            const double dj = basis_.reduced_cost[static_cast<size_t>(j)];
            const auto st = basis_.status[static_cast<size_t>(j)];

            // delta = +1 if at lower bound (increasing), -1 if at upper bound (decreasing)
            double delta = 0.0;
            if (st == BasisStatus::kAtLower) delta = 1.0;
            else if (st == BasisStatus::kAtUpper) delta = -1.0;

            // Condition to move x_{B_p} towards feasibility: sigma * delta * a_pj < 0
            if (delta != 0.0) {
                if (sigma * delta * a_pj < -tol::kZeroDrop) {
                    const double ratio = std::abs(dj) / std::abs(a_pj);
                    candidates.push_back({j, ratio, a_pj});
                }
            } else if (st == BasisStatus::kNonbasicFree) {
                if (std::abs(a_pj) > tol::kZeroDrop) {
                    candidates.push_back({j, 0.0, a_pj});
                }
            }
        }

        if (candidates.empty()) {
            // Primal Infeasible / Dual Unbounded -> emit Farkas Certificate
            result.status = SolveStatus::kInfeasible;
            result.status_message = "Primal infeasible (dual ray found)";
            result.certificate_type = "farkas";
            result.certificate_vector = std::move(v);
            break;
        }

        // Sort candidates in ascending order of ratio theta
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.theta < b.theta;
        });

        const int leave_var = basis_.basic_vars[static_cast<size_t>(leave_row)];
        const double x_leave = basis_.primal[static_cast<size_t>(leave_var)];
        double remaining_viol = (sigma > 0) ? (model_.lower[static_cast<size_t>(leave_var)] - x_leave)
                                            : (x_leave - model_.upper[static_cast<size_t>(leave_var)]);

        int enter_var = -1;
        for (const auto& cand : candidates) {
            const int j = cand.var;
            const double l = model_.lower[static_cast<size_t>(j)];
            const double u = model_.upper[static_cast<size_t>(j)];

            if (l > -1e20 && u < 1e20) {
                const double cap = u - l;
                const double delta_reduc = cap * std::abs(cand.a_pj);

                if (delta_reduc < remaining_viol - tol::kPrimalFeasibility) {
                    // Bound-flip without pivot
                    if (basis_.status[static_cast<size_t>(j)] == BasisStatus::kAtLower) {
                        basis_.status[static_cast<size_t>(j)] = BasisStatus::kAtUpper;
                        basis_.primal[static_cast<size_t>(j)] = u;
                    } else {
                        basis_.status[static_cast<size_t>(j)] = BasisStatus::kAtLower;
                        basis_.primal[static_cast<size_t>(j)] = l;
                    }
                    remaining_viol -= delta_reduc;
                    continue;
                }
            }

            enter_var = j;
            break;
        }

        if (enter_var == -1) {
            compute_primal_values();
            continue;
        }

        // 4. Pivot Column FTRAN
        std::vector<double> alpha;
        ftran(enter_var, alpha);
        const double pivot_elem = alpha[static_cast<size_t>(leave_row)];

        // 5. Update Basis and Devex weights
        const double pivot_sq = pivot_elem * pivot_elem;
        const double gamma_p = basis_.devex_weights[static_cast<size_t>(leave_row)];

        for (int i = 0; i < m; ++i) {
            if (i == leave_row) {
                basis_.devex_weights[static_cast<size_t>(i)] = std::max(0.01, gamma_p / pivot_sq);
            } else {
                const double a_iq = alpha[static_cast<size_t>(i)];
                const double ratio = a_iq / pivot_elem;
                basis_.devex_weights[static_cast<size_t>(i)] = std::max(basis_.devex_weights[static_cast<size_t>(i)],
                                                                        ratio * ratio * gamma_p);
            }
        }

        // Update variable statuses
        basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kBasic;
        if (sigma > 0) {
            basis_.status[static_cast<size_t>(leave_var)] = BasisStatus::kAtLower;
            basis_.primal[static_cast<size_t>(leave_var)] = model_.lower[static_cast<size_t>(leave_var)];
        } else {
            basis_.status[static_cast<size_t>(leave_var)] = BasisStatus::kAtUpper;
            basis_.primal[static_cast<size_t>(leave_var)] = model_.upper[static_cast<size_t>(leave_var)];
        }

        basis_.basic_vars[static_cast<size_t>(leave_row)] = enter_var;
        basis_.basic_index[static_cast<size_t>(enter_var)] = leave_row;
        basis_.basic_index[static_cast<size_t>(leave_var)] = -1;

        bool pfi_ok = lu_.update_basis_pfi(leave_row, alpha);
        if (!pfi_ok || lu_.needs_refactorization() || (iter % 50 == 0)) {
            if (!refactorize_basis()) {
                result.status = SolveStatus::kNumericalError;
                result.status_message = "Refactorization failed during dual simplex";
                break;
            }
        }

        compute_dual_values();
        compute_reduced_costs();
        compute_primal_values();
    }

    if (iter >= max_iterations && result.status == SolveStatus::kNotSolved) {
        result.status = SolveStatus::kIterationLimit;
        result.status_message = "Iteration limit reached";
    }

    result.iterations = iter;
    result.objective_value = compute_objective_value();

    result.col_value.resize(static_cast<size_t>(n));
    result.col_status.resize(static_cast<size_t>(n));
    result.col_dual.resize(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        result.col_value[static_cast<size_t>(j)] = basis_.primal[static_cast<size_t>(j)];
        result.col_status[static_cast<size_t>(j)] = basis_.status[static_cast<size_t>(j)];
        result.col_dual[static_cast<size_t>(j)] = basis_.reduced_cost[static_cast<size_t>(j)];
    }

    result.row_value.resize(static_cast<size_t>(m));
    result.row_status.resize(static_cast<size_t>(m));
    result.row_dual.resize(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
        // Ax = -s, so row activity is -primal[n + i]
        result.row_value[static_cast<size_t>(i)] = -basis_.primal[static_cast<size_t>(n + i)];
        result.row_status[static_cast<size_t>(i)] = basis_.status[static_cast<size_t>(n + i)];
        result.row_dual[static_cast<size_t>(i)] = basis_.dual[static_cast<size_t>(i)];
    }

    return result;
}

} // namespace indus::simplex
