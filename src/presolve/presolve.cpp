#include "indus/presolve.hpp"
#include "indus/tolerances.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <iostream>

namespace indus::presolve {

PresolveResult PresolveEngine::apply(const Model& orig, int max_passes) {
    PresolveResult res;
    res.presolved_model = orig;

    const int m = orig.num_rows;
    const int n = orig.num_cols;

    // 0. Crossed-bound check on original model
    for (int j = 0; j < n; ++j) {
        if (orig.col_lower[static_cast<size_t>(j)] > orig.col_upper[static_cast<size_t>(j)] + tol::kZeroDrop) {
            res.is_infeasible = true;
            res.status_message = "Crossed bounds on column " + orig.col_names[static_cast<size_t>(j)];
            return res;
        }
    }
    for (int i = 0; i < m; ++i) {
        if (orig.row_lower[static_cast<size_t>(i)] > orig.row_upper[static_cast<size_t>(i)] + tol::kZeroDrop) {
            res.is_infeasible = true;
            res.status_message = "Crossed bounds on row " + orig.row_names[static_cast<size_t>(i)];
            return res;
        }
    }

    // Working state
    std::vector<bool> row_active(static_cast<size_t>(m), true);
    std::vector<bool> col_active(static_cast<size_t>(n), true);
    std::vector<bool> row_modified(static_cast<size_t>(m), false);

    std::vector<double> row_lower = orig.row_lower;
    std::vector<double> row_upper = orig.row_upper;
    std::vector<double> col_lower = orig.col_lower;
    std::vector<double> col_upper = orig.col_upper;
    std::vector<double> cost = orig.c;
    double obj_offset = orig.objective_offset;

    // Dynamic sparse matrix adjacency lists:
    // row_mat[i]: col_idx -> coefficient
    // col_mat[j]: row_idx -> coefficient
    std::vector<std::unordered_map<int, double>> row_mat(static_cast<size_t>(m));
    std::vector<std::unordered_map<int, double>> col_mat(static_cast<size_t>(n));

    for (int j = 0; j < n; ++j) {
        const auto rows = orig.A.col_rows(j);
        const auto vals = orig.A.col_vals(j);
        const size_t sz = rows.size();
        for (size_t k = 0; k < sz; ++k) {
            const int r = rows[k];
            const double v = vals[k];
            if (std::abs(v) > tol::kZeroDrop) {
                row_mat[static_cast<size_t>(r)][j] = v;
                col_mat[static_cast<size_t>(j)][r] = v;
            }
        }
    }

    // Fixed-point reduction passes
    for (int pass = 0; pass < max_passes; ++pass) {
        int reductions_in_pass = 0;

        // 1. Empty Rows
        for (int i = 0; i < m; ++i) {
            if (!row_active[static_cast<size_t>(i)]) continue;
            if (row_mat[static_cast<size_t>(i)].empty()) {
                const double li = row_lower[static_cast<size_t>(i)];
                const double ui = row_upper[static_cast<size_t>(i)];
                if (li > tol::kPrimalFeasibility || ui < -tol::kPrimalFeasibility) {
                    res.is_infeasible = true;
                    res.status_message = "Empty row " + orig.row_names[static_cast<size_t>(i)] +
                                         " cannot be satisfied: [ " + std::to_string(li) + ", " +
                                         std::to_string(ui) + " ] does not contain 0";
                    return res;
                }
                row_active[static_cast<size_t>(i)] = false;
                ReductionRecord rec;
                rec.type = ReductionType::kEmptyRow;
                rec.row_idx = i;
                rec.name = orig.row_names[static_cast<size_t>(i)];
                rec.row_lower = li;
                rec.row_upper = ui;
                res.stack.push(std::move(rec));
                res.num_empty_rows++;
                reductions_in_pass++;
            }
        }

        // 2. Empty Columns
        for (int j = 0; j < n; ++j) {
            if (!col_active[static_cast<size_t>(j)]) continue;
            if (col_mat[static_cast<size_t>(j)].empty()) {
                col_active[static_cast<size_t>(j)] = false;
                const double cj = cost[static_cast<size_t>(j)];
                const double lj = col_lower[static_cast<size_t>(j)];
                const double uj = col_upper[static_cast<size_t>(j)];
                double fix_val = 0.0;

                if (cj > tol::kDualFeasibility) {
                    if (lj <= -1e19) {
                        res.is_unbounded = true;
                        res.status_message = "Unbounded: empty column " + orig.col_names[static_cast<size_t>(j)] + " with positive cost has no lower bound";
                        return res;
                    }
                    fix_val = lj;
                } else if (cj < -tol::kDualFeasibility) {
                    if (uj >= 1e19) {
                        res.is_unbounded = true;
                        res.status_message = "Unbounded: empty column " + orig.col_names[static_cast<size_t>(j)] + " with negative cost has no upper bound";
                        return res;
                    }
                    fix_val = uj;
                } else {
                    fix_val = (lj > -1e19) ? lj : ((uj < 1e19) ? uj : 0.0);
                }

                obj_offset += cj * fix_val;
                ReductionRecord rec;
                rec.type = ReductionType::kEmptyCol;
                rec.col_idx = j;
                rec.name = orig.col_names[static_cast<size_t>(j)];
                rec.fixed_value = fix_val;
                rec.obj_cost = cj;
                res.stack.push(std::move(rec));
                res.num_empty_cols++;
                reductions_in_pass++;
            }
        }

        // 3. Fixed Columns
        for (int j = 0; j < n; ++j) {
            if (!col_active[static_cast<size_t>(j)]) continue;
            if (std::abs(col_upper[static_cast<size_t>(j)] - col_lower[static_cast<size_t>(j)]) <= tol::kZeroDrop) {
                col_active[static_cast<size_t>(j)] = false;
                const double fix_val = col_lower[static_cast<size_t>(j)];
                const double cj = cost[static_cast<size_t>(j)];
                obj_offset += cj * fix_val;

                ReductionRecord rec;
                rec.type = ReductionType::kFixedCol;
                rec.col_idx = j;
                rec.name = orig.col_names[static_cast<size_t>(j)];
                rec.fixed_value = fix_val;
                rec.obj_cost = cj;

                for (const auto& [r, a_rj] : col_mat[static_cast<size_t>(j)]) {
                    rec.entries.push_back({r, a_rj});
                    if (row_lower[static_cast<size_t>(r)] > -1e19) {
                        row_lower[static_cast<size_t>(r)] -= a_rj * fix_val;
                    }
                    if (row_upper[static_cast<size_t>(r)] < 1e19) {
                        row_upper[static_cast<size_t>(r)] -= a_rj * fix_val;
                    }
                    row_mat[static_cast<size_t>(r)].erase(j);

                    if (row_lower[static_cast<size_t>(r)] > row_upper[static_cast<size_t>(r)] + tol::kZeroDrop) {
                        res.is_infeasible = true;
                        res.status_message = "Fixed variable " + orig.col_names[static_cast<size_t>(j)] +
                                             " created crossed bounds on row " + orig.row_names[static_cast<size_t>(r)];
                        return res;
                    }
                }

                col_mat[static_cast<size_t>(j)].clear();
                res.stack.push(std::move(rec));
                res.num_fixed_cols++;
                reductions_in_pass++;
            }
        }

        // 4. Singleton Rows
        for (int i = 0; i < m; ++i) {
            if (!row_active[static_cast<size_t>(i)]) continue;
            if (row_mat[static_cast<size_t>(i)].size() == 1) {
                const auto [j, a_ij] = *row_mat[static_cast<size_t>(i)].begin();
                if (std::abs(a_ij) > tol::kZeroDrop) {
                    const double li = row_lower[static_cast<size_t>(i)];
                    const double ui = row_upper[static_cast<size_t>(i)];
                    double impl_lb = -1e20;
                    double impl_ub = 1e20;

                    if (a_ij > 0.0) {
                        if (li > -1e19) impl_lb = li / a_ij;
                        if (ui < 1e19)  impl_ub = ui / a_ij;
                    } else {
                        if (ui < 1e19)  impl_lb = ui / a_ij;
                        if (li > -1e19) impl_ub = li / a_ij;
                    }

                    const double old_lb = col_lower[static_cast<size_t>(j)];
                    const double old_ub = col_upper[static_cast<size_t>(j)];
                    const double new_lb = std::max(old_lb, impl_lb);
                    const double new_ub = std::min(old_ub, impl_ub);

                    if (new_lb > new_ub + tol::kZeroDrop) {
                        res.is_infeasible = true;
                        res.status_message = "Singleton row " + orig.row_names[static_cast<size_t>(i)] +
                                             " creates crossed bounds on column " + orig.col_names[static_cast<size_t>(j)];
                        return res;
                    }

                    col_lower[static_cast<size_t>(j)] = new_lb;
                    col_upper[static_cast<size_t>(j)] = new_ub;

                    row_active[static_cast<size_t>(i)] = false;
                    row_mat[static_cast<size_t>(i)].clear();
                    col_mat[static_cast<size_t>(j)].erase(i);

                    ReductionRecord rec;
                    rec.type = ReductionType::kSingletonRow;
                    rec.row_idx = i;
                    rec.col_idx = j;
                    rec.coeff = a_ij;
                    rec.row_lower = li;
                    rec.row_upper = ui;
                    rec.old_lower = old_lb;
                    rec.old_upper = old_ub;
                    rec.name = orig.row_names[static_cast<size_t>(i)];
                    res.stack.push(std::move(rec));

                    res.num_singleton_rows++;
                    reductions_in_pass++;
                }
            }
        }

        // 5. Forcing Rows & Redundant Rows
        for (int i = 0; i < m; ++i) {
            if (!row_active[static_cast<size_t>(i)]) continue;
            if (row_mat[static_cast<size_t>(i)].size() <= 1) continue;

            double L_i = 0.0, U_i = 0.0;
            bool L_inf = false, U_inf = false;

            for (const auto& [j, a_ij] : row_mat[static_cast<size_t>(i)]) {
                const double lj = col_lower[static_cast<size_t>(j)];
                const double uj = col_upper[static_cast<size_t>(j)];
                if (a_ij > 0.0) {
                    if (lj <= -1e19) L_inf = true; else L_i += a_ij * lj;
                    if (uj >= 1e19)  U_inf = true; else U_i += a_ij * uj;
                } else {
                    if (uj >= 1e19)  L_inf = true; else L_i += a_ij * uj;
                    if (lj <= -1e19) U_inf = true; else U_i += a_ij * lj;
                }
            }

            const double li = row_lower[static_cast<size_t>(i)];
            const double ui = row_upper[static_cast<size_t>(i)];

            // Infeasibility
            if (!L_inf && L_i > ui + tol::kPrimalFeasibility) {
                res.is_infeasible = true;
                res.status_message = "Minimum activity of row " + orig.row_names[static_cast<size_t>(i)] + " exceeds upper bound";
                return res;
            }
            if (!U_inf && U_i < li - tol::kPrimalFeasibility) {
                res.is_infeasible = true;
                res.status_message = "Maximum activity of row " + orig.row_names[static_cast<size_t>(i)] + " is below lower bound";
                return res;
            }

            // Redundant row: only if bounds are STRICTLY looser than extremal activity
            const bool is_equality = (std::abs(li - ui) <= tol::kZeroDrop);
            const bool ub_redundant = (ui >= 1e19) || (!U_inf && U_i < ui - 1e-5);
            const bool lb_redundant = (li <= -1e19) || (!L_inf && L_i > li + 1e-5);

            if (!is_equality && ub_redundant && lb_redundant) {
                row_active[static_cast<size_t>(i)] = false;
                for (const auto& [j, a_ij] : row_mat[static_cast<size_t>(i)]) {
                    col_mat[static_cast<size_t>(j)].erase(i);
                }
                row_mat[static_cast<size_t>(i)].clear();

                ReductionRecord rec;
                rec.type = ReductionType::kRedundantRow;
                rec.row_idx = i;
                rec.name = orig.row_names[static_cast<size_t>(i)];
                // std::cout << "  [PRESOLVE] Redundant row: " << rec.name << " Li=" << L_i << " li=" << li << " Ui=" << U_i << " ui=" << ui << "\n";
                res.stack.push(std::move(rec));

                res.num_redundant_rows++;
                reductions_in_pass++;
                continue;
            }

            // Forcing row: forcing at lower limit (activity cannot exceed li)
            const bool forcing_lower = (!U_inf && std::abs(U_i - li) <= tol::kPrimalFeasibility);
            const bool forcing_upper = (!L_inf && std::abs(L_i - ui) <= tol::kPrimalFeasibility);

            if (forcing_lower || forcing_upper) {
                ReductionRecord rec;
                rec.type = ReductionType::kForcingRow;
                rec.row_idx = i;
                rec.name = orig.row_names[static_cast<size_t>(i)];
                rec.row_lower = li;
                rec.row_upper = ui;

                std::vector<std::pair<int, double>> vars_to_fix;
                for (const auto& [j, a_ij] : row_mat[static_cast<size_t>(i)]) {
                    rec.entries.push_back({j, a_ij});
                    double fix_val = 0.0;
                    if (forcing_lower) {
                        fix_val = (a_ij > 0.0) ? col_upper[static_cast<size_t>(j)] : col_lower[static_cast<size_t>(j)];
                    } else {
                        fix_val = (a_ij > 0.0) ? col_lower[static_cast<size_t>(j)] : col_upper[static_cast<size_t>(j)];
                    }
                    vars_to_fix.push_back({j, fix_val});
                }

                row_active[static_cast<size_t>(i)] = false;
                for (const auto& [j, a_ij] : row_mat[static_cast<size_t>(i)]) {
                    col_mat[static_cast<size_t>(j)].erase(i);
                }
                row_mat[static_cast<size_t>(i)].clear();

                res.stack.push(std::move(rec));
                res.num_forcing_rows++;
                reductions_in_pass++;

                for (const auto& [j, fval] : vars_to_fix) {
                    col_lower[static_cast<size_t>(j)] = fval;
                    col_upper[static_cast<size_t>(j)] = fval;
                }
                continue;
            }
        }

        // 6. Free Column Singletons
        for (int j = 0; j < n; ++j) {
            if (!col_active[static_cast<size_t>(j)]) continue;
            if (col_lower[static_cast<size_t>(j)] <= -1e19 && col_upper[static_cast<size_t>(j)] >= 1e19 &&
                col_mat[static_cast<size_t>(j)].size() == 1) {

                const auto [i, a_ij] = *col_mat[static_cast<size_t>(j)].begin();
                if (row_active[static_cast<size_t>(i)] &&
                    std::abs(row_lower[static_cast<size_t>(i)] - row_upper[static_cast<size_t>(i)]) <= tol::kPrimalFeasibility &&
                    std::abs(a_ij) > 1e-4) {

                    const double bi = row_lower[static_cast<size_t>(i)];
                    const double cj = cost[static_cast<size_t>(j)];

                    ReductionRecord rec;
                    rec.type = ReductionType::kFreeColSingleton;
                    rec.row_idx = i;
                    rec.col_idx = j;
                    rec.coeff = a_ij;
                    rec.fixed_value = bi;
                    rec.obj_cost = cj;
                    rec.name = orig.col_names[static_cast<size_t>(j)];

                    obj_offset += cj * (bi / a_ij);

                    for (const auto& [k, a_ik] : row_mat[static_cast<size_t>(i)]) {
                        if (k != j) {
                            rec.entries.push_back({k, a_ik});
                            cost[static_cast<size_t>(k)] -= cj * (a_ik / a_ij);
                            col_mat[static_cast<size_t>(k)].erase(i);
                        }
                    }

                    col_active[static_cast<size_t>(j)] = false;
                    row_active[static_cast<size_t>(i)] = false;
                    col_mat[static_cast<size_t>(j)].clear();
                    row_mat[static_cast<size_t>(i)].clear();

                    res.stack.push(std::move(rec));
                    res.num_free_col_singletons++;
                    reductions_in_pass++;
                }
            }
        }

        // 7. Doubleton Rows (Safe Equality Substitution: a1*x1 + a2*x2 = b)
        for (int i = 0; i < m; ++i) {
            if (!row_active[static_cast<size_t>(i)] || row_modified[static_cast<size_t>(i)]) continue;
            if (row_mat[static_cast<size_t>(i)].size() == 2 &&
                std::abs(row_lower[static_cast<size_t>(i)] - row_upper[static_cast<size_t>(i)]) <= tol::kZeroDrop) {

                auto it = row_mat[static_cast<size_t>(i)].begin();
                const int j1 = it->first;
                const double a1 = it->second;
                ++it;
                const int j2 = it->first;
                const double a2 = it->second;
                const double b = row_lower[static_cast<size_t>(i)];

                // Verify both coefficients and RHS are finite and nonzero
                if (!std::isfinite(a1) || !std::isfinite(a2) || !std::isfinite(b)) continue;
                if (std::abs(a1) < 1e-6 || std::abs(a2) < 1e-6) continue;

                // Numerical conditioning threshold (ratio between coefficients)
                const double max_abs = std::max(std::abs(a1), std::abs(a2));
                const double min_abs = std::min(std::abs(a1), std::abs(a2));
                if (max_abs / min_abs > 1e4) continue;

                // Mathematical safety: an eliminated variable must have all its original rows
                // still active in col_mat to guarantee exact, uncoupled dual postsolve reconstruction.
                auto is_clean_col = [&](int col) -> bool {
                    if (col_mat[static_cast<size_t>(col)].size() != orig.A.col_rows(col).size()) {
                        return false;
                    }
                    for (int r : orig.A.col_rows(col)) {
                        if (!row_active[static_cast<size_t>(r)]) return false;
                    }
                    return true;
                };

                const bool clean1 = is_clean_col(j1);
                const bool clean2 = is_clean_col(j2);
                if (!clean1 && !clean2) continue;

                // Sparsity heuristic:
                // 1. Prefer clean variable with lower column degree (fewer rows to update)
                // 2. If degrees are equal, prefer variable with larger |a| (Markowitz stability)
                const size_t deg1 = col_mat[static_cast<size_t>(j1)].size();
                const size_t deg2 = col_mat[static_cast<size_t>(j2)].size();

                int elim_col = -1, keep_col = -1;
                double a_elim = 0.0, a_keep = 0.0;

                if (clean1 && !clean2) {
                    elim_col = j1; keep_col = j2; a_elim = a1; a_keep = a2;
                } else if (!clean1 && clean2) {
                    elim_col = j2; keep_col = j1; a_elim = a2; a_keep = a1;
                } else {
                    if (deg1 < deg2) {
                        elim_col = j1; keep_col = j2; a_elim = a1; a_keep = a2;
                    } else if (deg2 < deg1) {
                        elim_col = j2; keep_col = j1; a_elim = a2; a_keep = a1;
                    } else {
                        if (std::abs(a1) >= std::abs(a2)) {
                            elim_col = j1; keep_col = j2; a_elim = a1; a_keep = a2;
                        } else {
                            elim_col = j2; keep_col = j1; a_elim = a2; a_keep = a1;
                        }
                    }
                }

                // Avoid fill-in explosion and chained coupled substitutions:
                // only eliminate if column degree <= 2 (appears in row i and at most 1 other row)
                if (col_mat[static_cast<size_t>(elim_col)].size() > 2) continue;

                // Numerical pivot stability:
                if (std::abs(a_elim) < 1e-4) continue;

                // Bound propagation from elim_col to keep_col:
                // x_keep = (b - a_elim * x_elim) / a_keep
                const double l_elim = col_lower[static_cast<size_t>(elim_col)];
                const double u_elim = col_upper[static_cast<size_t>(elim_col)];
                double impl_l = -1e20;
                double impl_u = 1e20;

                const double slope = -a_elim / a_keep;
                if (slope > 0.0) {
                    if (l_elim > -1e19) impl_l = (b - a_elim * l_elim) / a_keep;
                    if (u_elim < 1e19)  impl_u = (b - a_elim * u_elim) / a_keep;
                } else {
                    if (u_elim < 1e19)  impl_l = (b - a_elim * u_elim) / a_keep;
                    if (l_elim > -1e19) impl_u = (b - a_elim * l_elim) / a_keep;
                }

                const double new_keep_l = std::max(col_lower[static_cast<size_t>(keep_col)], impl_l);
                const double new_keep_u = std::min(col_upper[static_cast<size_t>(keep_col)], impl_u);

                if (new_keep_l > new_keep_u + tol::kZeroDrop) {
                    res.is_infeasible = true;
                    res.status_message = "Doubleton row " + orig.row_names[static_cast<size_t>(i)] +
                                         " creates crossed bounds on " + orig.col_names[static_cast<size_t>(keep_col)];
                    return res;
                }

                col_lower[static_cast<size_t>(keep_col)] = new_keep_l;
                col_upper[static_cast<size_t>(keep_col)] = new_keep_u;

                ReductionRecord rec;
                rec.type = ReductionType::kDoubletonRow;
                rec.row_idx = i;
                rec.col_idx = elim_col;
                rec.other_col_idx = keep_col;
                rec.coeff = a_elim;
                rec.other_coeff = a_keep;
                rec.fixed_value = b;
                rec.obj_cost = cost[static_cast<size_t>(elim_col)];
                rec.old_lower = l_elim;
                rec.old_upper = u_elim;
                rec.name = orig.row_names[static_cast<size_t>(i)];

                // Substitute in objective:
                // cost[elim] * x_elim = cost[elim] * (b / a_elim - (a_keep / a_elim) * x_keep)
                const double c_elim = cost[static_cast<size_t>(elim_col)];
                obj_offset += c_elim * (b / a_elim);
                cost[static_cast<size_t>(keep_col)] -= c_elim * (a_keep / a_elim);
                cost[static_cast<size_t>(elim_col)] = 0.0;

                // Collect other rows containing elim_col before modifying sparse matrix
                std::vector<std::pair<int, double>> other_row_entries;
                for (const auto& [r, a_r_elim] : col_mat[static_cast<size_t>(elim_col)]) {
                    if (r != i) {
                        other_row_entries.push_back({r, a_r_elim});
                        rec.entries.push_back({r, a_r_elim});
                    }
                }

                bool crossed_row_bounds = false;
                std::string crossed_row_name = "";

                // Substitute x_elim into other rows
                for (const auto& [r, a_r_elim] : other_row_entries) {
                    row_modified[static_cast<size_t>(r)] = true;
                    const double shift = a_r_elim * (b / a_elim);
                    if (row_lower[static_cast<size_t>(r)] > -1e19) row_lower[static_cast<size_t>(r)] -= shift;
                    if (row_upper[static_cast<size_t>(r)] < 1e19)  row_upper[static_cast<size_t>(r)] -= shift;

                    if (row_lower[static_cast<size_t>(r)] > row_upper[static_cast<size_t>(r)] + tol::kZeroDrop) {
                        crossed_row_bounds = true;
                        crossed_row_name = orig.row_names[static_cast<size_t>(r)];
                    }

                    const double delta = -a_r_elim * (a_keep / a_elim);
                    row_mat[static_cast<size_t>(r)][keep_col] += delta;
                    if (std::abs(row_mat[static_cast<size_t>(r)][keep_col]) <= tol::kZeroDrop) {
                        row_mat[static_cast<size_t>(r)].erase(keep_col);
                        col_mat[static_cast<size_t>(keep_col)].erase(r);
                    } else {
                        col_mat[static_cast<size_t>(keep_col)][r] = row_mat[static_cast<size_t>(r)][keep_col];
                    }
                    row_mat[static_cast<size_t>(r)].erase(elim_col);
                }

                if (crossed_row_bounds) {
                    res.is_infeasible = true;
                    res.status_message = "Doubleton substitution created crossed bounds on row " + crossed_row_name;
                    return res;
                }

                col_active[static_cast<size_t>(elim_col)] = false;
                row_active[static_cast<size_t>(i)] = false;
                col_mat[static_cast<size_t>(elim_col)].clear();
                row_mat[static_cast<size_t>(i)].clear();
                col_mat[static_cast<size_t>(keep_col)].erase(i);

                res.stack.push(std::move(rec));
                res.num_doubleton_rows++;
                reductions_in_pass++;
            }
        }

        if (reductions_in_pass == 0) {
            // Fixed point reached!
            break;
        }
    }

    // Build the compact reduced model
    std::vector<int> red_to_orig_row;
    std::vector<int> orig_to_red_row(static_cast<size_t>(m), -1);
    for (int i = 0; i < m; ++i) {
        if (row_active[static_cast<size_t>(i)]) {
            orig_to_red_row[static_cast<size_t>(i)] = static_cast<int>(red_to_orig_row.size());
            red_to_orig_row.push_back(i);
        }
    }

    std::vector<int> red_to_orig_col;
    std::vector<int> orig_to_red_col(static_cast<size_t>(n), -1);
    for (int j = 0; j < n; ++j) {
        if (col_active[static_cast<size_t>(j)]) {
            orig_to_red_col[static_cast<size_t>(j)] = static_cast<int>(red_to_orig_col.size());
            red_to_orig_col.push_back(j);
        }
    }

    res.stack.reduced_to_orig_row = red_to_orig_row;
    res.stack.reduced_to_orig_col = red_to_orig_col;
    res.stack.orig_to_reduced_row = orig_to_red_row;
    res.stack.orig_to_reduced_col = orig_to_red_col;

    const int red_m = static_cast<int>(red_to_orig_row.size());
    const int red_n = static_cast<int>(red_to_orig_col.size());

    Model red_model;
    red_model.name = orig.name + "_presolved";
    red_model.sense = orig.sense;
    red_model.objective_offset = obj_offset;
    red_model.num_rows = red_m;
    red_model.num_cols = red_n;

    red_model.c.resize(static_cast<size_t>(red_n));
    red_model.col_lower.resize(static_cast<size_t>(red_n));
    red_model.col_upper.resize(static_cast<size_t>(red_n));
    red_model.col_names.resize(static_cast<size_t>(red_n));

    for (int rj = 0; rj < red_n; ++rj) {
        const int oj = red_to_orig_col[static_cast<size_t>(rj)];
        red_model.c[static_cast<size_t>(rj)] = cost[static_cast<size_t>(oj)];
        red_model.col_lower[static_cast<size_t>(rj)] = col_lower[static_cast<size_t>(oj)];
        red_model.col_upper[static_cast<size_t>(rj)] = col_upper[static_cast<size_t>(oj)];
        red_model.col_names[static_cast<size_t>(rj)] = orig.col_names[static_cast<size_t>(oj)];
    }

    red_model.row_lower.resize(static_cast<size_t>(red_m));
    red_model.row_upper.resize(static_cast<size_t>(red_m));
    red_model.row_names.resize(static_cast<size_t>(red_m));

    for (int ri = 0; ri < red_m; ++ri) {
        const int oi = red_to_orig_row[static_cast<size_t>(ri)];
        red_model.row_lower[static_cast<size_t>(ri)] = row_lower[static_cast<size_t>(oi)];
        red_model.row_upper[static_cast<size_t>(ri)] = row_upper[static_cast<size_t>(oi)];
        red_model.row_names[static_cast<size_t>(ri)] = orig.row_names[static_cast<size_t>(oi)];
    }

    // Assemble reduced constraint matrix A
    std::vector<la::Triplet> triplets;
    for (int rj = 0; rj < red_n; ++rj) {
        const int oj = red_to_orig_col[static_cast<size_t>(rj)];
        for (const auto& [oi, val] : col_mat[static_cast<size_t>(oj)]) {
            const int ri = orig_to_red_row[static_cast<size_t>(oi)];
            if (ri >= 0 && std::abs(val) > tol::kZeroDrop) {
                triplets.push_back({ri, rj, val});
            }
        }
    }
    red_model.A.set_from_triplets(red_m, red_n, triplets, true);

    res.presolved_model = std::move(red_model);
    return res;
}

Solution PresolveEngine::postsolve(const Solution& red_sol,
                                   const PresolveStack& stack,
                                   const Model& orig) {
    const int orig_m = orig.num_rows;
    const int orig_n = orig.num_cols;

    Solution full_sol;
    full_sol.status = red_sol.status;
    full_sol.status_message = red_sol.status_message;
    full_sol.iterations = red_sol.iterations;
    full_sol.nodes = red_sol.nodes;
    full_sol.algorithm_used = red_sol.algorithm_used;
    full_sol.certificate_type = red_sol.certificate_type;
    full_sol.certificate_vector = red_sol.certificate_vector;

    full_sol.col_value.assign(static_cast<size_t>(orig_n), 0.0);
    full_sol.col_dual.assign(static_cast<size_t>(orig_n), 0.0);
    full_sol.row_value.assign(static_cast<size_t>(orig_m), 0.0);
    full_sol.row_dual.assign(static_cast<size_t>(orig_m), 0.0);
    full_sol.col_basis_status.assign(static_cast<size_t>(orig_n), BasisStatus::kAtLower);
    full_sol.row_basis_status.assign(static_cast<size_t>(orig_m), BasisStatus::kBasic);

    // 1. Map values from reduced solution for retained variables and constraints
    for (size_t rj = 0; rj < stack.reduced_to_orig_col.size(); ++rj) {
        const int oj = stack.reduced_to_orig_col[rj];
        if (rj < red_sol.col_value.size()) {
            full_sol.col_value[static_cast<size_t>(oj)] = red_sol.col_value[rj];
        }
        if (rj < red_sol.col_dual.size()) {
            full_sol.col_dual[static_cast<size_t>(oj)] = red_sol.col_dual[rj];
        }
        if (rj < red_sol.col_basis_status.size()) {
            full_sol.col_basis_status[static_cast<size_t>(oj)] = red_sol.col_basis_status[rj];
        }
    }

    for (size_t ri = 0; ri < stack.reduced_to_orig_row.size(); ++ri) {
        const int oi = stack.reduced_to_orig_row[ri];
        if (ri < red_sol.row_value.size()) {
            full_sol.row_value[static_cast<size_t>(oi)] = red_sol.row_value[ri];
        }
        if (ri < red_sol.row_dual.size()) {
            full_sol.row_dual[static_cast<size_t>(oi)] = red_sol.row_dual[ri];
        }
        if (ri < red_sol.row_basis_status.size()) {
            full_sol.row_basis_status[static_cast<size_t>(oi)] = red_sol.row_basis_status[ri];
        }
    }

    // 2. Unwind LIFO reduction stack in strict reverse order
    const auto& recs = stack.records();
    for (auto it = recs.rbegin(); it != recs.rend(); ++it) {
        const auto& rec = *it;

        switch (rec.type) {
            case ReductionType::kFixedCol: {
                full_sol.col_value[static_cast<size_t>(rec.col_idx)] = rec.fixed_value;
                full_sol.col_basis_status[static_cast<size_t>(rec.col_idx)] = BasisStatus::kFixed;
                break;
            }
            case ReductionType::kEmptyCol: {
                full_sol.col_value[static_cast<size_t>(rec.col_idx)] = rec.fixed_value;
                full_sol.col_dual[static_cast<size_t>(rec.col_idx)] = rec.obj_cost;
                full_sol.col_basis_status[static_cast<size_t>(rec.col_idx)] = BasisStatus::kAtLower;
                break;
            }
            case ReductionType::kEmptyRow: {
                full_sol.row_value[static_cast<size_t>(rec.row_idx)] = 0.0;
                full_sol.row_dual[static_cast<size_t>(rec.row_idx)] = 0.0;
                full_sol.row_basis_status[static_cast<size_t>(rec.row_idx)] = BasisStatus::kBasic;
                break;
            }
            case ReductionType::kSingletonRow: {
                const int orig_r = rec.row_idx;
                const int j = rec.col_idx;
                const double a = rec.coeff;
                const double xj = full_sol.col_value[static_cast<size_t>(j)];
                full_sol.row_value[static_cast<size_t>(orig_r)] = a * xj;

                if (std::abs(a) > 1e-15) {
                    double Aty_excl = 0.0;
                    const auto rows = orig.A.col_rows(j);
                    const auto vals = orig.A.col_vals(j);
                    for (size_t idx = 0; idx < rows.size(); ++idx) {
                        const int r = rows[idx];
                        if (r != orig_r) {
                            Aty_excl += vals[idx] * full_sol.row_dual[static_cast<size_t>(r)];
                        }
                    }

                    const double sense_factor = (orig.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
                    const double c_eff = sense_factor * orig.c[static_cast<size_t>(j)];
                    const double dj0 = c_eff - Aty_excl;

                    const double lj = orig.col_lower[static_cast<size_t>(j)];
                    const double uj = orig.col_upper[static_cast<size_t>(j)];
                    const bool is_equality = (std::abs(rec.row_lower - rec.row_upper) <= tol::kZeroDrop);

                    double yi = 0.0;
                    if (is_equality) {
                        yi = dj0 / a;
                        full_sol.row_basis_status[static_cast<size_t>(orig_r)] = BasisStatus::kFixed;
                    } else {
                        if (xj > lj + 1e-5 && xj < uj - 1e-5) {
                            yi = dj0 / a;
                        } else if (xj <= lj + 1e-5 && dj0 < -tol::kDualFeasibility) {
                            yi = dj0 / a;
                        } else if (xj >= uj - 1e-5 && dj0 > tol::kDualFeasibility) {
                            yi = dj0 / a;
                        }
                    }

                    if (!is_equality) {
                        if (rec.row_lower <= -1e19 && rec.row_upper < 1e19) {
                            // <= constraint requires yi <= 0 in minimization
                            yi = std::min(yi, 0.0);
                        } else if (rec.row_lower > -1e19 && rec.row_upper >= 1e19) {
                            // >= constraint requires yi >= 0 in minimization
                            yi = std::max(yi, 0.0);
                        }
                    }
                    full_sol.row_dual[static_cast<size_t>(orig_r)] = yi;
                }
                break;
            }
            case ReductionType::kForcingRow: {
                const int i = rec.row_idx;
                const double li = rec.row_lower;
                const double ui = rec.row_upper;
                const bool is_eq = (std::abs(li - ui) <= tol::kZeroDrop);

                double min_y = -1e20;
                double max_y = 1e20;
                if (!is_eq) {
                    if (li <= -1e19 && ui < 1e19) {
                        // <= constraint: yi <= 0 in minimization
                        max_y = std::min(max_y, 0.0);
                    } else if (li > -1e19 && ui >= 1e19) {
                        // >= constraint: yi >= 0 in minimization
                        min_y = std::max(min_y, 0.0);
                    }
                }

                const double sense_factor = (orig.sense == ObjSense::kMaximize) ? -1.0 : 1.0;

                for (const auto& [j, a_ij] : rec.entries) {
                    if (std::abs(a_ij) <= tol::kZeroDrop) continue;

                    double Aty_excl = 0.0;
                    const auto r_list = orig.A.col_rows(j);
                    const auto v_list = orig.A.col_vals(j);
                    for (size_t idx = 0; idx < r_list.size(); ++idx) {
                        if (r_list[idx] != i) {
                            Aty_excl += v_list[idx] * full_sol.row_dual[static_cast<size_t>(r_list[idx])];
                        }
                    }

                    const double c_eff = sense_factor * orig.c[static_cast<size_t>(j)];
                    const double c_tilde = c_eff - Aty_excl;
                    const double xj = full_sol.col_value[static_cast<size_t>(j)];
                    const double lj = orig.col_lower[static_cast<size_t>(j)];
                    const double uj = orig.col_upper[static_cast<size_t>(j)];

                    const bool at_lb = (xj <= lj + 1e-5);
                    const bool at_ub = (xj >= uj - 1e-5);

                    if (at_lb && !at_ub) {
                        // Can increase: dj >= 0 => c_tilde - a_ij * yi >= 0
                        if (a_ij > 0.0) {
                            max_y = std::min(max_y, c_tilde / a_ij);
                        } else {
                            min_y = std::max(min_y, c_tilde / a_ij);
                        }
                    } else if (at_ub && !at_lb) {
                        // Can decrease: dj <= 0 => c_tilde - a_ij * yi <= 0
                        if (a_ij > 0.0) {
                            min_y = std::max(min_y, c_tilde / a_ij);
                        } else {
                            max_y = std::min(max_y, c_tilde / a_ij);
                        }
                    } else if (!at_lb && !at_ub) {
                        // Interior: dj = 0 => yi = c_tilde / a_ij
                        const double target_y = c_tilde / a_ij;
                        min_y = std::max(min_y, target_y);
                        max_y = std::min(max_y, target_y);
                    }
                }

                double yi = 0.0;
                if (min_y <= max_y) {
                    if (min_y <= 0.0 && max_y >= 0.0) {
                        yi = 0.0;
                    } else if (min_y > 0.0) {
                        yi = min_y;
                    } else {
                        yi = max_y;
                    }
                } else {
                    yi = 0.5 * (min_y + max_y);
                }

                full_sol.row_dual[static_cast<size_t>(i)] = yi;
                full_sol.row_basis_status[static_cast<size_t>(i)] = BasisStatus::kFixed;
                break;
            }
            case ReductionType::kRedundantRow: {
                full_sol.row_dual[static_cast<size_t>(rec.row_idx)] = 0.0;
                full_sol.row_basis_status[static_cast<size_t>(rec.row_idx)] = BasisStatus::kBasic;
                break;
            }
            case ReductionType::kFreeColSingleton: {
                double sum_other = 0.0;
                for (const auto& [k, a_ik] : rec.entries) {
                    sum_other += a_ik * full_sol.col_value[static_cast<size_t>(k)];
                }
                full_sol.col_value[static_cast<size_t>(rec.col_idx)] = (rec.fixed_value - sum_other) / rec.coeff;
                full_sol.row_dual[static_cast<size_t>(rec.row_idx)] = rec.obj_cost / rec.coeff;
                full_sol.col_dual[static_cast<size_t>(rec.col_idx)] = 0.0;
                full_sol.col_basis_status[static_cast<size_t>(rec.col_idx)] = BasisStatus::kBasic;
                break;
            }
            case ReductionType::kDoubletonRow: {
                const int j1 = rec.col_idx;
                const int j2 = rec.other_col_idx;
                const double a1 = rec.coeff;
                const double a2 = rec.other_coeff;
                const double b = rec.fixed_value;

                const double x2 = full_sol.col_value[static_cast<size_t>(j2)];
                const double x1 = (b - a2 * x2) / a1;
                full_sol.col_value[static_cast<size_t>(j1)] = x1;

                const double l1 = rec.old_lower;
                const double u1 = rec.old_upper;
                double a1_ty = 0.0;
                for (const auto& [r, a_r1] : rec.entries) {
                    a1_ty += a_r1 * full_sol.row_dual[static_cast<size_t>(r)];
                }

                const double sense_factor = (orig.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
                const double c1 = sense_factor * rec.obj_cost;
                const double c1_tilde = c1 - a1_ty;

                const double l2 = orig.col_lower[static_cast<size_t>(j2)];
                const double u2 = orig.col_upper[static_cast<size_t>(j2)];
                const bool at_l2 = (l2 > -1e19 && x2 <= l2 + 1e-5);
                const bool at_u2 = (u2 < 1e19 && x2 >= u2 - 1e-5);
                const bool interior2 = !at_l2 && !at_u2;

                const bool at_l1 = (l1 > -1e19 && x1 <= l1 + 1e-5);
                const bool at_u1 = (u1 < 1e19 && x1 >= u1 - 1e-5);
                const bool interior1 = !at_l1 && !at_u1;

                double orig_a2 = 0.0;
                bool row_in_orig_j2 = false;
                const auto r2_list = orig.A.col_rows(j2);
                const auto v2_list = orig.A.col_vals(j2);
                for (size_t idx = 0; idx < r2_list.size(); ++idx) {
                    if (r2_list[idx] == rec.row_idx) {
                        row_in_orig_j2 = true;
                        orig_a2 = v2_list[idx];
                        break;
                    }
                }

                double yi = 0.0;
                if (interior2 && !interior1 && row_in_orig_j2 && std::abs(orig_a2) > 1e-12) {
                    // x2 is interior in original model, so d2 must be 0
                    double a2_ty = 0.0;
                    for (size_t idx = 0; idx < r2_list.size(); ++idx) {
                        if (r2_list[idx] != rec.row_idx) {
                            a2_ty += v2_list[idx] * full_sol.row_dual[static_cast<size_t>(r2_list[idx])];
                        }
                    }
                    const double c2 = sense_factor * orig.c[static_cast<size_t>(j2)];
                    yi = (c2 - a2_ty) / orig_a2;
                } else {
                    // Standard exact recovery: d1 = 0
                    yi = c1_tilde / a1;
                }

                full_sol.row_dual[static_cast<size_t>(rec.row_idx)] = yi;
                full_sol.row_basis_status[static_cast<size_t>(rec.row_idx)] = BasisStatus::kFixed;
                full_sol.col_basis_status[static_cast<size_t>(j1)] =
                    (!at_l1 && !at_u1) ? BasisStatus::kBasic :
                    (at_l1 && at_u1)   ? BasisStatus::kFixed :
                    at_l1             ? BasisStatus::kAtLower : BasisStatus::kAtUpper;
                break;
            }
        }
    }

    // 3. Recompute original row activities: Ax
    orig.A.multiply(full_sol.col_value, full_sol.row_value);

    // 4. Refine dual multipliers & reduced costs
    refine_dual_solution(orig, full_sol);

    // 5. Recompute objective value from original model
    double unscaled_obj = orig.objective_offset;
    for (int j = 0; j < orig_n; ++j) {
        unscaled_obj += orig.c[static_cast<size_t>(j)] * full_sol.col_value[static_cast<size_t>(j)];
    }
    full_sol.objective_value = unscaled_obj;
    full_sol.best_dual_bound = unscaled_obj;

    // 6. Recompute quality metrics against the unmodified original model
    full_sol.recompute_quality(orig);

    return full_sol;
}

void PresolveEngine::refine_dual_solution(const Model& orig, Solution& sol, double tol) {
    (void)tol;
    const int n = orig.num_cols;

    std::vector<double> Aty(static_cast<size_t>(n), 0.0);
    orig.A.multiply_transpose(sol.row_dual, Aty);

    const double sense_factor = (orig.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
    sol.col_dual.resize(static_cast<size_t>(n));

    for (int j = 0; j < n; ++j) {
        const double c_eff = sense_factor * orig.c[static_cast<size_t>(j)];
        const double dj = c_eff - Aty[static_cast<size_t>(j)];
        sol.col_dual[static_cast<size_t>(j)] = (orig.sense == ObjSense::kMaximize) ? -dj : dj;
    }
}

} // namespace indus::presolve
