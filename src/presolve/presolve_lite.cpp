#include "indus/presolve_lite.hpp"
#include "indus/tolerances.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <iostream>

namespace indus::presolve {

PresolveResult PresolveLite::apply(const Model& orig) {
    PresolveResult res;
    res.presolved_model = orig;

    const int m = orig.num_rows;
    const int n = orig.num_cols;

    // 1. Crossed-bound detection on original model
    for (int j = 0; j < n; ++j) {
        if (orig.col_lower[static_cast<size_t>(j)] > orig.col_upper[static_cast<size_t>(j)] + tol::kZeroDrop) {
            res.is_infeasible = true;
            res.infeasibility_reason = "Crossed bounds on column " + orig.col_names[static_cast<size_t>(j)] +
                                       ": " + std::to_string(orig.col_lower[static_cast<size_t>(j)]) +
                                       " > " + std::to_string(orig.col_upper[static_cast<size_t>(j)]);
            return res;
        }
    }
    for (int i = 0; i < m; ++i) {
        if (orig.row_lower[static_cast<size_t>(i)] > orig.row_upper[static_cast<size_t>(i)] + tol::kZeroDrop) {
            res.is_infeasible = true;
            res.infeasibility_reason = "Crossed bounds on row " + orig.row_names[static_cast<size_t>(i)] +
                                       ": " + std::to_string(orig.row_lower[static_cast<size_t>(i)]) +
                                       " > " + std::to_string(orig.row_upper[static_cast<size_t>(i)]);
            return res;
        }
    }

    // Count nonzeros per row
    std::vector<int> row_nnz(static_cast<size_t>(m), 0);
    std::vector<int> row_first_col(static_cast<size_t>(m), -1);
    std::vector<double> row_first_val(static_cast<size_t>(m), 0.0);

    for (int j = 0; j < n; ++j) {
        const auto rows = orig.A.col_rows(j);
        const auto vals = orig.A.col_vals(j);
        const size_t sz = rows.size();
        for (size_t idx = 0; idx < sz; ++idx) {
            const int r = rows[idx];
            if (std::abs(vals[idx]) > tol::kZeroDrop) {
                row_nnz[static_cast<size_t>(r)]++;
                row_first_col[static_cast<size_t>(r)] = j;
                row_first_val[static_cast<size_t>(r)] = vals[idx];
            }
        }
    }

    std::unordered_set<int> dropped_rows;
    std::vector<double> cur_col_lb = orig.col_lower;
    std::vector<double> cur_col_ub = orig.col_upper;

    // 2. Empty Rows
    for (int i = 0; i < m; ++i) {
        if (row_nnz[static_cast<size_t>(i)] == 0) {
            const double li = orig.row_lower[static_cast<size_t>(i)];
            const double ui = orig.row_upper[static_cast<size_t>(i)];
            if (li > tol::kPrimalFeasibility || ui < -tol::kPrimalFeasibility) {
                res.is_infeasible = true;
                res.infeasibility_reason = "Empty row " + orig.row_names[static_cast<size_t>(i)] +
                                           " cannot be satisfied: [ " + std::to_string(li) + ", " +
                                           std::to_string(ui) + " ] does not contain 0";
                return res;
            }
            dropped_rows.insert(i);
            res.stack.empty_rows.push_back({i, orig.row_names[static_cast<size_t>(i)], li, ui});
        }
    }

    // 3. Singleton Rows
    for (int i = 0; i < m; ++i) {
        if (dropped_rows.find(i) != dropped_rows.end()) continue;
        if (row_nnz[static_cast<size_t>(i)] == 1) {
            const int j = row_first_col[static_cast<size_t>(i)];
            const double a = row_first_val[static_cast<size_t>(i)];
            const double li = orig.row_lower[static_cast<size_t>(i)];
            const double ui = orig.row_upper[static_cast<size_t>(i)];

            double impl_lb = -1e20;
            double impl_ub = 1e20;

            if (a > 0.0) {
                if (li > -1e19) impl_lb = li / a;
                if (ui < 1e19) impl_ub = ui / a;
            } else {
                if (ui < 1e19) impl_lb = ui / a;
                if (li > -1e19) impl_ub = li / a;
            }

            const double old_lb = cur_col_lb[static_cast<size_t>(j)];
            const double old_ub = cur_col_ub[static_cast<size_t>(j)];

            const double new_lb = std::max(old_lb, impl_lb);
            const double new_ub = std::min(old_ub, impl_ub);

            if (new_lb > new_ub + tol::kZeroDrop) {
                res.is_infeasible = true;
                res.infeasibility_reason = "Singleton row " + orig.row_names[static_cast<size_t>(i)] +
                                           " creates crossed bounds on column " + orig.col_names[static_cast<size_t>(j)];
                return res;
            }

            cur_col_lb[static_cast<size_t>(j)] = new_lb;
            cur_col_ub[static_cast<size_t>(j)] = new_ub;

            dropped_rows.insert(i);
            res.stack.singleton_rows.push_back({i, orig.row_names[static_cast<size_t>(i)], j, a, li, ui, old_lb, old_ub});
        }
    }

    // 4. Fixed Columns
    std::unordered_set<int> fixed_cols_set;
    std::vector<double> fixed_val_map(static_cast<size_t>(n), 0.0);
    std::vector<double> adjusted_row_lb = orig.row_lower;
    std::vector<double> adjusted_row_ub = orig.row_upper;
    double offset_adj = 0.0;

    for (int j = 0; j < n; ++j) {
        if (std::abs(cur_col_lb[static_cast<size_t>(j)] - cur_col_ub[static_cast<size_t>(j)]) <= tol::kZeroDrop) {
            fixed_cols_set.insert(j);
            const double cval = cur_col_lb[static_cast<size_t>(j)];
            fixed_val_map[static_cast<size_t>(j)] = cval;
            offset_adj += orig.c[static_cast<size_t>(j)] * cval;

            FixedColStep fstep;
            fstep.orig_col_idx = j;
            fstep.col_name = orig.col_names[static_cast<size_t>(j)];
            fstep.fixed_val = cval;
            fstep.obj_cost = orig.c[static_cast<size_t>(j)];

            const auto rows = orig.A.col_rows(j);
            const auto vals = orig.A.col_vals(j);
            const size_t sz = rows.size();
            for (size_t idx = 0; idx < sz; ++idx) {
                const int r = rows[idx];
                const double v = vals[idx];
                fstep.entries.push_back({r, v});
                if (adjusted_row_lb[static_cast<size_t>(r)] > -1e19) {
                    adjusted_row_lb[static_cast<size_t>(r)] -= v * cval;
                }
                if (adjusted_row_ub[static_cast<size_t>(r)] < 1e19) {
                    adjusted_row_ub[static_cast<size_t>(r)] -= v * cval;
                }
            }
            res.stack.fixed_cols.push_back(std::move(fstep));
        }
    }

    // Construct reduced model
    std::vector<int> kept_rows;
    for (int i = 0; i < m; ++i) {
        if (dropped_rows.find(i) == dropped_rows.end()) {
            kept_rows.push_back(i);
        }
    }

    std::vector<int> kept_cols;
    for (int j = 0; j < n; ++j) {
        if (fixed_cols_set.find(j) == fixed_cols_set.end()) {
            kept_cols.push_back(j);
        }
    }

    res.stack.reduced_to_orig_row = kept_rows;
    res.stack.reduced_to_orig_col = kept_cols;

    std::unordered_map<int, int> orig_to_red_row;
    for (size_t r = 0; r < kept_rows.size(); ++r) {
        orig_to_red_row[kept_rows[r]] = static_cast<int>(r);
    }

    std::unordered_map<int, int> orig_to_red_col;
    for (size_t c = 0; c < kept_cols.size(); ++c) {
        orig_to_red_col[kept_cols[c]] = static_cast<int>(c);
    }

    const int red_m = static_cast<int>(kept_rows.size());
    const int red_n = static_cast<int>(kept_cols.size());

    res.presolved_model.name = orig.name + "_presolved";
    res.presolved_model.source_path = orig.source_path;
    res.presolved_model.sense = orig.sense;
    res.presolved_model.objective_offset = orig.objective_offset + offset_adj;
    res.presolved_model.num_rows = red_m;
    res.presolved_model.num_cols = red_n;

    res.presolved_model.row_names.resize(static_cast<size_t>(red_m));
    res.presolved_model.row_lower.resize(static_cast<size_t>(red_m));
    res.presolved_model.row_upper.resize(static_cast<size_t>(red_m));
    for (int r = 0; r < red_m; ++r) {
        const int orig_r = kept_rows[static_cast<size_t>(r)];
        res.presolved_model.row_names[static_cast<size_t>(r)] = (static_cast<size_t>(orig_r) < orig.row_names.size())
                                                                    ? orig.row_names[static_cast<size_t>(orig_r)]
                                                                    : ("R" + std::to_string(orig_r));
        res.presolved_model.row_lower[static_cast<size_t>(r)] = adjusted_row_lb[static_cast<size_t>(orig_r)];
        res.presolved_model.row_upper[static_cast<size_t>(r)] = adjusted_row_ub[static_cast<size_t>(orig_r)];
    }

    res.presolved_model.col_names.resize(static_cast<size_t>(red_n));
    res.presolved_model.col_lower.resize(static_cast<size_t>(red_n));
    res.presolved_model.col_upper.resize(static_cast<size_t>(red_n));
    res.presolved_model.col_type.resize(static_cast<size_t>(red_n));
    res.presolved_model.c.resize(static_cast<size_t>(red_n));
    for (int c = 0; c < red_n; ++c) {
        const int orig_c = kept_cols[static_cast<size_t>(c)];
        res.presolved_model.col_names[static_cast<size_t>(c)] = (static_cast<size_t>(orig_c) < orig.col_names.size())
                                                                    ? orig.col_names[static_cast<size_t>(orig_c)]
                                                                    : ("C" + std::to_string(orig_c));
        res.presolved_model.col_lower[static_cast<size_t>(c)] = cur_col_lb[static_cast<size_t>(orig_c)];
        res.presolved_model.col_upper[static_cast<size_t>(c)] = cur_col_ub[static_cast<size_t>(orig_c)];
        res.presolved_model.col_type[static_cast<size_t>(c)] = (static_cast<size_t>(orig_c) < orig.col_type.size())
                                                                   ? orig.col_type[static_cast<size_t>(orig_c)]
                                                                   : VarType::kContinuous;
        res.presolved_model.c[static_cast<size_t>(c)] = orig.c[static_cast<size_t>(orig_c)];
    }

    // Assemble reduced matrix A
    std::vector<la::Triplet> red_triplets;
    for (int c = 0; c < red_n; ++c) {
        const int orig_c = kept_cols[static_cast<size_t>(c)];
        const auto rows = orig.A.col_rows(orig_c);
        const auto vals = orig.A.col_vals(orig_c);
        const size_t sz = rows.size();
        for (size_t idx = 0; idx < sz; ++idx) {
            auto rit = orig_to_red_row.find(rows[idx]);
            if (rit != orig_to_red_row.end()) {
                red_triplets.push_back({rit->second, c, vals[idx]});
            }
        }
    }
    res.presolved_model.A.set_from_triplets(red_m, red_n, red_triplets, true);

    res.num_rows_removed = m - red_m;
    res.num_cols_removed = n - red_n;
    return res;
}

Solution PresolveLite::postsolve(const Solution& red_sol,
                                 const PresolveStack& stack,
                                 const Model& orig) {
    if (!claims_a_point(red_sol.status)) {
        Solution sol = red_sol;
        return sol;
    }

    const int m = orig.num_rows;
    const int n = orig.num_cols;

    Solution sol;
    sol.status = red_sol.status;
    sol.status_message = red_sol.status_message;
    sol.objective_value = red_sol.objective_value;
    sol.best_dual_bound = red_sol.best_dual_bound;
    sol.relative_gap = red_sol.relative_gap;
    sol.iterations = red_sol.iterations;
    sol.nodes = red_sol.nodes;
    sol.solve_time_seconds = red_sol.solve_time_seconds;
    sol.algorithm_used = red_sol.algorithm_used;

    sol.col_value.assign(static_cast<size_t>(n), 0.0);
    sol.col_dual.assign(static_cast<size_t>(n), 0.0);
    sol.col_basis_status.assign(static_cast<size_t>(n), BasisStatus::kNonbasicFree);

    sol.row_value.assign(static_cast<size_t>(m), 0.0);
    sol.row_dual.assign(static_cast<size_t>(m), 0.0);
    sol.row_basis_status.assign(static_cast<size_t>(m), BasisStatus::kBasic);

    // 1. Map kept columns
    for (size_t red_c = 0; red_c < stack.reduced_to_orig_col.size(); ++red_c) {
        const int orig_c = stack.reduced_to_orig_col[red_c];
        if (red_c < red_sol.col_value.size()) {
            sol.col_value[static_cast<size_t>(orig_c)] = red_sol.col_value[red_c];
        }
        if (red_c < red_sol.col_dual.size()) {
            sol.col_dual[static_cast<size_t>(orig_c)] = red_sol.col_dual[red_c];
        }
        if (red_c < red_sol.col_basis_status.size()) {
            sol.col_basis_status[static_cast<size_t>(orig_c)] = red_sol.col_basis_status[red_c];
        }
    }

    // 2. Restore fixed columns
    for (const auto& fc : stack.fixed_cols) {
        sol.col_value[static_cast<size_t>(fc.orig_col_idx)] = fc.fixed_val;
        sol.col_basis_status[static_cast<size_t>(fc.orig_col_idx)] = BasisStatus::kFixed;
    }

    // 3. Map kept rows
    for (size_t red_r = 0; red_r < stack.reduced_to_orig_row.size(); ++red_r) {
        const int orig_r = stack.reduced_to_orig_row[red_r];
        if (red_r < red_sol.row_value.size()) {
            sol.row_value[static_cast<size_t>(orig_r)] = red_sol.row_value[red_r];
        }
        if (red_r < red_sol.row_dual.size()) {
            sol.row_dual[static_cast<size_t>(orig_r)] = red_sol.row_dual[red_r];
        }
        if (red_r < red_sol.row_basis_status.size()) {
            sol.row_basis_status[static_cast<size_t>(orig_r)] = red_sol.row_basis_status[red_r];
        }
    }

    // 3.5. Postsolve singleton rows: assign shadow prices
    for (const auto& sr : stack.singleton_rows) {
        const int orig_r = sr.orig_row_idx;
        const int j = sr.col_idx;
        const double a = sr.coeff;
        if (std::abs(a) < 1e-15) continue;

        // Compute (A^T_{-i} y)_j
        double Aty_excl = 0.0;
        const auto rows = orig.A.col_rows(j);
        const auto vals = orig.A.col_vals(j);
        for (size_t idx = 0; idx < rows.size(); ++idx) {
            const int r = rows[idx];
            if (r != orig_r) {
                Aty_excl += vals[idx] * sol.row_dual[static_cast<size_t>(r)];
            }
        }

        const double sense_factor = (orig.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
        const double c_eff = sense_factor * orig.c[static_cast<size_t>(j)];
        const double dj0 = c_eff - Aty_excl; // reduced cost if yi = 0

        const double xj = sol.col_value[static_cast<size_t>(j)];
        const double lj = orig.col_lower[static_cast<size_t>(j)];
        const double uj = orig.col_upper[static_cast<size_t>(j)];
        const bool is_equality = (std::abs(sr.row_lower - sr.row_upper) <= tol::kZeroDrop);

        double yi = 0.0;
        if (is_equality) {
            // Equality singleton row: free to take any shadow price to zero out dj
            yi = dj0 / a;
            sol.row_basis_status[static_cast<size_t>(orig_r)] = BasisStatus::kFixed;
        } else {
            if (xj > lj + 1e-6 && xj < uj - 1e-6) {
                yi = dj0 / a;
            } else if (xj <= lj + 1e-6 && dj0 < -tol::kDualFeasibility) {
                yi = dj0 / a;
            } else if (xj >= uj - 1e-6 && dj0 > tol::kDualFeasibility) {
                yi = dj0 / a;
            }

            // Respect dual sign restrictions for inequality rows
            if (sr.row_lower <= -1e19 && sr.row_upper < 1e19) {
                if (yi > 0.0) yi = 0.0;
                sol.row_basis_status[static_cast<size_t>(orig_r)] = (std::abs(yi) > 1e-9) ? BasisStatus::kAtUpper : BasisStatus::kBasic;
            } else if (sr.row_lower > -1e19 && sr.row_upper >= 1e19) {
                if (yi < 0.0) yi = 0.0;
                sol.row_basis_status[static_cast<size_t>(orig_r)] = (std::abs(yi) > 1e-9) ? BasisStatus::kAtLower : BasisStatus::kBasic;
            }
        }
        sol.row_dual[static_cast<size_t>(orig_r)] = yi;
    }

    // 4. Compute row activities from definition: Ax
    orig.A.multiply(sol.col_value, sol.row_value);

    // 5. Compute reduced costs from definition: d_j = c_j' - A_{\cdot j}^T y
    std::vector<double> Aty(static_cast<size_t>(n), 0.0);
    orig.A.multiply_transpose(sol.row_dual, Aty);

    const double sense_factor = (orig.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
    for (int j = 0; j < n; ++j) {
        const double c_eff = sense_factor * orig.c[static_cast<size_t>(j)];
        const double dj = c_eff - Aty[static_cast<size_t>(j)];
        sol.col_dual[static_cast<size_t>(j)] = (orig.sense == ObjSense::kMaximize) ? -dj : dj;
    }

    sol.recompute_quality(orig);
    return sol;
}

} // namespace indus::presolve
