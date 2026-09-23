#include "src/solvers/simplex/simplex_core.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace indus::simplex {

SimplexResult SimplexCore::solve_primal(int64_t max_iterations) {
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

    compute_primal_values();

    int64_t total_iterations = 0;

    // ========================================================================
    // Phase 1: Composite Phase-1 (No Big-M)
    // ========================================================================
    bool needs_phase1 = false;
    for (int j = 0; j < num_vars; ++j) {
        const double x = basis_.primal[static_cast<size_t>(j)];
        const double l = model_.lower[static_cast<size_t>(j)];
        const double u = model_.upper[static_cast<size_t>(j)];
        if (x < l - tol::kPrimalFeasibility || x > u + tol::kPrimalFeasibility) {
            needs_phase1 = true;
            break;
        }
    }

    const std::vector<double> orig_cost = model_.cost;

    if (needs_phase1) {
        int64_t p1_iter = 0;
        constexpr int64_t kMaxPhase1Iters = 50000;

        while (p1_iter < kMaxPhase1Iters && total_iterations < max_iterations) {
            p1_iter++;
            total_iterations++;

            // 1. Setup composite Phase-1 cost vector
            double total_infeasibility = 0.0;
            for (int j = 0; j < num_vars; ++j) {
                const double x = basis_.primal[static_cast<size_t>(j)];
                const double l = model_.lower[static_cast<size_t>(j)];
                const double u = model_.upper[static_cast<size_t>(j)];

                if (x < l - tol::kPrimalFeasibility) {
                    model_.cost[static_cast<size_t>(j)] = -1.0;
                    total_infeasibility += (l - x);
                } else if (x > u + tol::kPrimalFeasibility) {
                    model_.cost[static_cast<size_t>(j)] = 1.0;
                    total_infeasibility += (x - u);
                } else {
                    model_.cost[static_cast<size_t>(j)] = 0.0;
                }
            }

            if (total_infeasibility <= tol::kPrimalFeasibility) {
                // Primal feasible! Phase 1 complete
                break;
            }

            // 2. Dual values & reduced costs for Phase 1
            compute_dual_values();
            compute_reduced_costs();

            // 3. Pricing: select entering variable q
            int enter_var = -1;
            double best_dj = 0.0;
            double enter_dir = 0.0; // +1 if increasing, -1 if decreasing

            for (int j = 0; j < num_vars; ++j) {
                if (basis_.basic_index[static_cast<size_t>(j)] != -1) continue;
                if (basis_.status[static_cast<size_t>(j)] == BasisStatus::kFixed) continue;

                const double dj = basis_.reduced_cost[static_cast<size_t>(j)];
                const auto st = basis_.status[static_cast<size_t>(j)];

                if (st == BasisStatus::kAtLower && dj < -tol::kDualFeasibility) {
                    if (dj < best_dj) {
                        best_dj = dj;
                        enter_var = j;
                        enter_dir = 1.0;
                    }
                } else if (st == BasisStatus::kAtUpper && dj > tol::kDualFeasibility) {
                    if (-dj < best_dj) {
                        best_dj = -dj;
                        enter_var = j;
                        enter_dir = -1.0;
                    }
                } else if (st == BasisStatus::kNonbasicFree && std::abs(dj) > tol::kDualFeasibility) {
                    if (-std::abs(dj) < best_dj) {
                        best_dj = -std::abs(dj);
                        enter_var = j;
                        enter_dir = (dj < 0.0) ? 1.0 : -1.0;
                    }
                }
            }

            if (enter_var == -1) {
                // Phase 1 optimal but total_infeasibility > 0 -> Infeasible!
                result.status = SolveStatus::kInfeasible;
                result.status_message = "Primal infeasible (Phase 1 objective > 0)";
                result.certificate_type = "farkas";
                result.certificate_vector = basis_.dual;
                model_.cost = orig_cost;
                return result;
            }

            // 4. Pivot column FTRAN
            std::vector<double> alpha;
            ftran(enter_var, alpha);

            // 5. Harris Two-Pass Ratio Test
            double theta_max = 1e20;
            const double l_q = model_.lower[static_cast<size_t>(enter_var)];
            const double u_q = model_.upper[static_cast<size_t>(enter_var)];

            if (enter_dir > 0 && u_q < 1e20) {
                theta_max = u_q - l_q;
            } else if (enter_dir < 0 && l_q > -1e20) {
                theta_max = u_q - l_q;
            }

            // Pass 1: compute maximum feasible step with tolerance
            for (int i = 0; i < m; ++i) {
                const int var = basis_.basic_vars[static_cast<size_t>(i)];
                const double x = basis_.primal[static_cast<size_t>(var)];
                const double l = model_.lower[static_cast<size_t>(var)];
                const double u = model_.upper[static_cast<size_t>(var)];
                const double a = alpha[static_cast<size_t>(i)] * enter_dir;

                if (a > tol::kZeroDrop && l > -1e20) {
                    const double step = (x - l + tol::kPrimalFeasibility) / a;
                    if (step < theta_max) theta_max = step;
                } else if (a < -tol::kZeroDrop && u < 1e20) {
                    const double step = (u - x + tol::kPrimalFeasibility) / (-a);
                    if (step < theta_max) theta_max = step;
                }
            }

            // Pass 2: select row with largest pivot element among candidates within theta_max
            int leave_row = -1;
            double max_pivot = 0.0;
            double actual_step = theta_max;

            for (int i = 0; i < m; ++i) {
                const int var = basis_.basic_vars[static_cast<size_t>(i)];
                const double x = basis_.primal[static_cast<size_t>(var)];
                const double l = model_.lower[static_cast<size_t>(var)];
                const double u = model_.upper[static_cast<size_t>(var)];
                const double a = alpha[static_cast<size_t>(i)] * enter_dir;

                if (a > tol::kZeroDrop && l > -1e20) {
                    const double step = (x - l) / a;
                    if (step <= theta_max) {
                        if (std::abs(a) > max_pivot) {
                            max_pivot = std::abs(a);
                            leave_row = i;
                            actual_step = std::max(0.0, step);
                        }
                    }
                } else if (a < -tol::kZeroDrop && u < 1e20) {
                    const double step = (u - x) / (-a);
                    if (step <= theta_max) {
                        if (std::abs(a) > max_pivot) {
                            max_pivot = std::abs(a);
                            leave_row = i;
                            actual_step = std::max(0.0, step);
                        }
                    }
                }
            }

            // Check if entering variable reached its own opposite bound
            if (leave_row == -1 && theta_max < 1e19) {
                // Bound flip of entering variable
                if (enter_dir > 0) {
                    basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kAtUpper;
                    basis_.primal[static_cast<size_t>(enter_var)] = u_q;
                } else {
                    basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kAtLower;
                    basis_.primal[static_cast<size_t>(enter_var)] = l_q;
                }
                compute_primal_values();
                continue;
            }

            if (leave_row == -1) {
                // Should not happen in Phase 1 bounded form
                break;
            }

            // Pivot execution
            const int leave_var = basis_.basic_vars[static_cast<size_t>(leave_row)];
            const double a_pq = alpha[static_cast<size_t>(leave_row)];

            basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kBasic;
            const double new_leave_val = basis_.primal[static_cast<size_t>(leave_var)] - actual_step * a_pq * enter_dir;

            if (std::abs(new_leave_val - model_.lower[static_cast<size_t>(leave_var)]) <=
                std::abs(new_leave_val - model_.upper[static_cast<size_t>(leave_var)])) {
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
            if (!pfi_ok || lu_.needs_refactorization() || (p1_iter % 50 == 0)) {
                refactorize_basis();
            }

            compute_primal_values();
        }
    }

    // Restore original cost vector
    model_.cost = orig_cost;

    // ========================================================================
    // Phase 2: Primal Simplex Optimization
    // ========================================================================
    int64_t p2_iter = 0;
    while (total_iterations < max_iterations) {
        p2_iter++;
        total_iterations++;

        compute_dual_values();
        compute_reduced_costs();

        // 1. Pricing: select entering variable q
        int enter_var = -1;
        double best_dj = 0.0;
        double enter_dir = 0.0;

        for (int j = 0; j < num_vars; ++j) {
            if (basis_.basic_index[static_cast<size_t>(j)] != -1) continue;
            if (basis_.status[static_cast<size_t>(j)] == BasisStatus::kFixed) continue;

            const double dj = basis_.reduced_cost[static_cast<size_t>(j)];
            const auto st = basis_.status[static_cast<size_t>(j)];

            if (st == BasisStatus::kAtLower && dj < -tol::kDualFeasibility) {
                if (dj < best_dj) {
                    best_dj = dj;
                    enter_var = j;
                    enter_dir = 1.0;
                }
            } else if (st == BasisStatus::kAtUpper && dj > tol::kDualFeasibility) {
                if (-dj < best_dj) {
                    best_dj = -dj;
                    enter_var = j;
                    enter_dir = -1.0;
                }
            } else if (st == BasisStatus::kNonbasicFree && std::abs(dj) > tol::kDualFeasibility) {
                if (-std::abs(dj) < best_dj) {
                    best_dj = -std::abs(dj);
                    enter_var = j;
                    enter_dir = (dj < 0.0) ? 1.0 : -1.0;
                }
            }
        }

        // If no entering variable found, solution is optimal!
        if (enter_var == -1) {
            result.status = SolveStatus::kOptimal;
            result.status_message = "Optimal solution found";
            break;
        }

        // 2. Pivot column FTRAN
        std::vector<double> alpha;
        ftran(enter_var, alpha);

        // 3. Harris Two-Pass Ratio Test
        double theta_max = 1e20;
        const double l_q = model_.lower[static_cast<size_t>(enter_var)];
        const double u_q = model_.upper[static_cast<size_t>(enter_var)];

        if (enter_dir > 0 && u_q < 1e20) {
            theta_max = u_q - l_q;
        } else if (enter_dir < 0 && l_q > -1e20) {
            theta_max = u_q - l_q;
        }

        for (int i = 0; i < m; ++i) {
            const int var = basis_.basic_vars[static_cast<size_t>(i)];
            const double x = basis_.primal[static_cast<size_t>(var)];
            const double l = model_.lower[static_cast<size_t>(var)];
            const double u = model_.upper[static_cast<size_t>(var)];
            const double a = alpha[static_cast<size_t>(i)] * enter_dir;

            if (a > tol::kZeroDrop && l > -1e20) {
                const double step = (x - l + tol::kPrimalFeasibility) / a;
                if (step < theta_max) theta_max = step;
            } else if (a < -tol::kZeroDrop && u < 1e20) {
                const double step = (u - x + tol::kPrimalFeasibility) / (-a);
                if (step < theta_max) theta_max = step;
            }
        }

        if (theta_max >= 1e19) {
            // Unbounded ray!
            result.status = SolveStatus::kUnbounded;
            result.status_message = "Primal unbounded (extreme ray found)";
            result.certificate_type = "ray";
            result.certificate_vector.assign(static_cast<size_t>(n), 0.0);
            if (enter_var < n) {
                result.certificate_vector[static_cast<size_t>(enter_var)] = enter_dir;
            }
            for (int i = 0; i < m; ++i) {
                const int var = basis_.basic_vars[static_cast<size_t>(i)];
                if (var < n) {
                    result.certificate_vector[static_cast<size_t>(var)] = -alpha[static_cast<size_t>(i)] * enter_dir;
                }
            }
            break;
        }

        // Pass 2
        int leave_row = -1;
        double max_pivot = 0.0;
        double actual_step = theta_max;

        for (int i = 0; i < m; ++i) {
            const int var = basis_.basic_vars[static_cast<size_t>(i)];
            const double x = basis_.primal[static_cast<size_t>(var)];
            const double l = model_.lower[static_cast<size_t>(var)];
            const double u = model_.upper[static_cast<size_t>(var)];
            const double a = alpha[static_cast<size_t>(i)] * enter_dir;

            if (a > tol::kZeroDrop && l > -1e20) {
                const double step = (x - l) / a;
                if (step <= theta_max) {
                    if (std::abs(a) > max_pivot) {
                        max_pivot = std::abs(a);
                        leave_row = i;
                        actual_step = std::max(0.0, step);
                    }
                }
            } else if (a < -tol::kZeroDrop && u < 1e20) {
                const double step = (u - x) / (-a);
                if (step <= theta_max) {
                    if (std::abs(a) > max_pivot) {
                        max_pivot = std::abs(a);
                        leave_row = i;
                        actual_step = std::max(0.0, step);
                    }
                }
            }
        }

        if (leave_row == -1) {
            // Flip entering variable to opposite bound
            if (enter_dir > 0) {
                basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kAtUpper;
                basis_.primal[static_cast<size_t>(enter_var)] = u_q;
            } else {
                basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kAtLower;
                basis_.primal[static_cast<size_t>(enter_var)] = l_q;
            }
            compute_primal_values();
            continue;
        }

        // Pivot execution
        const int leave_var = basis_.basic_vars[static_cast<size_t>(leave_row)];
        const double a_pq = alpha[static_cast<size_t>(leave_row)];

        basis_.status[static_cast<size_t>(enter_var)] = BasisStatus::kBasic;
        const double new_leave_val = basis_.primal[static_cast<size_t>(leave_var)] - actual_step * a_pq * enter_dir;

        if (std::abs(new_leave_val - model_.lower[static_cast<size_t>(leave_var)]) <=
            std::abs(new_leave_val - model_.upper[static_cast<size_t>(leave_var)])) {
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
        if (!pfi_ok || lu_.needs_refactorization() || (p2_iter % 50 == 0)) {
            refactorize_basis();
        }

        compute_primal_values();
    }

    if (total_iterations >= max_iterations && result.status == SolveStatus::kNotSolved) {
        result.status = SolveStatus::kIterationLimit;
        result.status_message = "Iteration limit reached";
    }

    result.iterations = total_iterations;
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
