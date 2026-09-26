#include "indus/pdhg.hpp"
#include "src/linalg/scaling.hpp"
#include "indus/tolerances.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

namespace indus::pdhg {

double estimate_spectral_norm(const la::SparseMatrixCSC& A, int max_iter, double tol) {
    const int m = A.m;
    const int n = A.n;
    if (m == 0 || n == 0 || A.nnz() == 0) {
        return 1.0;
    }

    std::vector<double> v(static_cast<size_t>(n), 1.0 / std::sqrt(static_cast<double>(n)));
    std::vector<double> w(static_cast<size_t>(m), 0.0);
    std::vector<double> u(static_cast<size_t>(n), 0.0);

    double lambda_prev = 0.0;
    double lambda = 1.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // w = A * v
        A.multiply(v, w);

        // u = A^T * w
        A.multiply_transpose(w, u);

        // Compute norm of u
        double norm_sq = 0.0;
        for (int j = 0; j < n; ++j) {
            norm_sq += u[static_cast<size_t>(j)] * u[static_cast<size_t>(j)];
        }
        const double norm = std::sqrt(norm_sq);
        if (norm < 1e-15 || std::isnan(norm) || std::isinf(norm)) {
            return (lambda > 1e-6) ? std::sqrt(lambda) : 1e-3;
        }

        lambda = norm;
        if (iter > 0 && std::abs(lambda - lambda_prev) <= tol * lambda) {
            break;
        }
        lambda_prev = lambda;

        const double inv_norm = 1.0 / norm;
        for (int j = 0; j < n; ++j) {
            v[static_cast<size_t>(j)] = u[static_cast<size_t>(j)] * inv_norm;
        }
    }

    const double sigma_max = std::sqrt(lambda);
    return (sigma_max > 1e-12) ? sigma_max : 1e-6;
}

CpuPdhgSolver::CpuPdhgSolver(const Model& model, const Options& options)
    : original_model_(model), options_(options) {
    pdhg_opts_.max_iterations = options.iteration_limit;
    pdhg_opts_.time_limit = options.time_limit;
    pdhg_opts_.primal_tolerance = options.get_double("tolerance", 1e-6);
    pdhg_opts_.dual_tolerance = options.get_double("tolerance", 1e-6);
    pdhg_opts_.gap_tolerance = options.get_double("gap_tolerance", 1e-6);
    pdhg_opts_.check_frequency = static_cast<int>(options.get_int("check_frequency", 100));
    pdhg_opts_.enable_scaling = options.enable_scaling;
    pdhg_opts_.verbose = options.get_bool("verbose", false);
}

Solution CpuPdhgSolver::solve() {
    const auto start_time = std::chrono::high_resolution_clock::now();
    diag_ = {};

    const int m = original_model_.num_rows;
    const int n = original_model_.num_cols;
    const bool is_max = (original_model_.sense == ObjSense::kMaximize);

    Solution sol;
    sol.algorithm_used = "pdhg_cpu";
    sol.col_value.resize(static_cast<size_t>(n), 0.0);
    sol.row_value.resize(static_cast<size_t>(m), 0.0);
    sol.row_dual.resize(static_cast<size_t>(m), 0.0);
    sol.col_dual.resize(static_cast<size_t>(n), 0.0);
    sol.col_basis_status.resize(static_cast<size_t>(n), BasisStatus::kUnknown);
    sol.row_basis_status.resize(static_cast<size_t>(m), BasisStatus::kUnknown);

    // Trivial dimensions check
    if (n == 0) {
        sol.status = SolveStatus::kOptimal;
        sol.objective_value = original_model_.objective_offset;
        sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        diag_.converged = true;
        return sol;
    }

    if (m == 0) {
        // Trivial unconstrained column-wise solve
        double obj = original_model_.objective_offset;
        for (int j = 0; j < n; ++j) {
            const double cj = is_max ? -original_model_.c[static_cast<size_t>(j)] : original_model_.c[static_cast<size_t>(j)];
            const double lj = original_model_.col_lower[static_cast<size_t>(j)];
            const double uj = original_model_.col_upper[static_cast<size_t>(j)];
            if (cj > tol::kDualFeasibility) {
                if (lj <= -1e20) {
                    sol.status = is_max ? SolveStatus::kUnbounded : SolveStatus::kUnbounded;
                    sol.status_message = "Unbounded column in unconstrained LP";
                    return sol;
                }
                sol.col_value[static_cast<size_t>(j)] = lj;
            } else if (cj < -tol::kDualFeasibility) {
                if (uj >= 1e20) {
                    sol.status = is_max ? SolveStatus::kUnbounded : SolveStatus::kUnbounded;
                    sol.status_message = "Unbounded column in unconstrained LP";
                    return sol;
                }
                sol.col_value[static_cast<size_t>(j)] = uj;
            } else {
                sol.col_value[static_cast<size_t>(j)] = (lj > 0.0) ? lj : ((uj < 0.0) ? uj : 0.0);
            }
            obj += original_model_.c[static_cast<size_t>(j)] * sol.col_value[static_cast<size_t>(j)];
        }
        sol.status = SolveStatus::kOptimal;
        sol.objective_value = obj;
        sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        diag_.converged = true;
        return sol;
    }

    // Prepare scaled or working copies of model data
    la::ScalingFactors scaling_factors;
    la::SparseMatrixCSC work_A = original_model_.A;
    std::vector<double> work_c = original_model_.c;
    if (is_max) {
        for (double& cj : work_c) cj = -cj;
    }
    std::vector<double> work_row_lower = original_model_.row_lower;
    std::vector<double> work_row_upper = original_model_.row_upper;
    std::vector<double> work_col_lower = original_model_.col_lower;
    std::vector<double> work_col_upper = original_model_.col_upper;

    const auto setup_start = std::chrono::high_resolution_clock::now();
    if (pdhg_opts_.enable_scaling) {
        scaling_factors = la::RuizScaler::compute_and_scale(
            work_A, work_c,
            work_row_lower, work_row_upper,
            work_col_lower, work_col_upper,
            10
        );
    }

    // Estimate spectral norm ||A||_2
    const double spectral_norm = estimate_spectral_norm(work_A, 25, 1e-4);
    diag_.spectral_norm_estimate = spectral_norm;

    // Step sizes: tau * sigma * ||A||^2 < 1
    const double safety = std::clamp(pdhg_opts_.step_size_safety, 0.5, 0.99);
    double tau = safety / spectral_norm;
    double sigma = safety / spectral_norm;
    diag_.tau = tau;
    diag_.sigma = sigma;

    // Iterate vectors
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    std::vector<double> x_bar(static_cast<size_t>(n), 0.0);
    std::vector<double> y(static_cast<size_t>(m), 0.0);

    for (int j = 0; j < n; ++j) {
        const double lj = work_col_lower[static_cast<size_t>(j)];
        const double uj = work_col_upper[static_cast<size_t>(j)];
        double val = 0.0;
        if (val < lj) val = lj;
        if (val > uj) val = uj;
        x[static_cast<size_t>(j)] = val;
        x_bar[static_cast<size_t>(j)] = val;
    }

    std::vector<double> w(static_cast<size_t>(m), 0.0); // A * x_bar
    std::vector<double> v(static_cast<size_t>(n), 0.0); // A^T * y
    std::vector<double> Ax(static_cast<size_t>(m), 0.0);

    std::vector<double> best_x = x;
    std::vector<double> best_y = y;
    double best_infeasibility = 1e20;
    int stagnation_count = 0;

    const auto setup_end = std::chrono::high_resolution_clock::now();
    diag_.setup_time_sec = std::chrono::duration<double>(setup_end - setup_start).count();

    const auto iter_start = std::chrono::high_resolution_clock::now();
    bool converged = false;
    int64_t iter = 0;

    const int64_t max_iter = pdhg_opts_.max_iterations;
    const int check_freq = std::max(1, pdhg_opts_.check_frequency);

    for (iter = 1; iter <= max_iter; ++iter) {
        // 1. Dual Step: Forward SpMV w = A * x_bar
        work_A.multiply(x_bar, w);

        // 2. Dual Update & Moreau projection
        // y_i^{k+1} = y_hat - sigma * clamp(y_hat / sigma, l_i, u_i)
        bool has_nan = false;
        for (int i = 0; i < m; ++i) {
            const double y_hat = y[static_cast<size_t>(i)] + sigma * w[static_cast<size_t>(i)];
            const double li = work_row_lower[static_cast<size_t>(i)];
            const double ui = work_row_upper[static_cast<size_t>(i)];
            const double val = y_hat / sigma;
            const double clamped = (val < li) ? li : ((val > ui) ? ui : val);
            const double y_new = y_hat - sigma * clamped;
            if (std::isnan(y_new) || std::isinf(y_new)) {
                has_nan = true;
            }
            y[static_cast<size_t>(i)] = y_new;
        }

        if (has_nan) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "PDHG dual iterate encountered NaN/Inf";
            break;
        }

        // 3. Primal Step: Transpose SpMV v = A^T * y
        work_A.multiply_transpose(y, v);

        // 4. Primal Update & Bound Clipping & Extrapolation
        // x_j^{k+1} = clamp(x_j^k - tau * (v_j + c_j), l_j, u_j)
        // x_bar_j^{k+1} = 2 * x_j^{k+1} - x_j^k
        for (int j = 0; j < n; ++j) {
            const double x_old = x[static_cast<size_t>(j)];
            const double grad = v[static_cast<size_t>(j)] + work_c[static_cast<size_t>(j)];
            double x_new = x_old - tau * grad;
            const double lj = work_col_lower[static_cast<size_t>(j)];
            const double uj = work_col_upper[static_cast<size_t>(j)];
            if (x_new < lj) x_new = lj;
            if (x_new > uj) x_new = uj;
            if (std::isnan(x_new) || std::isinf(x_new)) {
                has_nan = true;
            }
            x[static_cast<size_t>(j)] = x_new;
            x_bar[static_cast<size_t>(j)] = 2.0 * x_new - x_old;
        }

        if (has_nan) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "PDHG primal iterate encountered NaN/Inf";
            break;
        }

        // 5. Periodic Convergence & Adaptive Restart Check
        if (iter % check_freq == 0 || iter == max_iter) {
            const auto now = std::chrono::high_resolution_clock::now();
            const double elapsed = std::chrono::duration<double>(now - start_time).count();
            if (elapsed > pdhg_opts_.time_limit) {
                sol.status = SolveStatus::kTimeLimit;
                sol.status_message = "PDHG time limit reached";
                break;
            }

            // Compute primal row violation
            work_A.multiply(x, Ax);
            double max_prim_viol = 0.0;
            for (int i = 0; i < m; ++i) {
                const double axi = Ax[static_cast<size_t>(i)];
                const double li = work_row_lower[static_cast<size_t>(i)];
                const double ui = work_row_upper[static_cast<size_t>(i)];
                if (axi < li) max_prim_viol = std::max(max_prim_viol, li - axi);
                if (axi > ui) max_prim_viol = std::max(max_prim_viol, axi - ui);
            }
            for (int j = 0; j < n; ++j) {
                const double xj = x[static_cast<size_t>(j)];
                const double lj = work_col_lower[static_cast<size_t>(j)];
                const double uj = work_col_upper[static_cast<size_t>(j)];
                if (xj < lj) max_prim_viol = std::max(max_prim_viol, lj - xj);
                if (xj > uj) max_prim_viol = std::max(max_prim_viol, xj - uj);
            }

            // Compute dual reduced cost violation
            // Reduced cost for minimization: dj = work_c[j] + (A^T y)_j
            double max_dual_viol = 0.0;
            double dual_obj = 0.0;
            for (int i = 0; i < m; ++i) {
                const double yi = y[static_cast<size_t>(i)];
                const double li = work_row_lower[static_cast<size_t>(i)];
                const double ui = work_row_upper[static_cast<size_t>(i)];
                if (yi > 1e-12 && ui < 1e20) {
                    dual_obj -= yi * ui;
                } else if (yi < -1e-12 && li > -1e20) {
                    dual_obj -= yi * li;
                }
            }

            for (int j = 0; j < n; ++j) {
                const double dj = work_c[static_cast<size_t>(j)] + v[static_cast<size_t>(j)];
                const double xj = x[static_cast<size_t>(j)];
                const double lj = work_col_lower[static_cast<size_t>(j)];
                const double uj = work_col_upper[static_cast<size_t>(j)];

                if (xj > lj + 1e-5 && dj > 1e-6) {
                    max_dual_viol = std::max(max_dual_viol, dj);
                }
                if (xj < uj - 1e-5 && dj < -1e-6) {
                    max_dual_viol = std::max(max_dual_viol, -dj);
                }
                if (lj <= -1e20 && uj >= 1e20) {
                    max_dual_viol = std::max(max_dual_viol, std::abs(dj));
                }

                if (dj > 1e-12 && lj > -1e20) {
                    dual_obj += dj * lj;
                } else if (dj < -1e-12 && uj < 1e20) {
                    dual_obj += dj * uj;
                }
            }

            // Primal objective
            double primal_obj = 0.0;
            for (int j = 0; j < n; ++j) {
                primal_obj += work_c[static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
            }

            const double gap = std::abs(primal_obj - dual_obj) / (1.0 + std::abs(primal_obj) + std::abs(dual_obj));

            diag_.primal_residual = max_prim_viol;
            diag_.dual_residual = max_dual_viol;
            diag_.duality_gap = gap;

            const double total_infeas = max_prim_viol + max_dual_viol;
            if (total_infeas < best_infeasibility) {
                best_infeasibility = total_infeas;
                best_x = x;
                best_y = y;
                stagnation_count = 0;
            } else {
                stagnation_count += check_freq;
            }

            // Check stopping criteria
            if (max_prim_viol <= pdhg_opts_.primal_tolerance &&
                max_dual_viol <= pdhg_opts_.dual_tolerance &&
                gap <= pdhg_opts_.gap_tolerance) {
                converged = true;
                diag_.converged = true;
                best_x = x;
                best_y = y;
                sol.status = SolveStatus::kOptimal;
                sol.status_message = "PDHG converged to optimal primal and dual tolerance";
                break;
            }

            // Adaptive step size rebalancing & restart
            if (max_prim_viol > 10.0 * max_dual_viol && max_dual_viol > 0.0) {
                sigma = std::min(sigma * 1.3, 1e6);
                tau = (safety * safety) / (spectral_norm * spectral_norm * sigma);
            } else if (max_dual_viol > 10.0 * max_prim_viol && max_prim_viol > 0.0) {
                tau = std::min(tau * 1.3, 1e6);
                sigma = (safety * safety) / (spectral_norm * spectral_norm * tau);
            }

            if (stagnation_count >= pdhg_opts_.restart_stagnation_limit) {
                // Adaptive restart: reset extrapolation
                x_bar = x;
                diag_.restart_count++;
                stagnation_count = 0;
            }
        }
    }

    const auto iter_end = std::chrono::high_resolution_clock::now();
    diag_.iteration_time_sec = std::chrono::duration<double>(iter_end - iter_start).count();
    diag_.total_time_sec = std::chrono::duration<double>(iter_end - start_time).count();
    diag_.iterations = iter;

    if (!converged && sol.status == SolveStatus::kNotSolved) {
        if (iter >= max_iter) {
            sol.status = (best_infeasibility <= 1e-4) ? SolveStatus::kFeasible : SolveStatus::kIterationLimit;
            sol.status_message = "PDHG reached iteration limit (" + std::to_string(max_iter) + ")";
        }
    }

    // Unscale iterates to recover original problem solution
    std::vector<double> sol_x = best_x;
    std::vector<double> sol_y_raw = best_y;

    if (scaling_factors.is_scaled) {
        la::RuizScaler::unscale_primal(scaling_factors, sol_x);
        la::RuizScaler::unscale_dual(scaling_factors, sol_y_raw);
    }

    // Assign primal variables
    sol.col_value = sol_x;

    // Compute row activities on original model: Ax
    original_model_.A.multiply(sol.col_value, sol.row_value);

    // Compute true objective value on original model
    double unscaled_obj = original_model_.objective_offset;
    for (int j = 0; j < n; ++j) {
        unscaled_obj += original_model_.c[static_cast<size_t>(j)] * sol.col_value[static_cast<size_t>(j)];
    }
    sol.objective_value = unscaled_obj;
    sol.best_dual_bound = unscaled_obj;

    // Convert dual multipliers:
    // PDHG minimized with term + y^T (Ax - b).
    // Verifier / simplex dual multipliers use convention:
    //   <= row (li = -inf, ui < inf): yi <= 0
    //   >= row (li > -inf, ui = inf): yi >= 0
    // Thus y_sol = -y_pdhg.
    for (int i = 0; i < m; ++i) {
        sol.row_dual[static_cast<size_t>(i)] = is_max ? sol_y_raw[static_cast<size_t>(i)] : -sol_y_raw[static_cast<size_t>(i)];
    }

    // Compute reduced costs: d = c - A^T y_sol
    std::vector<double> a_trans_y(static_cast<size_t>(n), 0.0);
    original_model_.A.multiply_transpose(sol.row_dual, a_trans_y);
    for (int j = 0; j < n; ++j) {
        const double dj = original_model_.c[static_cast<size_t>(j)] - a_trans_y[static_cast<size_t>(j)];
        sol.col_dual[static_cast<size_t>(j)] = dj;

        // Basis status heuristics for continuous bounds
        const double xj = sol.col_value[static_cast<size_t>(j)];
        const double lj = original_model_.col_lower[static_cast<size_t>(j)];
        const double uj = original_model_.col_upper[static_cast<size_t>(j)];
        if (std::abs(lj - uj) <= tol::kZeroDrop) {
            sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kFixed;
        } else if (std::abs(xj - lj) <= 1e-5) {
            sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kAtLower;
        } else if (std::abs(xj - uj) <= 1e-5) {
            sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kAtUpper;
        } else {
            sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kBasic;
        }
    }

    sol.iterations = iter;
    sol.solve_time_seconds = diag_.total_time_sec;
    sol.recompute_quality(original_model_);

    return sol;
}

static thread_local PdhgDiagnostics g_last_cpu_pdhg_diag;

const PdhgDiagnostics& get_last_cpu_pdhg_diagnostics() noexcept {
    return g_last_cpu_pdhg_diag;
}

Solution solve_pdhg_cpu(const Model& model, const Options& options) {
    CpuPdhgSolver solver(model, options);
    Solution sol = solver.solve();
    g_last_cpu_pdhg_diag = solver.diagnostics();
    return sol;
}

} // namespace indus::pdhg
