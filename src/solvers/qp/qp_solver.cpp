#include "indus/qp.hpp"
#include "indus/tolerances.hpp"
#include "src/linalg/ldl.hpp"
#include "src/linalg/lu.hpp"
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <vector>
#include <set>
#include <unordered_set>

namespace indus::qp {

namespace {

enum class BoundState : uint8_t {
    kFree = 0,
    kAtLower = 1,
    kAtUpper = 2,
    kFixed = 3
};

} // namespace

QpConvexity check_convexity(const Model& model, std::string* message) {
    if (model.num_cols <= 0) {
        if (message) *message = "Zero variables";
        return QpConvexity::kPositiveSemidefinite;
    }
    if (!model.has_quadratic_objective()) {
        if (message) *message = "Linear objective (Q is zero)";
        return QpConvexity::kPositiveSemidefinite;
    }

    const int n = model.num_cols;
    la::SparseMatrixCSC Q_sym = model.get_symmetric_Q();

    // Check sense: for maximization, -Q must be positive semidefinite
    const double sign = (model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;

    std::vector<la::Triplet> test_triplets;
    test_triplets.reserve(static_cast<size_t>(Q_sym.nnz()));
    for (int j = 0; j < Q_sym.n; ++j) {
        const auto rows = Q_sym.col_rows(j);
        const auto vals = Q_sym.col_vals(j);
        for (size_t k = 0; k < rows.size(); ++k) {
            test_triplets.push_back({rows[k], j, sign * vals[k]});
        }
    }
    la::SparseMatrixCSC Q_test = la::SparseMatrixCSC::from_triplets(n, n, test_triplets, true);

    la::SparseLDL ldl;
    if (!ldl.factorize(Q_test, true)) {
        if (message) *message = "Factorization failed on Q matrix";
        return QpConvexity::kInvalid;
    }

    const auto& in = ldl.inertia();
    if (in.num_negative > 0) {
        if (message) {
            if (model.sense == ObjSense::kMaximize) {
                *message = "Non-concave maximization QP: Q has positive eigenvalues (" +
                           std::to_string(in.num_negative) + " negative in -Q)";
            } else {
                *message = "Nonconvex QP: Q matrix has " + std::to_string(in.num_negative) + " negative eigenvalues";
            }
        }
        return QpConvexity::kIndefinite;
    }

    if (in.num_positive == n && in.num_zero == 0) {
        if (message) *message = "Strictly convex (positive definite)";
        return QpConvexity::kPositiveDefinite;
    }

    if (message) *message = "Convex (positive semidefinite, " + std::to_string(in.num_zero) + " zero eigenvalues)";
    return QpConvexity::kPositiveSemidefinite;
}

Solution solve_qp(const Model& model, const Options& options) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    Solution sol;
    sol.algorithm_used = "convex_qp";

    auto elapsed_sec = [&]() -> double {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    };

    // 1. Model Validation
    try {
        model.validate();
    } catch (const std::exception& e) {
        sol.status = SolveStatus::kModelError;
        sol.status_message = std::string("Model validation error: ") + e.what();
        sol.solve_time_seconds = elapsed_sec();
        return sol;
    }

    // 2. Reject MIQP
    if (model.has_integers()) {
        sol.status = SolveStatus::kUnsupported;
        sol.status_message = "MIQP (Mixed-Integer Quadratic Programming) is unsupported in Phase 8";
        sol.solve_time_seconds = elapsed_sec();
        return sol;
    }

    // 3. Convexity check
    std::string conv_msg;
    const QpConvexity convexity = check_convexity(model, &conv_msg);
    if (convexity == QpConvexity::kIndefinite || convexity == QpConvexity::kNegativeDefinite || convexity == QpConvexity::kInvalid) {
        sol.status = SolveStatus::kModelError;
        sol.status_message = conv_msg;
        sol.solve_time_seconds = elapsed_sec();
        return sol;
    }

    const int n = model.num_cols;
    const int m = model.num_rows;

    if (n == 0) {
        sol.status = SolveStatus::kOptimal;
        sol.objective_value = model.objective_offset;
        sol.solve_time_seconds = elapsed_sec();
        sol.quality.is_primal_feasible = true;
        sol.quality.is_dual_feasible = true;
        sol.quality.is_stationary = true;
        return sol;
    }

    // Handle problem sense: internally minimize 0.5 * x^T Q_eff x + c_eff^T x
    const double sense_factor = (model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;
    la::SparseMatrixCSC Q_raw = model.get_symmetric_Q();
    
    // Scale Q and c by sense_factor
    std::vector<la::Triplet> Q_eff_triplets;
    Q_eff_triplets.reserve(static_cast<size_t>(Q_raw.nnz()));
    for (int j = 0; j < Q_raw.n; ++j) {
        const auto rows = Q_raw.col_rows(j);
        const auto vals = Q_raw.col_vals(j);
        for (size_t k = 0; k < rows.size(); ++k) {
            Q_eff_triplets.push_back({rows[k], j, sense_factor * vals[k]});
        }
    }
    la::SparseMatrixCSC Q_eff = la::SparseMatrixCSC::from_triplets(n, n, Q_eff_triplets, true);

    std::vector<double> c_eff(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        c_eff[static_cast<size_t>(j)] = sense_factor * model.c[static_cast<size_t>(j)];
    }

    // 4. Initial iterate x: if m == 0, bounds clamp is immediately feasible; else run Phase I LP
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    if (m == 0) {
        for (int j = 0; j < n; ++j) {
            x[static_cast<size_t>(j)] = std::clamp(0.0, model.col_lower[static_cast<size_t>(j)], model.col_upper[static_cast<size_t>(j)]);
        }
    } else {
        Model phase1_model;
        phase1_model.name = model.name + "_phase1";
        phase1_model.sense = ObjSense::kMinimize;
        phase1_model.num_rows = m;
        phase1_model.num_cols = n;
        phase1_model.A = model.A;
        phase1_model.c.assign(static_cast<size_t>(n), 0.0); // Zero objective for pure feasibility
        phase1_model.objective_offset = 0.0;
        phase1_model.row_lower = model.row_lower;
        phase1_model.row_upper = model.row_upper;
        phase1_model.col_lower = model.col_lower;
        phase1_model.col_upper = model.col_upper;

        Options phase1_opts = options;
        phase1_opts.algorithm = "dual_simplex";
        phase1_opts.time_limit = options.time_limit;

        Solution phase1_sol = solve(phase1_model, phase1_opts);
        if (phase1_sol.status == SolveStatus::kInfeasible) {
            sol.status = SolveStatus::kInfeasible;
            sol.status_message = "QP is primal infeasible (Phase I feasibility check failed)";
            sol.certificate_type = phase1_sol.certificate_type;
            sol.certificate_vector = phase1_sol.certificate_vector;
            sol.solve_time_seconds = elapsed_sec();
            return sol;
        }
        if (phase1_sol.status != SolveStatus::kOptimal && phase1_sol.status != SolveStatus::kFeasible) {
            sol.status = phase1_sol.status;
            sol.status_message = std::string("Phase I solver terminated with status: ") + to_string(phase1_sol.status);
            sol.solve_time_seconds = elapsed_sec();
            return sol;
        }

        x = phase1_sol.col_value;
        if (static_cast<int>(x.size()) != n) {
            x.assign(static_cast<size_t>(n), 0.0);
        }
        for (int j = 0; j < n; ++j) {
            x[static_cast<size_t>(j)] = std::clamp(x[static_cast<size_t>(j)], model.col_lower[static_cast<size_t>(j)], model.col_upper[static_cast<size_t>(j)]);
        }
    }

    // 5. Initialize Active Set Working Set
    std::vector<BoundState> col_state(static_cast<size_t>(n), BoundState::kFree);
    for (int j = 0; j < n; ++j) {
        const double lj = model.col_lower[static_cast<size_t>(j)];
        const double uj = model.col_upper[static_cast<size_t>(j)];
        if (std::abs(lj - uj) <= tol::kZeroDrop) {
            col_state[static_cast<size_t>(j)] = BoundState::kFixed;
        } else if (std::abs(x[static_cast<size_t>(j)] - lj) <= 1e-7) {
            col_state[static_cast<size_t>(j)] = BoundState::kAtLower;
        } else if (std::abs(x[static_cast<size_t>(j)] - uj) <= 1e-7) {
            col_state[static_cast<size_t>(j)] = BoundState::kAtUpper;
        }
    }

    std::vector<BoundState> row_state(static_cast<size_t>(m), BoundState::kFree);
    std::vector<int> active_rows;
    std::vector<double> Ax(static_cast<size_t>(m), 0.0);
    if (m > 0) {
        model.A.multiply(x, Ax);
        for (int i = 0; i < m; ++i) {
            const double li = model.row_lower[static_cast<size_t>(i)];
            const double ui = model.row_upper[static_cast<size_t>(i)];
            if (std::abs(li - ui) <= tol::kZeroDrop) {
                row_state[static_cast<size_t>(i)] = BoundState::kFixed;
                active_rows.push_back(i);
            } else if (std::abs(Ax[static_cast<size_t>(i)] - li) <= 1e-7) {
                row_state[static_cast<size_t>(i)] = BoundState::kAtLower;
                active_rows.push_back(i);
            } else if (std::abs(Ax[static_cast<size_t>(i)] - ui) <= 1e-7) {
                row_state[static_cast<size_t>(i)] = BoundState::kAtUpper;
                active_rows.push_back(i);
            }
        }
    }

    const int64_t max_iters = std::max<int64_t>(0, options.iteration_limit);
    const double qp_tol = 1e-7;
    int64_t iter = 0;

    std::vector<double> y_dual(static_cast<size_t>(m), 0.0);
    std::vector<double> z_dual(static_cast<size_t>(n), 0.0);

    // 6. Primal Active Set Loop
    while (iter < max_iters) {
        if (elapsed_sec() > options.time_limit) {
            sol.status = SolveStatus::kTimeLimit;
            sol.status_message = "Time limit exceeded during QP active-set solve";
            break;
        }
        iter++;

        // Identify free variables
        std::vector<int> free_cols;
        free_cols.reserve(static_cast<size_t>(n));
        std::vector<int> col_to_free(static_cast<size_t>(n), -1);
        for (int j = 0; j < n; ++j) {
            if (col_state[static_cast<size_t>(j)] == BoundState::kFree) {
                col_to_free[static_cast<size_t>(j)] = static_cast<int>(free_cols.size());
                free_cols.push_back(j);
            }
        }

        const int n_f = static_cast<int>(free_cols.size());
        const int m_w = static_cast<int>(active_rows.size());

        // Current QP gradient: g = Q_eff * x + c_eff
        std::vector<double> g = c_eff;
        Q_eff.multiply(x, g, 1.0, 1.0);

        std::vector<double> p(static_cast<size_t>(n), 0.0);
        std::vector<double> lambda_w(static_cast<size_t>(m_w), 0.0);

        if (n_f > 0) {
            // Build reduced Augmented Hessian Q_FF + gamma * A_{W, F}^T * A_{W, F} + eps * I
            std::vector<la::Triplet> Q_FF_triplets;
            for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                const int orig_c = free_cols[static_cast<size_t>(f_idx)];
                const auto r_rows = Q_eff.col_rows(orig_c);
                const auto r_vals = Q_eff.col_vals(orig_c);
                for (size_t k = 0; k < r_rows.size(); ++k) {
                    const int orig_r = r_rows[k];
                    const int f_r = col_to_free[static_cast<size_t>(orig_r)];
                    if (f_r >= 0) {
                        Q_FF_triplets.push_back({f_r, f_idx, r_vals[k]});
                    }
                }
            }

            if (m_w > 0) {
                const double gamma = 1.0;
                for (int i_w = 0; i_w < m_w; ++i_w) {
                    const int row_orig = active_rows[static_cast<size_t>(i_w)];
                    std::vector<std::pair<int, double>> row_entries;
                    for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                        const int col_orig = free_cols[static_cast<size_t>(f_idx)];
                        const double coeff = model.A.coeff(row_orig, col_orig);
                        if (std::abs(coeff) > tol::kZeroDrop) {
                            row_entries.emplace_back(f_idx, coeff);
                        }
                    }
                    for (const auto& [f1, c1] : row_entries) {
                        for (const auto& [f2, c2] : row_entries) {
                            Q_FF_triplets.push_back({f1, f2, gamma * c1 * c2});
                        }
                    }
                }
            }

            const double eps_reg = 1e-9;
            for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                Q_FF_triplets.push_back({f_idx, f_idx, eps_reg});
            }
            la::SparseMatrixCSC Q_FF = la::SparseMatrixCSC::from_triplets(n_f, n_f, Q_FF_triplets, true);

            la::SparseLDL ldl_Q;
            ldl_Q.factorize(Q_FF, true);

            // Solve Q_FF * v = g_F
            std::vector<double> v(static_cast<size_t>(n_f), 0.0);
            for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                v[static_cast<size_t>(f_idx)] = g[static_cast<size_t>(free_cols[static_cast<size_t>(f_idx)])];
            }
            ldl_Q.solve(v);

            if (m_w == 0) {
                for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                    p[static_cast<size_t>(free_cols[static_cast<size_t>(f_idx)])] = -v[static_cast<size_t>(f_idx)];
                }
            } else {
                // Form Schur complement S = A_{W, F} * Q_{FF}^{-1} * A_{W, F}^T
                // Solve Q_FF * w_i = (A_i)_F
                std::vector<std::vector<double>> W_cols(static_cast<size_t>(m_w));
                for (int i_w = 0; i_w < m_w; ++i_w) {
                    const int row_orig = active_rows[static_cast<size_t>(i_w)];
                    std::vector<double> w_i(static_cast<size_t>(n_f), 0.0);
                    for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                        const int col_orig = free_cols[static_cast<size_t>(f_idx)];
                        w_i[static_cast<size_t>(f_idx)] = model.A.coeff(row_orig, col_orig);
                    }
                    ldl_Q.solve(w_i);
                    W_cols[static_cast<size_t>(i_w)] = std::move(w_i);
                }

                // S is m_w x m_w
                std::vector<la::Triplet> S_triplets;
                for (int i_w = 0; i_w < m_w; ++i_w) {
                    const int row_i = active_rows[static_cast<size_t>(i_w)];
                    for (int j_w = 0; j_w < m_w; ++j_w) {
                        double s_ij = 0.0;
                        for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                            const int col_orig = free_cols[static_cast<size_t>(f_idx)];
                            s_ij += model.A.coeff(row_i, col_orig) * W_cols[static_cast<size_t>(j_w)][static_cast<size_t>(f_idx)];
                        }
                        if (i_w == j_w) s_ij += 1e-12; // Regularization for linear independence
                        if (std::abs(s_ij) > tol::kZeroDrop) {
                            S_triplets.push_back({i_w, j_w, s_ij});
                        }
                    }
                }
                la::SparseMatrixCSC S_mat = la::SparseMatrixCSC::from_triplets(m_w, m_w, S_triplets, true);

                // Right-hand side: r_S = -A_{W, F} * v
                std::vector<double> r_S(static_cast<size_t>(m_w), 0.0);
                for (int i_w = 0; i_w < m_w; ++i_w) {
                    const int row_orig = active_rows[static_cast<size_t>(i_w)];
                    double Av = 0.0;
                    for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                        const int col_orig = free_cols[static_cast<size_t>(f_idx)];
                        Av += model.A.coeff(row_orig, col_orig) * v[static_cast<size_t>(f_idx)];
                    }
                    r_S[static_cast<size_t>(i_w)] = -Av;
                }

                la::SparseLDL ldl_S;
                ldl_S.factorize(S_mat, false);
                lambda_w = r_S;
                ldl_S.solve(lambda_w);

                // Step p_F = -v - sum_i lambda_w[i] * w_i
                for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                    double step = -v[static_cast<size_t>(f_idx)];
                    for (int i_w = 0; i_w < m_w; ++i_w) {
                        step -= lambda_w[static_cast<size_t>(i_w)] * W_cols[static_cast<size_t>(i_w)][static_cast<size_t>(f_idx)];
                    }
                    p[static_cast<size_t>(free_cols[static_cast<size_t>(f_idx)])] = step;
                }
            }
        }

        // Check step norm ||p||_inf
        double norm_p = 0.0;
        for (int j = 0; j < n; ++j) {
            norm_p = std::max(norm_p, std::abs(p[static_cast<size_t>(j)]));
        }

        if (norm_p > 1e-10) {
            // Non-zero step: find maximum step size alpha in [0, 1]
            double alpha = 1.0;
            int blocking_col = -1;
            BoundState blocking_col_state = BoundState::kFree;
            int blocking_row = -1;
            BoundState blocking_row_state = BoundState::kFree;

            // Check column bounds
            for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                const int j = free_cols[static_cast<size_t>(f_idx)];
                const double pj = p[static_cast<size_t>(j)];
                const double xj = x[static_cast<size_t>(j)];
                const double lj = model.col_lower[static_cast<size_t>(j)];
                const double uj = model.col_upper[static_cast<size_t>(j)];

                if (pj > 1e-12 && uj < 1e19) {
                    const double a = (uj - xj) / pj;
                    if (a < alpha) {
                        alpha = std::max(0.0, a);
                        blocking_col = j;
                        blocking_col_state = BoundState::kAtUpper;
                        blocking_row = -1;
                    }
                } else if (pj < -1e-12 && lj > -1e19) {
                    const double a = (lj - xj) / pj;
                    if (a < alpha) {
                        alpha = std::max(0.0, a);
                        blocking_col = j;
                        blocking_col_state = BoundState::kAtLower;
                        blocking_row = -1;
                    }
                }
            }

            // Check row bounds
            std::vector<double> Ap(static_cast<size_t>(m), 0.0);
            if (m > 0) {
                model.A.multiply(p, Ap);
                model.A.multiply(x, Ax);
                for (int i = 0; i < m; ++i) {
                    if (row_state[static_cast<size_t>(i)] != BoundState::kFree) continue;

                    const double api = Ap[static_cast<size_t>(i)];
                    const double axi = Ax[static_cast<size_t>(i)];
                    const double li = model.row_lower[static_cast<size_t>(i)];
                    const double ui = model.row_upper[static_cast<size_t>(i)];

                    if (api > 1e-12 && ui < 1e19) {
                        const double a = (ui - axi) / api;
                        if (a < alpha) {
                            alpha = std::max(0.0, a);
                            blocking_row = i;
                            blocking_row_state = BoundState::kAtUpper;
                            blocking_col = -1;
                        }
                    } else if (api < -1e-12 && li > -1e19) {
                        const double a = (li - axi) / api;
                        if (a < alpha) {
                            alpha = std::max(0.0, a);
                            blocking_row = i;
                            blocking_row_state = BoundState::kAtLower;
                            blocking_col = -1;
                        }
                    }
                }
            }

            if (blocking_col == -1 && blocking_row == -1) {
                // Check if p is an unbounded ray of descent
                bool completely_unblocked = true;
                for (int f_idx = 0; f_idx < n_f; ++f_idx) {
                    const int j = free_cols[static_cast<size_t>(f_idx)];
                    const double pj = p[static_cast<size_t>(j)];
                    if (pj > 1e-12 && model.col_upper[static_cast<size_t>(j)] < 1e19) completely_unblocked = false;
                    if (pj < -1e-12 && model.col_lower[static_cast<size_t>(j)] > -1e19) completely_unblocked = false;
                }
                if (m > 0) {
                    for (int i = 0; i < m; ++i) {
                        const double api = Ap[static_cast<size_t>(i)];
                        if (api > 1e-12 && model.row_upper[static_cast<size_t>(i)] < 1e19) completely_unblocked = false;
                        if (api < -1e-12 && model.row_lower[static_cast<size_t>(i)] > -1e19) completely_unblocked = false;
                    }
                }

                if (completely_unblocked) {
                    double g_dot_p = 0.0;
                    for (int j = 0; j < n; ++j) g_dot_p += g[static_cast<size_t>(j)] * p[static_cast<size_t>(j)];
                    std::vector<double> Qp(static_cast<size_t>(n), 0.0);
                    Q_eff.multiply(p, Qp);
                    double p_Q_p = 0.0;
                    for (int j = 0; j < n; ++j) p_Q_p += p[static_cast<size_t>(j)] * Qp[static_cast<size_t>(j)];

                    if (g_dot_p < -1e-6 && p_Q_p <= 1e-6 * norm_p * norm_p) {
                        sol.status = SolveStatus::kUnbounded;
                        sol.status_message = "QP is unbounded (detected ray of infinite descent with zero curvature)";
                        sol.certificate_type = "ray";
                        sol.certificate_vector = p;
                        sol.solve_time_seconds = elapsed_sec();
                        return sol;
                    }
                }
            }

            // Update iterate x += alpha * p
            for (int j = 0; j < n; ++j) {
                x[static_cast<size_t>(j)] += alpha * p[static_cast<size_t>(j)];
                x[static_cast<size_t>(j)] = std::clamp(x[static_cast<size_t>(j)], model.col_lower[static_cast<size_t>(j)], model.col_upper[static_cast<size_t>(j)]);
            }

            if (alpha < 1.0) {
                // Add blocking constraint to working set
                if (blocking_col >= 0) {
                    col_state[static_cast<size_t>(blocking_col)] = blocking_col_state;
                } else if (blocking_row >= 0) {
                    row_state[static_cast<size_t>(blocking_row)] = blocking_row_state;
                    active_rows.push_back(blocking_row);
                }
            }
        } else {
            // p == 0: subspace minimizer. Check Lagrange multiplier signs
            // Compute row duals y
            y_dual.assign(static_cast<size_t>(m), 0.0);
            for (int i_w = 0; i_w < m_w; ++i_w) {
                const int row_orig = active_rows[static_cast<size_t>(i_w)];
                y_dual[static_cast<size_t>(row_orig)] = -lambda_w[static_cast<size_t>(i_w)];
            }

            // Compute reduced costs d = g - A^T y
            std::vector<double> Aty(static_cast<size_t>(n), 0.0);
            if (m > 0) {
                model.A.multiply_transpose(y_dual, Aty);
            }

            z_dual.assign(static_cast<size_t>(n), 0.0);
            for (int j = 0; j < n; ++j) {
                z_dual[static_cast<size_t>(j)] = g[static_cast<size_t>(j)] - Aty[static_cast<size_t>(j)];
            }

            // Check multiplier sign violations:
            // For minimization:
            // col at lower bound: z_j should be >= 0 (violation if z_j < -tol)
            // col at upper bound: z_j should be <= 0 (violation if z_j > tol)
            // row at lower bound (Ax = rl): y_i should be >= 0 (violation if y_i < -tol)
            // row at upper bound (Ax = ru): y_i should be <= 0 (violation if y_i > tol)
            double max_viol = 0.0;
            int drop_col = -1;
            int drop_row_idx = -1;

            for (int j = 0; j < n; ++j) {
                const auto st = col_state[static_cast<size_t>(j)];
                const double zj = z_dual[static_cast<size_t>(j)];
                if (st == BoundState::kAtLower && zj < -qp_tol) {
                    const double viol = -zj;
                    if (viol > max_viol) {
                        max_viol = viol;
                        drop_col = j;
                        drop_row_idx = -1;
                    }
                } else if (st == BoundState::kAtUpper && zj > qp_tol) {
                    const double viol = zj;
                    if (viol > max_viol) {
                        max_viol = viol;
                        drop_col = j;
                        drop_row_idx = -1;
                    }
                }
            }

            for (int i_w = 0; i_w < m_w; ++i_w) {
                const int row_orig = active_rows[static_cast<size_t>(i_w)];
                const auto st = row_state[static_cast<size_t>(row_orig)];
                const double yi = y_dual[static_cast<size_t>(row_orig)];

                if (st == BoundState::kAtLower && yi < -qp_tol) {
                    const double viol = -yi;
                    if (viol > max_viol) {
                        max_viol = viol;
                        drop_col = -1;
                        drop_row_idx = i_w;
                    }
                } else if (st == BoundState::kAtUpper && yi > qp_tol) {
                    const double viol = yi;
                    if (viol > max_viol) {
                        max_viol = viol;
                        drop_col = -1;
                        drop_row_idx = i_w;
                    }
                }
            }

            if (max_viol <= qp_tol) {
                // Optimality reached! All KKT stationarity, feasibility, and complementarity conditions hold.
                sol.status = SolveStatus::kOptimal;
                sol.status_message = "Optimal solution found (verified KKT stationary point)";
                break;
            }

            // Drop the constraint with the largest sign violation
            if (drop_col >= 0) {
                col_state[static_cast<size_t>(drop_col)] = BoundState::kFree;
            } else if (drop_row_idx >= 0) {
                const int row_to_drop = active_rows[static_cast<size_t>(drop_row_idx)];
                row_state[static_cast<size_t>(row_to_drop)] = BoundState::kFree;
                active_rows.erase(active_rows.begin() + drop_row_idx);
            }
        }
    }

    if (iter >= max_iters && sol.status != SolveStatus::kOptimal && sol.status != SolveStatus::kTimeLimit) {
        sol.status = SolveStatus::kIterationLimit;
        sol.status_message = "Iteration limit reached in QP active-set solve";
    }

    sol.iterations = iter;
    sol.solve_time_seconds = elapsed_sec();
    sol.col_value = x;

    // Compute row values Ax
    sol.row_value.assign(static_cast<size_t>(m), 0.0);
    if (m > 0) {
        model.A.multiply(x, sol.row_value);
    }

    // Set dual vectors (canonical minimization coordinates for recompute_quality)
    sol.row_dual = y_dual;
    sol.col_dual = z_dual;

    // Basis statuses
    sol.col_basis_status.resize(static_cast<size_t>(n), BasisStatus::kUnknown);
    for (int j = 0; j < n; ++j) {
        switch (col_state[static_cast<size_t>(j)]) {
            case BoundState::kFree: sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kBasic; break;
            case BoundState::kAtLower: sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kAtLower; break;
            case BoundState::kAtUpper: sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kAtUpper; break;
            case BoundState::kFixed: sol.col_basis_status[static_cast<size_t>(j)] = BasisStatus::kFixed; break;
        }
    }
    sol.row_basis_status.resize(static_cast<size_t>(m), BasisStatus::kUnknown);
    for (int i = 0; i < m; ++i) {
        switch (row_state[static_cast<size_t>(i)]) {
            case BoundState::kFree: sol.row_basis_status[static_cast<size_t>(i)] = BasisStatus::kBasic; break;
            case BoundState::kAtLower: sol.row_basis_status[static_cast<size_t>(i)] = BasisStatus::kAtLower; break;
            case BoundState::kAtUpper: sol.row_basis_status[static_cast<size_t>(i)] = BasisStatus::kAtUpper; break;
            case BoundState::kFixed: sol.row_basis_status[static_cast<size_t>(i)] = BasisStatus::kFixed; break;
        }
    }

    // Recompute exact objective: 0.5 * x^T Q_raw x + c^T x + offset
    double quad_obj = 0.0;
    if (Q_raw.nnz() > 0) {
        std::vector<double> Qx(static_cast<size_t>(n), 0.0);
        Q_raw.multiply(x, Qx);
        for (int j = 0; j < n; ++j) {
            quad_obj += 0.5 * x[static_cast<size_t>(j)] * Qx[static_cast<size_t>(j)];
        }
    }
    double lin_obj = 0.0;
    for (int j = 0; j < n; ++j) {
        lin_obj += model.c[static_cast<size_t>(j)] * x[static_cast<size_t>(j)];
    }
    sol.objective_value = quad_obj + lin_obj + model.objective_offset;

    // Recompute solution quality metrics
    sol.recompute_quality(model);

    return sol;
}

} // namespace indus::qp
