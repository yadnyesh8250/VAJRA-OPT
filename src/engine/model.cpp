#include "indus/model.hpp"
#include "indus/milp.hpp"
#include "indus/qp.hpp"
#include "indus/presolve.hpp"
#include "indus/verifier.hpp"
#include "indus/pdhg.hpp"
#include "indus/gpu.hpp"
#include "src/solvers/simplex/simplex_core.hpp"
#include "src/linalg/scaling.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <iostream>

namespace indus {

bool Model::has_integers() const noexcept {
    for (VarType t : col_type) {
        if (t == VarType::kInteger) return true;
    }
    return false;
}

ProblemClass Model::classify() const noexcept {
    const bool is_qp = has_quadratic_objective();
    const bool is_mip = has_integers();

    if (is_qp && is_mip) return ProblemClass::kMiqp;
    if (is_qp) return ProblemClass::kQp;
    if (is_mip) return ProblemClass::kMilp;
    return ProblemClass::kLp;
}

void Model::validate() const {
    if (num_rows < 0 || num_cols < 0) {
        throw std::invalid_argument("Model dimensions cannot be negative");
    }
    if (num_rows == 0) {
        if (A.m != 0 || (A.n != 0 && A.n != num_cols)) {
            throw std::invalid_argument("Constraint matrix A dimensions do not match model num_rows/num_cols");
        }
    } else {
        if (A.m != num_rows || A.n != num_cols) {
            throw std::invalid_argument("Constraint matrix A dimensions do not match model num_rows/num_cols");
        }
    }
    if (static_cast<int>(c.size()) != num_cols) {
        throw std::invalid_argument("Objective vector c size does not match num_cols");
    }
    if (static_cast<int>(col_lower.size()) != num_cols || static_cast<int>(col_upper.size()) != num_cols) {
        throw std::invalid_argument("Column bounds sizes do not match num_cols");
    }
    if (static_cast<int>(row_lower.size()) != num_rows || static_cast<int>(row_upper.size()) != num_rows) {
        throw std::invalid_argument("Row bounds sizes do not match num_rows");
    }

    // Integrality metadata validation
    if (!col_type.empty() && static_cast<int>(col_type.size()) != num_cols) {
        throw std::invalid_argument("col_type size (" + std::to_string(col_type.size()) +
                                    ") does not match num_cols (" + std::to_string(num_cols) + ")");
    }

    for (int j = 0; j < num_cols; ++j) {
        const double lj = col_lower[static_cast<size_t>(j)];
        const double uj = col_upper[static_cast<size_t>(j)];
        if (std::isnan(lj) || std::isnan(uj)) {
            throw std::invalid_argument("Column " + std::to_string(j) + " contains NaN bound");
        }
        if (lj > uj + tol::kZeroDrop) {
            throw std::invalid_argument("Column " + std::to_string(j) + " lower bound exceeds upper bound");
        }

        // Integer variable validation
        if (static_cast<size_t>(j) < col_type.size() && col_type[static_cast<size_t>(j)] == VarType::kInteger) {
            if (std::isinf(lj) || std::isinf(uj)) {
                throw std::invalid_argument("Integer variable " + std::to_string(j) + " has infinite bound");
            }
            if (lj > uj) {
                throw std::invalid_argument("Integer variable " + std::to_string(j) + " lower bound exceeds upper bound");
            }
            if (std::ceil(lj) > std::floor(uj)) {
                throw std::invalid_argument("Integer variable " + std::to_string(j) + " has empty integer range [" +
                                            std::to_string(lj) + ", " + std::to_string(uj) + "]");
            }
        }
    }
    for (int i = 0; i < num_rows; ++i) {
        const double li = row_lower[static_cast<size_t>(i)];
        const double ui = row_upper[static_cast<size_t>(i)];
        if (std::isnan(li) || std::isnan(ui)) {
            throw std::invalid_argument("Row " + std::to_string(i) + " contains NaN bound");
        }
        if (li > ui + tol::kZeroDrop) {
            throw std::invalid_argument("Row " + std::to_string(i) + " lower bound exceeds upper bound");
        }
    }

    // Quadratic Hessian matrix Q validation
    if (Q.nnz() > 0) {
        if (Q.m != num_cols || Q.n != num_cols) {
            throw std::invalid_argument("Q matrix dimensions (" + std::to_string(Q.m) + "x" +
                                        std::to_string(Q.n) + ") do not match num_cols (" +
                                        std::to_string(num_cols) + ")");
        }
        for (double v : Q.values) {
            if (std::isnan(v)) {
                throw std::invalid_argument("Q matrix contains NaN value");
            }
            if (std::isinf(v)) {
                throw std::invalid_argument("Q matrix contains Inf value");
            }
        }
        for (int j = 0; j < Q.n; ++j) {
            const auto rows = Q.col_rows(j);
            for (int r : rows) {
                if (r < 0 || r >= num_cols) {
                    throw std::invalid_argument("Q matrix contains out-of-bounds row index: " + std::to_string(r));
                }
            }
        }
        // Symmetry validation when both (r, c) and (c, r) are explicitly stored
        for (int c = 0; c < Q.n; ++c) {
            const auto rows = Q.col_rows(c);
            const auto vals = Q.col_vals(c);
            for (size_t k = 0; k < rows.size(); ++k) {
                const int r = rows[k];
                if (r != c) {
                    const double v_rc = vals[k];
                    const auto opp_rows = Q.col_rows(r);
                    const auto opp_vals = Q.col_vals(r);
                    for (size_t l = 0; l < opp_rows.size(); ++l) {
                        if (opp_rows[l] == c) {
                            const double v_cr = opp_vals[l];
                            const double scale = 1.0 + std::max(std::abs(v_rc), std::abs(v_cr));
                            if (std::abs(v_rc - v_cr) > 1e-5 * scale) {
                                throw std::invalid_argument("Q matrix is asymmetric: Q(" +
                                                            std::to_string(r) + "," + std::to_string(c) + ")=" +
                                                            std::to_string(v_rc) + " != Q(" +
                                                            std::to_string(c) + "," + std::to_string(r) + ")=" +
                                                            std::to_string(v_cr));
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

la::SparseMatrixCSC Model::get_symmetric_Q() const {
    if (Q.nnz() == 0) {
        return la::SparseMatrixCSC(num_cols, num_cols);
    }
    std::vector<la::Triplet> triplets;
    triplets.reserve(static_cast<size_t>(Q.nnz() * 2));

    for (int c = 0; c < Q.n; ++c) {
        const auto rows = Q.col_rows(c);
        const auto vals = Q.col_vals(c);
        for (size_t k = 0; k < rows.size(); ++k) {
            const int r = rows[k];
            const double v = vals[k];
            if (std::abs(v) <= tol::kZeroDrop) continue;

            if (r == c) {
                triplets.push_back({r, c, v});
            } else if (r > c) {
                double opp = 0.0;
                bool has_opp = false;
                const auto opp_rows = Q.col_rows(r);
                const auto opp_vals = Q.col_vals(r);
                for (size_t l = 0; l < opp_rows.size(); ++l) {
                    if (opp_rows[l] == c) {
                        opp = opp_vals[l];
                        has_opp = true;
                        break;
                    }
                }
                const double sym_v = has_opp ? (0.5 * (v + opp)) : v;
                triplets.push_back({r, c, sym_v});
                triplets.push_back({c, r, sym_v});
            } else { // r < c
                bool has_opp = false;
                const auto opp_rows = Q.col_rows(r);
                for (int opp_r : opp_rows) {
                    if (opp_r == c) {
                        has_opp = true;
                        break;
                    }
                }
                if (!has_opp) {
                    triplets.push_back({r, c, v});
                    triplets.push_back({c, r, v});
                }
            }
        }
    }
    return la::SparseMatrixCSC::from_triplets(num_cols, num_cols, triplets, true);
}

void Solution::recompute_quality(const Model& original_model) {
    quality = {};
    if (!claims_a_point(status)) {
        return;
    }

    const int m = original_model.num_rows;
    const int n = original_model.num_cols;

    if (static_cast<int>(col_value.size()) != n) {
        return;
    }

    double max_rel_primal_violation = 0.0;

    // 1. Primal column bound violations
    for (int j = 0; j < n; ++j) {
        const double xj = col_value[static_cast<size_t>(j)];
        const double lj = original_model.col_lower[static_cast<size_t>(j)];
        const double uj = original_model.col_upper[static_cast<size_t>(j)];

        if (xj < lj) {
            const double viol = lj - xj;
            quality.max_primal_violation = std::max(quality.max_primal_violation, viol);
            const double denom = 1.0 + std::abs(lj);
            max_rel_primal_violation = std::max(max_rel_primal_violation, viol / denom);
        }
        if (xj > uj) {
            const double viol = xj - uj;
            quality.max_primal_violation = std::max(quality.max_primal_violation, viol);
            const double denom = 1.0 + std::abs(uj);
            max_rel_primal_violation = std::max(max_rel_primal_violation, viol / denom);
        }
    }

    // 2. Row activity Ax and row bound violations
    std::vector<double> Ax(static_cast<size_t>(m), 0.0);
    original_model.A.multiply(col_value, Ax);
    row_value = Ax;

    for (int i = 0; i < m; ++i) {
        const double ax = Ax[static_cast<size_t>(i)];
        const double li = original_model.row_lower[static_cast<size_t>(i)];
        const double ui = original_model.row_upper[static_cast<size_t>(i)];

        if (ax < li) {
            const double viol = li - ax;
            quality.max_primal_violation = std::max(quality.max_primal_violation, viol);
            const double denom = 1.0 + (li > -1e19 ? std::abs(li) : 1.0);
            max_rel_primal_violation = std::max(max_rel_primal_violation, viol / denom);
        }
        if (ax > ui) {
            const double viol = ax - ui;
            quality.max_primal_violation = std::max(quality.max_primal_violation, viol);
            const double denom = 1.0 + (ui < 1e19 ? std::abs(ui) : 1.0);
            max_rel_primal_violation = std::max(max_rel_primal_violation, viol / denom);
        }
    }

    quality.is_primal_feasible = (quality.max_primal_violation <= tol::kPrimalFeasibility ||
                                  max_rel_primal_violation <= tol::kPrimalFeasibility);

    // 3. Integrality check for MILP
    if (original_model.has_integers()) {
        quality.max_integrality_violation = 0.0;
        quality.is_integer_feasible = true;
        for (int j = 0; j < n; ++j) {
            if (original_model.col_type.size() > static_cast<size_t>(j) &&
                original_model.col_type[static_cast<size_t>(j)] == VarType::kInteger) {
                const double xj = col_value[static_cast<size_t>(j)];
                const double int_viol = std::abs(xj - std::round(xj));
                quality.max_integrality_violation = std::max(quality.max_integrality_violation, int_viol);
                if (int_viol > tol::kPrimalFeasibility) {
                    quality.is_integer_feasible = false;
                }
            }
        }
        quality.is_dual_feasible = true; // Continuous KKT dual conditions do not apply to MILP
        quality.max_dual_violation = 0.0;
        quality.max_complementarity_violation = 0.0;
    } else {
        quality.is_integer_feasible = true;
        quality.max_integrality_violation = 0.0;
        // 4. Dual feasibility & complementary slackness (if dual vector present)
        if (static_cast<int>(row_dual.size()) == m) {
            std::vector<double> Aty(static_cast<size_t>(n), 0.0);
            original_model.A.multiply_transpose(row_dual, Aty);

            const double sense_factor = (original_model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;

            std::vector<double> grad = original_model.c;
            if (original_model.has_quadratic_objective()) {
                la::SparseMatrixCSC Q_sym = original_model.get_symmetric_Q();
                std::vector<double> Qx(static_cast<size_t>(n), 0.0);
                Q_sym.multiply(col_value, Qx);
                for (int j = 0; j < n; ++j) {
                    grad[static_cast<size_t>(j)] += Qx[static_cast<size_t>(j)];
                }
            }

            col_dual.resize(static_cast<size_t>(n));
            for (int j = 0; j < n; ++j) {
                const double c_eff = sense_factor * grad[static_cast<size_t>(j)];
                const double dj = c_eff - Aty[static_cast<size_t>(j)];
                col_dual[static_cast<size_t>(j)] = (original_model.sense == ObjSense::kMaximize) ? -dj : dj;

                const double xj = col_value[static_cast<size_t>(j)];
                const double lj = original_model.col_lower[static_cast<size_t>(j)];
                const double uj = original_model.col_upper[static_cast<size_t>(j)];

                // Reduced cost violations in canonical minimization
                if (xj < uj - tol::kPrimalFeasibility && dj < -tol::kDualFeasibility) {
                    quality.max_dual_violation = std::max(quality.max_dual_violation, -dj);
                }
                if (xj > lj + tol::kPrimalFeasibility && dj > tol::kDualFeasibility) {
                    quality.max_dual_violation = std::max(quality.max_dual_violation, dj);
                }

                // Complementarity
                if (dj > tol::kDualFeasibility) {
                    const double dist = std::abs(xj - lj);
                    quality.max_complementarity_violation = std::max(
                        quality.max_complementarity_violation, dj * dist);
                } else if (dj < -tol::kDualFeasibility) {
                    const double dist = std::abs(xj - uj);
                    quality.max_complementarity_violation = std::max(
                        quality.max_complementarity_violation, (-dj) * dist);
                }

                // Stationarity residual for QP
                if (original_model.has_quadratic_objective()) {
                    double z_proj = 0.0;
                    if (xj <= lj + tol::kPrimalFeasibility && xj < uj - tol::kPrimalFeasibility) {
                        z_proj = std::max(0.0, dj);
                    } else if (xj >= uj - tol::kPrimalFeasibility && xj > lj + tol::kPrimalFeasibility) {
                        z_proj = std::min(0.0, dj);
                    } else if (std::abs(lj - uj) <= tol::kZeroDrop) {
                        z_proj = dj;
                    }
                    const double stat_res = std::abs(dj - z_proj);
                    quality.max_stationarity_residual = std::max(quality.max_stationarity_residual, stat_res);
                }
            }

            // Dual multiplier row condition violations in canonical minimization
            for (int i = 0; i < m; ++i) {
                const double yi = row_dual[static_cast<size_t>(i)];
                const double li = original_model.row_lower[static_cast<size_t>(i)];
                const double ui = original_model.row_upper[static_cast<size_t>(i)];

                if (li <= -1e20 && ui < 1e20) {
                    // <= constraint: y_i should be <= 0 in minimization
                    if (yi > tol::kDualFeasibility) {
                        quality.max_dual_violation = std::max(quality.max_dual_violation, yi);
                    }
                } else if (li > -1e20 && ui >= 1e20) {
                    // >= constraint: y_i should be >= 0 in minimization
                    if (yi < -tol::kDualFeasibility) {
                        quality.max_dual_violation = std::max(quality.max_dual_violation, -yi);
                    }
                }
            }

            if (original_model.has_quadratic_objective()) {
                quality.is_stationary = (quality.max_stationarity_residual <= tol::kDualFeasibility);
                quality.is_dual_feasible = quality.is_stationary && (quality.max_dual_violation <= tol::kDualFeasibility);
            } else {
                quality.is_dual_feasible = (quality.max_dual_violation <= tol::kDualFeasibility);
            }
        }
    }
}

Solution solve(const Model& model, const Options& options) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    try {
        model.validate();
    } catch (const std::exception& e) {
        Solution sol;
        sol.status = SolveStatus::kModelError;
        sol.status_message = std::string("Model validation error: ") + e.what();
        sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        return sol;
    }

    if (model.classify() == ProblemClass::kMiqp) {
        Solution sol;
        sol.status = SolveStatus::kUnsupported;
        sol.status_message = "MIQP (Mixed-Integer Quadratic Programming) is unsupported in Phase 8";
        sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        return sol;
    }

    if (model.classify() == ProblemClass::kQp || options.algorithm == "qp") {
        return qp::solve_qp(model, options);
    }
    if (model.has_integers() || options.algorithm == "milp") {
        return milp::solve_milp(model, options);
    }

    if (options.time_limit <= 0.0) {
        Solution sol;
        sol.status = SolveStatus::kTimeLimit;
        sol.status_message = "Time limit exceeded before solve commenced";
        sol.solve_time_seconds = 0.0;
        return sol;
    }

    // Edge case 1: Zero variables (empty or constraint-only model)
    if (model.num_cols == 0) {
        Solution sol;
        sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        sol.objective_value = model.objective_offset;
        for (int i = 0; i < model.num_rows; ++i) {
            if (0.0 < model.row_lower[static_cast<size_t>(i)] - tol::kPrimalFeasibility ||
                0.0 > model.row_upper[static_cast<size_t>(i)] + tol::kPrimalFeasibility) {
                sol.status = SolveStatus::kInfeasible;
                sol.status_message = "Empty model constraint violated at row " + std::to_string(i);
                return sol;
            }
        }
        sol.status = SolveStatus::kOptimal;
        sol.status_message = "Model with zero variables solved trivially";
        return sol;
    }

    // Edge case 2: Zero constraints (box-bounded variables)
    if (model.num_rows == 0) {
        Solution sol;
        sol.col_value.resize(static_cast<size_t>(model.num_cols), 0.0);
        sol.col_dual.resize(static_cast<size_t>(model.num_cols), 0.0);
        double obj = model.objective_offset;
        for (int j = 0; j < model.num_cols; ++j) {
            const double raw_cj = model.c[static_cast<size_t>(j)];
            const double cj = (model.sense == ObjSense::kMaximize) ? -raw_cj : raw_cj;
            const double lj = model.col_lower[static_cast<size_t>(j)];
            const double uj = model.col_upper[static_cast<size_t>(j)];
            double xj = 0.0;
            if (cj > tol::kDualFeasibility) {
                if (lj <= -1e20) {
                    sol.status = SolveStatus::kUnbounded;
                    sol.status_message = "Unbounded variable " + (model.col_names.empty() ? std::to_string(j) : model.col_names[static_cast<size_t>(j)]);
                    sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
                    return sol;
                }
                xj = lj;
            } else if (cj < -tol::kDualFeasibility) {
                if (uj >= 1e20) {
                    sol.status = SolveStatus::kUnbounded;
                    sol.status_message = "Unbounded variable " + (model.col_names.empty() ? std::to_string(j) : model.col_names[static_cast<size_t>(j)]);
                    sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
                    return sol;
                }
                xj = uj;
            } else {
                xj = std::clamp(0.0, lj, uj);
            }
            sol.col_value[static_cast<size_t>(j)] = xj;
            sol.col_dual[static_cast<size_t>(j)] = cj;
            obj += raw_cj * xj;
        }
        sol.status = SolveStatus::kOptimal;
        sol.status_message = "Unconstrained model solved trivially";
        sol.objective_value = obj;
        sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        return sol;
    }

    // Phase 4 Reversible Presolve Engine
    if (options.enable_presolve) {
        auto presolve_res = presolve::PresolveEngine::apply(model);
        if (presolve_res.is_infeasible) {
            Solution sol;
            sol.status = SolveStatus::kInfeasible;
            sol.status_message = "Presolve detected infeasibility: " + presolve_res.status_message;
            sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
            return sol;
        }
        if (presolve_res.is_unbounded) {
            Solution sol;
            sol.status = SolveStatus::kUnbounded;
            sol.status_message = "Presolve detected unboundedness: " + presolve_res.status_message;
            sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
            return sol;
        }

        if (presolve_res.total_reductions() > 0) {
            Options no_presolve_opts = options;
            no_presolve_opts.enable_presolve = false;

            Solution red_sol;
            if (presolve_res.presolved_model.num_cols == 0 || presolve_res.presolved_model.num_rows == 0) {
                // Entire problem resolved by presolve!
                red_sol.status = SolveStatus::kOptimal;
                red_sol.status_message = "Solved to optimality by presolve reductions";
                red_sol.objective_value = presolve_res.presolved_model.objective_offset;
            } else {
                red_sol = solve(presolve_res.presolved_model, no_presolve_opts);
            }

            Solution full_sol = presolve::PresolveEngine::postsolve(red_sol, presolve_res.stack, model);
            full_sol.solve_time_seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
            full_sol.presolve_num_rows = presolve_res.presolved_model.num_rows;
            full_sol.presolve_num_cols = presolve_res.presolved_model.num_cols;
            full_sol.presolve_total_reductions = presolve_res.total_reductions();
            full_sol.presolve_doubleton_reductions = presolve_res.num_doubleton_rows;

            // Run strict verifier on original model to ensure reconstructed solution passes
            if (full_sol.status == SolveStatus::kOptimal) {
                auto vres = verifier::verify_solution(model, full_sol, 1e-4);
                if (!vres.passed) {
                    full_sol.status = SolveStatus::kNumericalError;
                    full_sol.status_message = "Postsolve verification failed: " + vres.summary;
                }
            }

            return full_sol;
        }
    }

    Solution sol;
    if (options.algorithm == "pdhg_cuda" || (options.algorithm == "pdhg" && options.use_gpu) || (options.use_gpu)) {
        sol = gpu::solve_pdhg_gpu(model, options);
    } else if (options.algorithm == "pdhg" || options.algorithm == "pdhg_cpu") {
        sol = pdhg::solve_pdhg_cpu(model, options);
    } else {
        const int n = model.num_cols;
        const int m = model.num_rows;

        la::ScalingFactors scaling_factors;
        la::SparseMatrixCSC scaled_A = model.A;
        std::vector<double> scaled_c = model.c;
        std::vector<double> scaled_row_lower = model.row_lower;
        std::vector<double> scaled_row_upper = model.row_upper;
        std::vector<double> scaled_col_lower = model.col_lower;
        std::vector<double> scaled_col_upper = model.col_upper;

        if (options.enable_scaling) {
            scaling_factors = la::RuizScaler::compute_and_scale(
                scaled_A, scaled_c,
                scaled_row_lower, scaled_row_upper,
                scaled_col_lower, scaled_col_upper
            );
        }

        // Map into simplex representation [A  I] [x  s]^T = 0
        simplex::SimplexModel smodel;
        smodel.num_rows = m;
        smodel.num_cols = n;
        smodel.num_vars = m + n;
        smodel.sense = model.sense;
        smodel.objective_offset = model.objective_offset;
        smodel.A = std::move(scaled_A);

        smodel.lower.resize(static_cast<size_t>(n + m));
        smodel.upper.resize(static_cast<size_t>(n + m));
        smodel.cost.resize(static_cast<size_t>(n + m));

        for (int j = 0; j < n; ++j) {
            smodel.lower[static_cast<size_t>(j)] = scaled_col_lower[static_cast<size_t>(j)];
            smodel.upper[static_cast<size_t>(j)] = scaled_col_upper[static_cast<size_t>(j)];
            smodel.cost[static_cast<size_t>(j)] = (model.sense == ObjSense::kMaximize) ? -scaled_c[static_cast<size_t>(j)]
                                                                                       : scaled_c[static_cast<size_t>(j)];
        }

        for (int i = 0; i < m; ++i) {
            smodel.lower[static_cast<size_t>(n + i)] = -scaled_row_upper[static_cast<size_t>(i)];
            smodel.upper[static_cast<size_t>(n + i)] = -scaled_row_lower[static_cast<size_t>(i)];
            smodel.cost[static_cast<size_t>(n + i)] = 0.0;
        }

        simplex::SimplexCore core(std::move(smodel));
        simplex::SimplexResult sres;

        if (options.algorithm == "primal_simplex") {
            sres = core.solve_primal(options.iteration_limit);
        } else if (options.algorithm == "dual_simplex") {
            sres = core.solve_dual(options.iteration_limit);
        } else {
            sres = core.solve(options.iteration_limit);
        }

        if (scaling_factors.is_scaled && claims_a_point(sres.status)) {
            la::RuizScaler::unscale_primal(scaling_factors, sres.col_value);
            la::RuizScaler::unscale_dual(scaling_factors, sres.row_dual);
            la::RuizScaler::unscale_reduced_costs(scaling_factors, sres.col_dual);

            sres.row_value.resize(static_cast<size_t>(m));
            model.A.multiply(sres.col_value, sres.row_value);

            double unscaled_obj = model.objective_offset;
            for (int j = 0; j < n; ++j) {
                unscaled_obj += model.c[static_cast<size_t>(j)] * sres.col_value[static_cast<size_t>(j)];
            }
            sres.objective_value = unscaled_obj;
        }

        sol.status = sres.status;
        sol.status_message = sres.status_message;
        sol.objective_value = sres.objective_value;
        sol.best_dual_bound = sres.objective_value;
        sol.col_value = std::move(sres.col_value);
        sol.row_value = std::move(sres.row_value);
        sol.row_dual = std::move(sres.row_dual);
        sol.col_dual = std::move(sres.col_dual);
        sol.col_basis_status = std::move(sres.col_status);
        sol.row_basis_status = std::move(sres.row_status);
        sol.certificate_type = std::move(sres.certificate_type);
        sol.certificate_vector = std::move(sres.certificate_vector);
        sol.iterations = sres.iterations;
        sol.algorithm_used = options.algorithm.empty() ? "simplex_auto" : options.algorithm;
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    sol.solve_time_seconds = std::chrono::duration<double>(end_time - start_time).count();

    sol.presolve_num_rows = model.num_rows;
    sol.presolve_num_cols = model.num_cols;
    sol.presolve_total_reductions = 0;
    sol.presolve_doubleton_reductions = 0;

    sol.recompute_quality(model);

    // STATUS GUARD: Never trust solver internals blindly!
    // 1. Check for NaN / Inf in objective and solution vectors
    if (std::isnan(sol.objective_value) || std::isinf(sol.objective_value)) {
        sol.status = SolveStatus::kNumericalError;
        sol.status_message = "Status downgraded to NUMERICAL_ERROR: objective is NaN or Inf";
    }
    for (double xj : sol.col_value) {
        if (std::isnan(xj) || std::isinf(xj)) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "Status downgraded to NUMERICAL_ERROR: primal variable is NaN or Inf";
            break;
        }
    }
    for (double yi : sol.row_dual) {
        if (std::isnan(yi) || std::isinf(yi)) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "Status downgraded to NUMERICAL_ERROR: dual multiplier is NaN or Inf";
            break;
        }
    }

    // 2. Audit optimality & feasibility
    if (sol.status == SolveStatus::kOptimal) {
        if (!sol.quality.is_primal_feasible) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "Status downgraded to NUMERICAL_ERROR: primal feasibility violation (" +
                                 std::to_string(sol.quality.max_primal_violation) + ") exceeds tolerance";
        } else if (!sol.quality.is_dual_feasible) {
            sol.status = SolveStatus::kFeasible;
            sol.status_message = "Status downgraded to FEASIBLE: dual feasibility / complementarity violation (" +
                                 std::to_string(sol.quality.max_dual_violation) + ") exceeds tolerance";
        }
    } else if (sol.status == SolveStatus::kFeasible) {
        if (!sol.quality.is_primal_feasible) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "Status downgraded to NUMERICAL_ERROR: primal feasibility violation (" +
                                 std::to_string(sol.quality.max_primal_violation) + ") exceeds tolerance";
        }
    }

    return sol;
}

} // namespace indus
