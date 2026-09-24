#include "indus/model.hpp"
#include "indus/presolve.hpp"
#include "indus/verifier.hpp"
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
    if (A.m != num_rows || A.n != num_cols) {
        throw std::invalid_argument("Constraint matrix A dimensions do not match model num_rows/num_cols");
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

    for (int j = 0; j < num_cols; ++j) {
        if (col_lower[static_cast<size_t>(j)] > col_upper[static_cast<size_t>(j)] + tol::kZeroDrop) {
            throw std::invalid_argument("Column " + std::to_string(j) + " lower bound exceeds upper bound");
        }
    }
    for (int i = 0; i < num_rows; ++i) {
        if (row_lower[static_cast<size_t>(i)] > row_upper[static_cast<size_t>(i)] + tol::kZeroDrop) {
            throw std::invalid_argument("Row " + std::to_string(i) + " lower bound exceeds upper bound");
        }
    }
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

    // 3. Dual feasibility & complementary slackness (if dual vector present)
    if (static_cast<int>(row_dual.size()) == m) {
        std::vector<double> Aty(static_cast<size_t>(n), 0.0);
        original_model.A.multiply_transpose(row_dual, Aty);

        const double sense_factor = (original_model.sense == ObjSense::kMaximize) ? -1.0 : 1.0;

        col_dual.resize(static_cast<size_t>(n));
        for (int j = 0; j < n; ++j) {
            const double c_eff = sense_factor * original_model.c[static_cast<size_t>(j)];
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

        quality.is_dual_feasible = (quality.max_dual_violation <= tol::kDualFeasibility);
    }
}

Solution solve(const Model& model, const Options& options) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    model.validate();

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

    Solution sol;
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
