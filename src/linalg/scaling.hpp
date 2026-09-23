#pragma once

#include <vector>
#include <span>
#include "indus/types.hpp"
#include "indus/tolerances.hpp"
#include "indus/sparse.hpp"

namespace indus::la {

struct ScalingFactors {
    std::vector<double> row_scale;      // R: dimension m
    std::vector<double> col_scale;      // C: dimension n
    std::vector<double> inv_row_scale;  // R⁻¹
    std::vector<double> inv_col_scale;  // C⁻¹
    bool is_scaled = false;
};

class RuizScaler {
public:
    RuizScaler() = default;

    // Computes two-sided l_inf equilibration factors and scales matrix in place
    static ScalingFactors compute_and_scale(
        SparseMatrixCSC& A,
        std::vector<double>& c,
        std::vector<double>& row_lower,
        std::vector<double>& row_upper,
        std::vector<double>& col_lower,
        std::vector<double>& col_upper,
        int max_iters = tol::kMaxRuizIterations,
        double tolerance = tol::kRuizTolerance
    );

    // Unscales primal solution: x = C * x_scaled
    static void unscale_primal(const ScalingFactors& sf, std::span<double> x);

    // Unscales dual solution: y = R * y_scaled
    static void unscale_dual(const ScalingFactors& sf, std::span<double> y);

    // Unscales reduced costs: d = C⁻¹ * d_scaled
    static void unscale_reduced_costs(const ScalingFactors& sf, std::span<double> d);
};

class PockChambollePreconditioner {
public:
    // Computes diagonal preconditioning vectors D_R and D_C for PDHG
    static void compute(
        const SparseMatrixCSC& A,
        std::vector<double>& d_row,
        std::vector<double>& d_col,
        double alpha = 1.0
    );
};

} // namespace indus::la
