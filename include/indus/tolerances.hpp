#pragma once

namespace indus {
namespace tol {

// Maximum allowed row/column bound violation: ||Ax - b||_inf <= 1e-7
inline constexpr double kPrimalFeasibility = 1e-7;

// Maximum violation of dual sign conditions / reduced costs: d_j >= -1e-7
inline constexpr double kDualFeasibility = 1e-7;

// Distance from integer: |x_j - round(x_j)| <= 1e-6
inline constexpr double kIntegrality = 1e-6;

// Relative optimality gap for MILP: |z_inc - z_bound| / (|z_inc| + 1e-10) <= 1e-4
inline constexpr double kMipRelativeGap = 1e-4;

// Absolute optimality gap for MILP: |z_inc - z_bound| <= 1e-6
inline constexpr double kMipAbsoluteGap = 1e-6;

// Markowitz threshold pivoting parameter u in [0.01, 0.1]
inline constexpr double kMarkowitzThreshold = 0.01;

// Strict Markowitz threshold when conditioning degrades
inline constexpr double kMarkowitzStrictThreshold = 0.1;

// Elements smaller than 1e-11 are dropped from sparse factorizations
inline constexpr double kZeroDrop = 1e-11;

// Absolute pivot tolerance for LU and LDL factorization
inline constexpr double kPivotTolerance = 1e-13;

// Singularity threshold below which diagonal is declared rank deficient
inline constexpr double kSingularityTolerance = 1e-14;

// Minimum observations before trusting variable branching pseudocosts
inline constexpr int kPseudocostReliability = 8;

// Maximum candidate variables evaluated by strong branching per node
inline constexpr int kStrongBranchingCandidates = 10;

// Maximum dual simplex iterations allowed per strong branching child probe
inline constexpr int kStrongBranchingIterations = 50;

// Maximum number of Ruiz scaling iterations
inline constexpr int kMaxRuizIterations = 10;

// Ruiz scaling convergence tolerance
inline constexpr double kRuizTolerance = 0.05;

// Maximum consecutive eta updates in PFI before mandatory LU refactorization
inline constexpr int kMaxEtaUpdates = 100;

// Residual drift threshold triggering LU refactorization
inline constexpr double kRefactorResidualTolerance = 1e-8;

} // namespace tol
} // namespace indus
