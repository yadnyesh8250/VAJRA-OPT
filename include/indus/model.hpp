#pragma once

#include <string>
#include <vector>
#include <memory>
#include "indus/types.hpp"
#include "indus/sparse.hpp"
#include "indus/options.hpp"

namespace indus {

enum class ProblemClass : uint8_t {
    kLp, kMilp, kQp, kMiqp
};

class Model {
public:
    std::string name;
    std::string source_path;
    ObjSense sense = ObjSense::kMinimize;
    double objective_offset = 0.0;

    int num_rows = 0;
    int num_cols = 0;

    // Linear Constraint Matrix A in CSC format (m x n)
    la::SparseMatrixCSC A;

    // Objective vector c
    std::vector<double> c;

    // Two-sided Row bounds: row_lower <= A x <= row_upper
    std::vector<double> row_lower;
    std::vector<double> row_upper;

    // Two-sided Column bounds: col_lower <= x <= col_upper
    std::vector<double> col_lower;
    std::vector<double> col_upper;

    // Integrality & Column types
    std::vector<VarType> col_type;

    // Names (optional)
    std::vector<std::string> row_names;
    std::vector<std::string> col_names;

    // Quadratic Hessian matrix Q (lower triangular, symmetric)
    la::SparseMatrixCSC Q;

    [[nodiscard]] bool has_quadratic_objective() const noexcept { return Q.nnz() > 0; }
    [[nodiscard]] bool has_integers() const noexcept;
    [[nodiscard]] ProblemClass classify() const noexcept;
    void validate() const;
};

struct SolutionQuality {
    double max_primal_violation = 0.0;
    double max_dual_violation = 0.0;
    double max_complementarity_violation = 0.0;
    double duality_gap = 0.0;
    double relative_duality_gap = 0.0;
    bool is_primal_feasible = false;
    bool is_dual_feasible = false;
};

class Solution {
public:
    SolveStatus status = SolveStatus::kNotSolved;
    std::string status_message;

    double objective_value = 0.0;
    double best_dual_bound = 0.0;
    double relative_gap = 0.0;

    // Primal vector x and Row activity Ax
    std::vector<double> col_value;
    std::vector<double> row_value;

    // Dual vector y and Reduced costs d = c - Aᵀ y
    std::vector<double> row_dual;
    std::vector<double> col_dual;

    // Basis statuses
    std::vector<BasisStatus> col_basis_status;
    std::vector<BasisStatus> row_basis_status;

    // Infeasibility / Unboundedness proof certificate
    std::string certificate_type = "none"; // "farkas", "ray", "none"
    std::vector<double> certificate_vector;

    // Runtime statistics
    std::string algorithm_used = "dual_simplex";
    int64_t iterations = 0;
    int64_t nodes = 0;
    double solve_time_seconds = 0.0;

    // Presolve statistics
    int presolve_num_rows = -1;
    int presolve_num_cols = -1;
    int presolve_total_reductions = 0;
    int presolve_doubleton_reductions = 0;

    SolutionQuality quality;
    void recompute_quality(const Model& original_model);
};

// Master Dispatcher Seam
Solution solve(const Model& model, const Options& options = Options());

} // namespace indus
