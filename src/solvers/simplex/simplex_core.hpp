#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <span>
#include <memory>
#include "indus/types.hpp"
#include "indus/tolerances.hpp"
#include "indus/sparse.hpp"
#include "src/linalg/lu.hpp"

namespace indus::simplex {

struct SimplexModel {
    int num_rows = 0;                  // m
    int num_cols = 0;                  // n
    int num_vars = 0;                  // n + m (structural + slacks)
    ObjSense sense = ObjSense::kMinimize;
    double objective_offset = 0.0;

    // Structural constraint matrix A (m x n)
    la::SparseMatrixCSC A;

    // Full variable bounds and costs of dimension n + m:
    // Indices 0 ... n - 1: structural variables x
    // Indices n ... n + m - 1: slack variables s (where A*x + I*s = 0, so -u_row <= s <= -l_row)
    std::vector<double> lower;
    std::vector<double> upper;
    std::vector<double> cost;

    // Query column j of [A  I]
    void get_column(int j, std::vector<int>& rows, std::vector<double>& vals) const {
        rows.clear();
        vals.clear();
        if (j < num_cols) {
            const auto r = A.col_rows(j);
            const auto v = A.col_vals(j);
            rows.assign(r.begin(), r.end());
            vals.assign(v.begin(), v.end());
        } else if (j < num_vars) {
            const int slack_row = j - num_cols;
            rows.push_back(slack_row);
            vals.push_back(1.0);
        }
    }
};

struct BasisState {
    int m = 0;
    int num_vars = 0;

    // basic_vars[i] is the variable index (0 ... n+m-1) that is basic in row i
    std::vector<int> basic_vars;

    // basic_index[j] is the row index in basis (0 ... m-1), or -1 if j is non-basic
    std::vector<int> basic_index;

    // Status of each variable in 0 ... n+m-1
    std::vector<BasisStatus> status;

    // Primal values of all variables in 0 ... n+m-1
    std::vector<double> primal;

    // Dual values y of dimension m
    std::vector<double> dual;

    // Reduced costs d of dimension n+m
    std::vector<double> reduced_cost;

    // Devex pricing weights for each basic row i in 0 ... m-1
    std::vector<double> devex_weights;

    void initialize(int rows, int vars) {
        m = rows;
        num_vars = vars;
        basic_vars.assign(static_cast<size_t>(m), -1);
        basic_index.assign(static_cast<size_t>(num_vars), -1);
        status.assign(static_cast<size_t>(num_vars), BasisStatus::kNonbasicFree);
        primal.assign(static_cast<size_t>(num_vars), 0.0);
        dual.assign(static_cast<size_t>(m), 0.0);
        reduced_cost.assign(static_cast<size_t>(num_vars), 0.0);
        devex_weights.assign(static_cast<size_t>(m), 1.0);
    }
};

struct SimplexResult {
    SolveStatus status = SolveStatus::kNotSolved;
    std::string status_message;
    double objective_value = 0.0;

    std::vector<double> col_value;        // Structural primal values (size n)
    std::vector<double> row_value;        // Row activity A*x (size m)
    std::vector<double> row_dual;         // Dual multipliers y (size m)
    std::vector<double> col_dual;         // Reduced costs d (size n)

    std::vector<BasisStatus> col_status;  // size n
    std::vector<BasisStatus> row_status;  // size m

    int64_t iterations = 0;
    std::string certificate_type = "none"; // "farkas", "ray", "none"
    std::vector<double> certificate_vector;
};

class SimplexCore {
public:
    explicit SimplexCore(SimplexModel model);

    // Initial basis construction: slack basis
    void init_slack_basis();

    // Refactorize basis matrix B from scratch using Markowitz LU
    bool refactorize_basis();

    // Recompute primal values: x_B = B⁻¹ ( - N * x_N )
    void compute_primal_values();

    // Recompute dual values: Bᵀ * y = c_B
    void compute_dual_values();

    // Recompute reduced costs: d_j = c_j - a_jᵀ * y
    void compute_reduced_costs();

    // Compute current objective value
    [[nodiscard]] double compute_objective_value() const;

    // FTRAN: solve B * alpha = a_q
    void ftran(int enter_var, std::vector<double>& alpha) const;

    // BTRAN: solve Bᵀ * v = e_p
    void btran(int row_p, std::vector<double>& v, double sign = 1.0) const;

    // Tableau row entry for variable j: a_pj = vᵀ * a_j
    [[nodiscard]] double compute_tableau_entry(int j, const std::vector<double>& v) const;

    // Solvers
    SimplexResult solve_dual(int64_t max_iterations = 100000);
    SimplexResult solve_primal(int64_t max_iterations = 100000);
    SimplexResult solve(int64_t max_iterations = 100000);

    [[nodiscard]] const SimplexModel& model() const noexcept { return model_; }
    [[nodiscard]] const BasisState& basis() const noexcept { return basis_; }

private:
    SimplexModel model_;
    BasisState basis_;
    la::SparseLU lu_;
};

} // namespace indus::simplex
