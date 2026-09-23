#pragma once

#include <vector>
#include <cstdint>
#include <span>
#include <memory>
#include "indus/types.hpp"
#include "indus/tolerances.hpp"
#include "indus/sparse.hpp"

namespace indus::la {

struct EtaVector {
    int pivot_row = 0;
    double pivot_val = 1.0;            // 1.0 / alpha_pq
    std::vector<int> rows;             // indices of off-diagonal entries (i != pivot_row)
    std::vector<double> values;        // -alpha_iq / alpha_pq
};

enum class FactorizationStatus : uint8_t {
    kSuccess = 0,
    kSingularRepaired = 1,
    kFailed = 2
};

class SparseLU {
public:
    SparseLU() = default;
    explicit SparseLU(int dim);

    // Factorize basis matrix B
    FactorizationStatus factorize(const SparseMatrixCSC& B, double markowitz_threshold = tol::kMarkowitzThreshold);

    // Forward solve: B x = b (FTRAN)
    void ftran(std::span<double> x) const;
    void ftran_sparse(const std::vector<int>& rhs_indices, const std::vector<double>& rhs_values, 
                      std::vector<int>& sol_indices, std::vector<double>& sol_values, std::vector<double>& dense_work) const;

    // Backward solve: Bᵀ y = c (BTRAN)
    void btran(std::span<double> y) const;
    void btran_sparse(const std::vector<int>& rhs_indices, const std::vector<double>& rhs_values,
                      std::vector<int>& sol_indices, std::vector<double>& sol_values, std::vector<double>& dense_work) const;

    // Basis update via Product Form of the Inverse (PFI)
    // Entering column replaces column at basic_index (0-indexed column of B).
    // alpha is the FTRAN solution B * alpha = a_entering.
    bool update_basis_pfi(int leaving_row, std::span<const double> alpha);

    // Check if refactorization is recommended
    [[nodiscard]] bool needs_refactorization() const noexcept {
        return static_cast<int>(etas_.size()) >= tol::kMaxEtaUpdates;
    }

    [[nodiscard]] int dimension() const noexcept { return m_; }
    [[nodiscard]] int num_etas() const noexcept { return static_cast<int>(etas_.size()); }
    [[nodiscard]] bool is_repaired() const noexcept { return is_repaired_; }
    [[nodiscard]] const std::vector<int>& repaired_rows() const noexcept { return repaired_rows_; }

    void clear();

private:
    int m_ = 0;
    bool is_repaired_ = false;
    std::vector<int> repaired_rows_;

    // Permutations: row_perm[i] is row in original B that maps to row i in LU
    // col_perm[j] is col in original B that maps to col j in LU
    std::vector<int> p_row_;     // row permutation: P * B * Q = L * U
    std::vector<int> inv_p_row_;
    std::vector<int> q_col_;     // col permutation
    std::vector<int> inv_q_col_;

    // Factored L and U representations
    // L is unit lower triangular
    SparseMatrixCSC L_csc_;
    SparseMatrixCSR L_csr_;
    // U is upper triangular (includes diagonal)
    SparseMatrixCSC U_csc_;
    SparseMatrixCSR U_csr_;
    std::vector<double> u_diag_; // Diagonal of U for fast access

    // PFI eta file
    std::vector<EtaVector> etas_;

    // Internal solve helpers for L and U
    void solve_L_forward(std::span<double> x) const;
    void solve_L_transpose_backward(std::span<double> x) const;
    void solve_U_backward(std::span<double> x) const;
    void solve_U_transpose_forward(std::span<double> x) const;

    // Hyper-sparse DFS reachability helpers
    void dfs_L(int node, std::vector<bool>& visited, std::vector<int>& order) const;
    void dfs_U_transpose(int node, std::vector<bool>& visited, std::vector<int>& order) const;
};

} // namespace indus::la
