#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include "indus/types.hpp"
#include "indus/tolerances.hpp"
#include "indus/sparse.hpp"

namespace indus::la {

struct Inertia {
    int num_positive = 0;
    int num_zero = 0;
    int num_negative = 0;

    [[nodiscard]] bool is_positive_definite(int n) const noexcept {
        return num_positive == n && num_zero == 0 && num_negative == 0;
    }
    [[nodiscard]] bool is_positive_semidefinite() const noexcept {
        return num_negative == 0;
    }
};

class AmdOrdering {
public:
    // Computes fill-reducing permutation p and inverse permutation inv_p for an n x n symmetric matrix
    static void compute(int n, const SparseMatrixCSC& sym_matrix, std::vector<int>& perm, std::vector<int>& inv_perm);
};

class SparseLDL {
public:
    SparseLDL() = default;
    explicit SparseLDL(int dim);

    // Factorize symmetric matrix A (only lower or upper triangular entries need to be provided)
    bool factorize(const SparseMatrixCSC& A, bool use_amd = true);

    // Solve A x = b (in place in x)
    void solve(std::span<double> x) const;

    [[nodiscard]] const Inertia& inertia() const noexcept { return inertia_; }
    [[nodiscard]] bool is_positive_definite() const noexcept { return inertia_.is_positive_definite(n_); }
    [[nodiscard]] bool is_positive_semidefinite() const noexcept { return inertia_.is_positive_semidefinite(); }

    // Extracts negative curvature direction v such that v^T A v < 0 if indefinite
    bool get_negative_curvature_witness(std::vector<double>& v) const;

    [[nodiscard]] int dimension() const noexcept { return n_; }
    [[nodiscard]] const std::vector<double>& diagonal() const noexcept { return D_; }

    void clear();

private:
    int n_ = 0;
    Inertia inertia_;

    std::vector<int> perm_;
    std::vector<int> inv_perm_;

    // Elimination tree
    std::vector<int> parent_;

    // Factor L (unit lower triangular CSC)
    SparseMatrixCSC L_;
    SparseMatrixCSR L_csr_;

    // Factor D (diagonal)
    std::vector<double> D_;

    int first_negative_pivot_idx_ = -1;
};

} // namespace indus::la
