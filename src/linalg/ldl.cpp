#include "src/linalg/ldl.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <set>

namespace indus::la {

// ============================================================================
// AmdOrdering Implementation
// ============================================================================

void AmdOrdering::compute(int n, const SparseMatrixCSC& sym_matrix, std::vector<int>& perm, std::vector<int>& inv_perm) {
    perm.assign(static_cast<size_t>(n), 0);
    inv_perm.assign(static_cast<size_t>(n), 0);

    if (n <= 0) return;
    if (n == 1) {
        perm[0] = 0;
        inv_perm[0] = 0;
        return;
    }

    // Build symmetric undirected adjacency graph
    std::vector<std::set<int>> adj(static_cast<size_t>(n));
    for (int j = 0; j < sym_matrix.n; ++j) {
        const auto rows = sym_matrix.col_rows(j);
        for (int r : rows) {
            if (r != j && r < n && j < n) {
                adj[static_cast<size_t>(j)].insert(r);
                adj[static_cast<size_t>(r)].insert(j);
            }
        }
    }

    std::vector<bool> eliminated(static_cast<size_t>(n), false);
    std::vector<int> degrees(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        degrees[static_cast<size_t>(i)] = static_cast<int>(adj[static_cast<size_t>(i)].size());
    }

    // Greedy Minimum Degree selection
    for (int step = 0; step < n; ++step) {
        int min_deg = std::numeric_limits<int>::max();
        int best_node = -1;

        for (int i = 0; i < n; ++i) {
            if (!eliminated[static_cast<size_t>(i)]) {
                const int deg = degrees[static_cast<size_t>(i)];
                if (deg < min_deg) {
                    min_deg = deg;
                    best_node = i;
                    if (min_deg == 0) break;
                }
            }
        }

        if (best_node == -1) {
            for (int i = 0; i < n; ++i) {
                if (!eliminated[static_cast<size_t>(i)]) {
                    best_node = i;
                    break;
                }
            }
        }

        perm[static_cast<size_t>(step)] = best_node;
        inv_perm[static_cast<size_t>(best_node)] = step;
        eliminated[static_cast<size_t>(best_node)] = true;

        // Quotient graph clique formation: connect all uneliminated neighbors of best_node
        std::vector<int> neighbors;
        for (int nbr : adj[static_cast<size_t>(best_node)]) {
            if (!eliminated[static_cast<size_t>(nbr)]) {
                neighbors.push_back(nbr);
            }
        }

        const size_t n_nbr = neighbors.size();
        for (size_t i = 0; i < n_nbr; ++i) {
            const int u = neighbors[i];
            adj[static_cast<size_t>(u)].erase(best_node);
            for (size_t j = i + 1; j < n_nbr; ++j) {
                const int v = neighbors[j];
                adj[static_cast<size_t>(u)].insert(v);
                adj[static_cast<size_t>(v)].insert(u);
            }
            degrees[static_cast<size_t>(u)] = static_cast<int>(adj[static_cast<size_t>(u)].size());
        }
    }
}

// ============================================================================
// SparseLDL Implementation
// ============================================================================

SparseLDL::SparseLDL(int dim) : n_(dim) {}

void SparseLDL::clear() {
    n_ = 0;
    inertia_ = {};
    perm_.clear();
    inv_perm_.clear();
    parent_.clear();
    L_.clear();
    L_csr_.clear();
    D_.clear();
    first_negative_pivot_idx_ = -1;
}

bool SparseLDL::factorize(const SparseMatrixCSC& A, bool use_amd) {
    clear();
    n_ = A.m;
    if (n_ <= 0 || A.n != n_) {
        return false;
    }

    if (use_amd) {
        AmdOrdering::compute(n_, A, perm_, inv_perm_);
    } else {
        perm_.resize(static_cast<size_t>(n_));
        inv_perm_.resize(static_cast<size_t>(n_));
        std::iota(perm_.begin(), perm_.end(), 0);
        std::iota(inv_perm_.begin(), inv_perm_.end(), 0);
    }

    // Build permuted symmetric matrix M = P * A * P^T
    std::vector<std::vector<std::pair<int, double>>> M_cols(static_cast<size_t>(n_));
    for (int j = 0; j < A.n; ++j) {
        const auto rows = A.col_rows(j);
        const auto vals = A.col_vals(j);
        const size_t sz = rows.size();
        for (size_t k = 0; k < sz; ++k) {
            const int r = rows[k];
            const double v = vals[k];
            if (std::abs(v) <= tol::kZeroDrop) continue;

            const int p_row = inv_perm_[static_cast<size_t>(r)];
            const int p_col = inv_perm_[static_cast<size_t>(j)];

            if (r == j) {
                M_cols[static_cast<size_t>(p_col)].emplace_back(p_row, v);
            } else if (r > j) {
                const double opp = A.coeff(j, r);
                const double sym_v = (std::abs(opp) > tol::kZeroDrop) ? (0.5 * (v + opp)) : v;
                M_cols[static_cast<size_t>(p_col)].emplace_back(p_row, sym_v);
                M_cols[static_cast<size_t>(p_row)].emplace_back(p_col, sym_v);
            } else { // r < j
                const double opp = A.coeff(j, r);
                if (std::abs(opp) <= tol::kZeroDrop) {
                    M_cols[static_cast<size_t>(p_col)].emplace_back(p_row, v);
                    M_cols[static_cast<size_t>(p_row)].emplace_back(p_col, v);
                }
            }
        }
    }

    // Up-looking Sparse LDL^T factorization
    D_.assign(static_cast<size_t>(n_), 0.0);
    std::vector<std::vector<std::pair<int, double>>> L_cols(static_cast<size_t>(n_));
    std::vector<double> work(static_cast<size_t>(n_), 0.0);
    std::vector<bool> in_touched(static_cast<size_t>(n_), false);
    std::vector<int> touched;

    for (int j = 0; j < n_; ++j) {
        touched.clear();

        // Step 1: Unpack column j of M (only lower entries i >= j) into work buffer
        for (const auto& [r, v] : M_cols[static_cast<size_t>(j)]) {
            if (r >= j) {
                if (!in_touched[static_cast<size_t>(r)]) {
                    in_touched[static_cast<size_t>(r)] = true;
                    touched.push_back(r);
                }
                work[static_cast<size_t>(r)] += v;
            }
        }

        // Step 2: Subtract L_{j, k} * D_k * L_{:, k} for previous columns k < j
        for (int k = 0; k < j; ++k) {
            double L_jk = 0.0;
            for (const auto& [r, v] : L_cols[static_cast<size_t>(k)]) {
                if (r == j) {
                    L_jk = v;
                    break;
                }
            }

            if (std::abs(L_jk) > tol::kZeroDrop) {
                const double mult = L_jk * D_[static_cast<size_t>(k)];
                for (const auto& [r, v] : L_cols[static_cast<size_t>(k)]) {
                    if (r >= j) {
                        if (!in_touched[static_cast<size_t>(r)]) {
                            in_touched[static_cast<size_t>(r)] = true;
                            touched.push_back(r);
                        }
                        work[static_cast<size_t>(r)] -= mult * v;
                    }
                }
            }
        }

        // Step 3: Compute diagonal D_j
        double diag_j = work[static_cast<size_t>(j)];
        if (std::abs(diag_j) < tol::kSingularityTolerance) {
            diag_j = 0.0;
            inertia_.num_zero++;
        } else if (diag_j < -tol::kPivotTolerance) {
            inertia_.num_negative++;
            if (first_negative_pivot_idx_ == -1) {
                first_negative_pivot_idx_ = j;
            }
        } else {
            inertia_.num_positive++;
        }
        D_[static_cast<size_t>(j)] = diag_j;

        // Step 4: Compute L_{i, j} = work[i] / D_j for i > j
        L_cols[static_cast<size_t>(j)].emplace_back(j, 1.0); // Diagonal of L is 1.0

        if (std::abs(diag_j) > tol::kSingularityTolerance) {
            // Sort touched rows for consistent ordering
            std::sort(touched.begin(), touched.end());
            for (int r : touched) {
                if (r > j) {
                    const double val = work[static_cast<size_t>(r)] / diag_j;
                    if (std::abs(val) > tol::kZeroDrop) {
                        L_cols[static_cast<size_t>(j)].emplace_back(r, val);
                    }
                }
            }
        }

        // Clean work buffer
        for (int r : touched) {
            work[static_cast<size_t>(r)] = 0.0;
            in_touched[static_cast<size_t>(r)] = false;
        }
    }

    // Assemble L_ into SparseMatrixCSC
    std::vector<Triplet> L_triplets;
    for (int j = 0; j < n_; ++j) {
        for (const auto& [r, v] : L_cols[static_cast<size_t>(j)]) {
            L_triplets.push_back({r, j, v});
        }
    }
    L_.set_from_triplets(n_, n_, L_triplets);
    L_csr_ = L_.to_csr();

    return true;
}

void SparseLDL::solve(std::span<double> x) const {
    if (n_ <= 0) return;

    // 1. Permute b: work[i] = x[perm_[i]]
    std::vector<double> work(static_cast<size_t>(n_));
    for (int i = 0; i < n_; ++i) {
        work[static_cast<size_t>(i)] = x[static_cast<size_t>(perm_[static_cast<size_t>(i)])];
    }

    // 2. Forward solve: L * z = work (unit lower triangular)
    for (int i = 0; i < n_; ++i) {
        const auto cols = L_csr_.row_cols(i);
        const auto vals = L_csr_.row_vals(i);
        const size_t sz = cols.size();
        double sum = work[static_cast<size_t>(i)];
        for (size_t k = 0; k < sz; ++k) {
            const int c = cols[k];
            if (c < i) {
                sum -= vals[k] * work[static_cast<size_t>(c)];
            }
        }
        work[static_cast<size_t>(i)] = sum;
    }

    // 3. Diagonal solve: D * w = z
    for (int i = 0; i < n_; ++i) {
        const double d = D_[static_cast<size_t>(i)];
        if (std::abs(d) > tol::kSingularityTolerance) {
            work[static_cast<size_t>(i)] /= d;
        } else {
            work[static_cast<size_t>(i)] = 0.0;
        }
    }

    // 4. Backward solve: L^T * y = w (unit upper triangular)
    for (int i = n_ - 1; i >= 0; --i) {
        const auto rows = L_.col_rows(i);
        const auto vals = L_.col_vals(i);
        const size_t sz = rows.size();
        double sum = work[static_cast<size_t>(i)];
        for (size_t k = 0; k < sz; ++k) {
            const int r = rows[k];
            if (r > i) {
                sum -= vals[k] * work[static_cast<size_t>(r)];
            }
        }
        work[static_cast<size_t>(i)] = sum;
    }

    // 5. Unpermute: x[perm_[i]] = work[i]
    for (int i = 0; i < n_; ++i) {
        x[static_cast<size_t>(perm_[static_cast<size_t>(i)])] = work[static_cast<size_t>(i)];
    }
}

bool SparseLDL::get_negative_curvature_witness(std::vector<double>& v) const {
    v.assign(static_cast<size_t>(n_), 0.0);
    if (first_negative_pivot_idx_ < 0 || first_negative_pivot_idx_ >= n_) {
        return false;
    }

    const int k = first_negative_pivot_idx_;
    // Solve L^T * w = e_k in permuted coordinates
    std::vector<double> w(static_cast<size_t>(n_), 0.0);
    w[static_cast<size_t>(k)] = 1.0;

    for (int i = n_ - 1; i >= 0; --i) {
        const auto rows = L_.col_rows(i);
        const auto vals = L_.col_vals(i);
        const size_t sz = rows.size();
        double sum = w[static_cast<size_t>(i)];
        for (size_t idx = 0; idx < sz; ++idx) {
            const int r = rows[idx];
            if (r > i) {
                sum -= vals[idx] * w[static_cast<size_t>(r)];
            }
        }
        w[static_cast<size_t>(i)] = sum;
    }

    // Unpermute: v = P^T * w, so v[perm_[i]] = w[i]
    for (int i = 0; i < n_; ++i) {
        v[static_cast<size_t>(perm_[static_cast<size_t>(i)])] = w[static_cast<size_t>(i)];
    }

    // Normalize witness vector ||v||_2 = 1.0
    double norm_sq = 0.0;
    for (double val : v) norm_sq += val * val;
    if (norm_sq > 0.0) {
        const double inv_norm = 1.0 / std::sqrt(norm_sq);
        for (double& val : v) val *= inv_norm;
    }

    return true;
}

} // namespace indus::la
