#include "src/linalg/lu.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <queue>

namespace indus::la {

SparseLU::SparseLU(int dim) : m_(dim) {}

void SparseLU::clear() {
    m_ = 0;
    is_repaired_ = false;
    repaired_rows_.clear();
    p_row_.clear();
    inv_p_row_.clear();
    q_col_.clear();
    inv_q_col_.clear();
    L_csc_.clear();
    L_csr_.clear();
    U_csc_.clear();
    U_csr_.clear();
    u_diag_.clear();
    etas_.clear();
}

FactorizationStatus SparseLU::factorize(const SparseMatrixCSC& B, double markowitz_threshold) {
    clear();
    m_ = B.m;
    if (m_ <= 0 || B.n != m_) {
        return FactorizationStatus::kFailed;
    }

    p_row_.assign(static_cast<size_t>(m_), -1);
    inv_p_row_.assign(static_cast<size_t>(m_), -1);
    q_col_.assign(static_cast<size_t>(m_), -1);
    inv_q_col_.assign(static_cast<size_t>(m_), -1);

    // Active matrix representation using linked lists / adjacency vectors
    // row_entries[i] = vector of (col, val)
    std::vector<std::vector<std::pair<int, double>>> row_entries(static_cast<size_t>(m_));
    // col_entries[j] = vector of (row, val)
    std::vector<std::vector<std::pair<int, double>>> col_entries(static_cast<size_t>(m_));

    for (int j = 0; j < m_; ++j) {
        const auto rows = B.col_rows(j);
        const auto vals = B.col_vals(j);
        const size_t sz = rows.size();
        for (size_t k = 0; k < sz; ++k) {
            const int r = rows[k];
            const double v = vals[k];
            if (std::abs(v) > tol::kZeroDrop) {
                row_entries[static_cast<size_t>(r)].emplace_back(j, v);
                col_entries[static_cast<size_t>(j)].emplace_back(r, v);
            }
        }
    }

    std::vector<bool> row_elim(static_cast<size_t>(m_), false);
    std::vector<bool> col_elim(static_cast<size_t>(m_), false);

    // Dynamic degree counts
    std::vector<int> row_deg(static_cast<size_t>(m_), 0);
    std::vector<int> col_deg(static_cast<size_t>(m_), 0);
    for (int i = 0; i < m_; ++i) row_deg[static_cast<size_t>(i)] = static_cast<int>(row_entries[static_cast<size_t>(i)].size());
    for (int j = 0; j < m_; ++j) col_deg[static_cast<size_t>(j)] = static_cast<int>(col_entries[static_cast<size_t>(j)].size());

    // Triplets for L and U in permuted step coordinates (step k = 0 ... m - 1)
    std::vector<Triplet> L_triplets;
    std::vector<Triplet> U_triplets;
    u_diag_.assign(static_cast<size_t>(m_), 0.0);

    // Work buffer for row expansion during Gaussian elimination
    std::vector<double> dense_row(static_cast<size_t>(m_), 0.0);
    std::vector<bool> in_dense_row(static_cast<size_t>(m_), false);
    std::vector<int> touched_cols;

    for (int step = 0; step < m_; ++step) {
        int best_p = -1;
        int best_q = -1;
        int64_t best_merit = std::numeric_limits<int64_t>::max();
        double best_pivot_val = 0.0;

        // Search strategy:
        // 1. Check singleton columns (merit = 0)
        for (int j = 0; j < m_; ++j) {
            if (col_elim[static_cast<size_t>(j)] || col_deg[static_cast<size_t>(j)] != 1) continue;
            for (const auto& [r, v] : col_entries[static_cast<size_t>(j)]) {
                if (!row_elim[static_cast<size_t>(r)] && std::abs(v) > tol::kPivotTolerance) {
                    best_p = r;
                    best_q = j;
                    best_pivot_val = v;
                    best_merit = 0;
                    break;
                }
            }
            if (best_p != -1) break;
        }

        // 2. Check singleton rows (merit = 0)
        if (best_p == -1) {
            for (int i = 0; i < m_; ++i) {
                if (row_elim[static_cast<size_t>(i)] || row_deg[static_cast<size_t>(i)] != 1) continue;
                for (const auto& [c, v] : row_entries[static_cast<size_t>(i)]) {
                    if (!col_elim[static_cast<size_t>(c)] && std::abs(v) > tol::kPivotTolerance) {
                        // Check column threshold criterion
                        double max_col = 0.0;
                        for (const auto& [cr, cv] : col_entries[static_cast<size_t>(c)]) {
                            if (!row_elim[static_cast<size_t>(cr)]) {
                                max_col = std::max(max_col, std::abs(cv));
                            }
                        }
                        if (std::abs(v) >= markowitz_threshold * max_col) {
                            best_p = i;
                            best_q = c;
                            best_pivot_val = v;
                            best_merit = 0;
                            break;
                        }
                    }
                }
                if (best_p != -1) break;
            }
        }

        // 3. General Markowitz search over columns with lowest degrees
        if (best_p == -1) {
            int columns_searched = 0;
            constexpr int kMaxSearchColumns = 12;

            // Find columns sorted by degree
            std::vector<int> candidate_cols;
            for (int j = 0; j < m_; ++j) {
                if (!col_elim[static_cast<size_t>(j)] && col_deg[static_cast<size_t>(j)] > 0) {
                    candidate_cols.push_back(j);
                }
            }
            std::sort(candidate_cols.begin(), candidate_cols.end(), [&](int a, int b) {
                return col_deg[static_cast<size_t>(a)] < col_deg[static_cast<size_t>(b)];
            });

            for (int q : candidate_cols) {
                double max_col = 0.0;
                for (const auto& [r, v] : col_entries[static_cast<size_t>(q)]) {
                    if (!row_elim[static_cast<size_t>(r)]) {
                        max_col = std::max(max_col, std::abs(v));
                    }
                }

                if (max_col <= tol::kPivotTolerance) continue;

                for (const auto& [p, v] : col_entries[static_cast<size_t>(q)]) {
                    if (row_elim[static_cast<size_t>(p)]) continue;
                    if (std::abs(v) >= markowitz_threshold * max_col && std::abs(v) > tol::kPivotTolerance) {
                        const int64_t merit = static_cast<int64_t>(row_deg[static_cast<size_t>(p)] - 1) * 
                                              static_cast<int64_t>(col_deg[static_cast<size_t>(q)] - 1);
                        if (merit < best_merit) {
                            best_merit = merit;
                            best_p = p;
                            best_q = q;
                            best_pivot_val = v;
                            if (merit == 0) break;
                        }
                    }
                }

                if (best_merit == 0) break;
                if (++columns_searched >= kMaxSearchColumns && best_p != -1) break;
            }
        }

        // 4. Singularity / Rank Defect Repair
        if (best_p == -1) {
            // Find first uneliminated row and column
            for (int i = 0; i < m_; ++i) {
                if (!row_elim[static_cast<size_t>(i)]) { best_p = i; break; }
            }
            for (int j = 0; j < m_; ++j) {
                if (!col_elim[static_cast<size_t>(j)]) { best_q = j; break; }
            }
            best_pivot_val = 1.0;
            is_repaired_ = true;
            repaired_rows_.push_back(best_p);
        }

        // Record permutation for step
        p_row_[static_cast<size_t>(step)] = best_p;
        inv_p_row_[static_cast<size_t>(best_p)] = step;
        q_col_[static_cast<size_t>(step)] = best_q;
        inv_q_col_[static_cast<size_t>(best_q)] = step;

        row_elim[static_cast<size_t>(best_p)] = true;
        col_elim[static_cast<size_t>(best_q)] = true;

        // Diagonal of U at step
        u_diag_[static_cast<size_t>(step)] = best_pivot_val;
        U_triplets.push_back({step, step, best_pivot_val});
        L_triplets.push_back({step, step, 1.0});

        // Collect row p non-zero entries in U
        touched_cols.clear();
        for (const auto& [c, v] : row_entries[static_cast<size_t>(best_p)]) {
            if (c == best_q || col_elim[static_cast<size_t>(c)]) continue;
            dense_row[static_cast<size_t>(c)] = v;
            in_dense_row[static_cast<size_t>(c)] = true;
            touched_cols.push_back(c);
        }

        // Multipliers L_{i, step} for rows sharing column best_q
        std::vector<std::pair<int, double>> multipliers;
        for (const auto& [r, v] : col_entries[static_cast<size_t>(best_q)]) {
            if (r == best_p || row_elim[static_cast<size_t>(r)]) continue;
            const double mult = v / best_pivot_val;
            if (std::abs(mult) > tol::kZeroDrop) {
                multipliers.emplace_back(r, mult);
            }
        }

        // Outer product updates: A_{i, c} -= mult * A_{p, c}
        for (const auto& [r, mult] : multipliers) {
            // Unpack row r into dense row buffer
            std::vector<int> r_touched;
            std::vector<double> r_dense(static_cast<size_t>(m_), 0.0);
            for (const auto& [c, v] : row_entries[static_cast<size_t>(r)]) {
                if (!col_elim[static_cast<size_t>(c)]) {
                    r_dense[static_cast<size_t>(c)] = v;
                    r_touched.push_back(c);
                }
            }

            // Subtract multiplier * pivot row
            for (int c : touched_cols) {
                if (r_dense[static_cast<size_t>(c)] == 0.0) {
                    r_touched.push_back(c);
                }
                r_dense[static_cast<size_t>(c)] -= mult * dense_row[static_cast<size_t>(c)];
            }

            // Repack row r
            row_entries[static_cast<size_t>(r)].clear();
            for (int c : r_touched) {
                const double val = r_dense[static_cast<size_t>(c)];
                if (std::abs(val) > tol::kZeroDrop) {
                    row_entries[static_cast<size_t>(r)].emplace_back(c, val);
                }
            }
            row_deg[static_cast<size_t>(r)] = static_cast<int>(row_entries[static_cast<size_t>(r)].size());
        }

        // Clean up dense_row
        for (int c : touched_cols) {
            dense_row[static_cast<size_t>(c)] = 0.0;
            in_dense_row[static_cast<size_t>(c)] = false;
        }

        // Rebuild column entries for columns in touched_cols
        for (int c : touched_cols) {
            col_entries[static_cast<size_t>(c)].clear();
            for (int r = 0; r < m_; ++r) {
                if (row_elim[static_cast<size_t>(r)]) continue;
                for (const auto& [col_idx, v] : row_entries[static_cast<size_t>(r)]) {
                    if (col_idx == c) {
                        col_entries[static_cast<size_t>(c)].emplace_back(r, v);
                        break;
                    }
                }
            }
            col_deg[static_cast<size_t>(c)] = static_cast<int>(col_entries[static_cast<size_t>(c)].size());
        }
    }

    // Now all steps 0..m-1 are determined!
    // Finalize L and U in step coordinates:
    // U contains entries of row best_p at each step:
    // L contains multipliers at each step:

    // Direct construction:
    // We already know P and Q!
    // Re-run clean numeric elimination with permutation to build L_triplets and U_triplets precisely:
    std::vector<std::vector<std::pair<int, double>>> perm_rows(static_cast<size_t>(m_));
    for (int i = 0; i < m_; ++i) {
        const int orig_r = p_row_[static_cast<size_t>(i)];
        for (int j = 0; j < m_; ++j) {
            const int orig_c = q_col_[static_cast<size_t>(j)];
            const double val = B.coeff(orig_r, orig_c);
            if (std::abs(val) > tol::kZeroDrop) {
                perm_rows[static_cast<size_t>(i)].emplace_back(j, val);
            }
        }
    }

    std::vector<Triplet> exact_L_triplets;
    std::vector<Triplet> exact_U_triplets;

    for (int k = 0; k < m_; ++k) {
        // Expand row k into dense buffer
        std::vector<double> rk(static_cast<size_t>(m_), 0.0);
        for (const auto& [col, val] : perm_rows[static_cast<size_t>(k)]) {
            rk[static_cast<size_t>(col)] = val;
        }

        // Eliminate against previous rows j < k
        exact_L_triplets.push_back({k, k, 1.0});
        for (int j = 0; j < k; ++j) {
            if (std::abs(rk[static_cast<size_t>(j)]) > tol::kZeroDrop) {
                const double u_jj = u_diag_[static_cast<size_t>(j)];
                const double mult = (std::abs(u_jj) > tol::kSingularityTolerance) ? (rk[static_cast<size_t>(j)] / u_jj) : 0.0;
                if (std::abs(mult) > tol::kZeroDrop) {
                    exact_L_triplets.push_back({k, j, mult});
                    // Subtract mult * row j
                    for (const auto& [col, val] : perm_rows[static_cast<size_t>(j)]) {
                        if (col >= j) {
                            rk[static_cast<size_t>(col)] -= mult * val;
                        }
                    }
                }
                rk[static_cast<size_t>(j)] = 0.0;
            }
        }

        // Store row k of U
        double diag_k = rk[static_cast<size_t>(k)];
        if (std::abs(diag_k) < tol::kSingularityTolerance) {
            diag_k = 1.0;
            is_repaired_ = true;
        }
        u_diag_[static_cast<size_t>(k)] = diag_k;
        exact_U_triplets.push_back({k, k, diag_k});

        perm_rows[static_cast<size_t>(k)].clear();
        perm_rows[static_cast<size_t>(k)].emplace_back(k, diag_k);

        for (int j = k + 1; j < m_; ++j) {
            if (std::abs(rk[static_cast<size_t>(j)]) > tol::kZeroDrop) {
                exact_U_triplets.push_back({k, j, rk[static_cast<size_t>(j)]});
                perm_rows[static_cast<size_t>(k)].emplace_back(j, rk[static_cast<size_t>(j)]);
            }
        }
    }

    L_csc_.set_from_triplets(m_, m_, exact_L_triplets);
    L_csr_ = L_csc_.to_csr();

    U_csc_.set_from_triplets(m_, m_, exact_U_triplets);
    U_csr_ = U_csc_.to_csr();

    etas_.clear();
    return is_repaired_ ? FactorizationStatus::kSingularRepaired : FactorizationStatus::kSuccess;
}

void SparseLU::solve_L_forward(std::span<double> x) const {
    // Solve L * z = x in place (L is unit lower triangular CSR)
    for (int i = 0; i < m_; ++i) {
        const auto cols = L_csr_.row_cols(i);
        const auto vals = L_csr_.row_vals(i);
        const size_t sz = cols.size();
        double sum = x[static_cast<size_t>(i)];
        for (size_t k = 0; k < sz; ++k) {
            const int c = cols[k];
            if (c < i) {
                sum -= vals[k] * x[static_cast<size_t>(c)];
            }
        }
        x[static_cast<size_t>(i)] = sum;
    }
}

void SparseLU::solve_L_transpose_backward(std::span<double> x) const {
    // Solve L^T * z = x in place (L is unit lower triangular)
    for (int i = m_ - 1; i >= 0; --i) {
        const double xi = x[static_cast<size_t>(i)];
        if (std::abs(xi) <= tol::kZeroDrop) continue;
        const auto cols = L_csr_.row_cols(i);
        const auto vals = L_csr_.row_vals(i);
        const size_t sz = cols.size();
        for (size_t k = 0; k < sz; ++k) {
            const int c = cols[k];
            if (c < i) {
                x[static_cast<size_t>(c)] -= vals[k] * xi;
            }
        }
    }
}

void SparseLU::solve_U_backward(std::span<double> x) const {
    // Solve U * w = x in place (U is upper triangular CSR)
    for (int i = m_ - 1; i >= 0; --i) {
        const auto cols = U_csr_.row_cols(i);
        const auto vals = U_csr_.row_vals(i);
        const size_t sz = cols.size();
        double sum = x[static_cast<size_t>(i)];
        double diag = u_diag_[static_cast<size_t>(i)];
        for (size_t k = 0; k < sz; ++k) {
            const int c = cols[k];
            if (c > i) {
                sum -= vals[k] * x[static_cast<size_t>(c)];
            } else if (c == i) {
                diag = vals[k];
            }
        }
        x[static_cast<size_t>(i)] = (std::abs(diag) > tol::kSingularityTolerance) ? (sum / diag) : sum;
    }
}

void SparseLU::solve_U_transpose_forward(std::span<double> x) const {
    // Solve U^T * w = x in place
    for (int i = 0; i < m_; ++i) {
        const double diag = u_diag_[static_cast<size_t>(i)];
        const double wi = (std::abs(diag) > tol::kSingularityTolerance) ? (x[static_cast<size_t>(i)] / diag) : x[static_cast<size_t>(i)];
        x[static_cast<size_t>(i)] = wi;
        if (std::abs(wi) <= tol::kZeroDrop) continue;

        const auto cols = U_csr_.row_cols(i);
        const auto vals = U_csr_.row_vals(i);
        const size_t sz = cols.size();
        for (size_t k = 0; k < sz; ++k) {
            const int c = cols[k];
            if (c > i) {
                x[static_cast<size_t>(c)] -= vals[k] * wi;
            }
        }
    }
}

void SparseLU::ftran(std::span<double> x) const {
    if (m_ <= 0) return;

    // 1. Permute b by P: step_work[k] = b[p_row_[k]]
    std::vector<double> work(static_cast<size_t>(m_));
    for (int k = 0; k < m_; ++k) {
        work[static_cast<size_t>(k)] = x[static_cast<size_t>(p_row_[static_cast<size_t>(k)])];
    }

    // 2. Solve L * z = P * b
    solve_L_forward(work);

    // 3. Solve U * w = z
    solve_U_backward(work);

    // 4. Unpermute by Q: sol[q_col_[k]] = work[k]
    for (int k = 0; k < m_; ++k) {
        x[static_cast<size_t>(q_col_[static_cast<size_t>(k)])] = work[static_cast<size_t>(k)];
    }

    // 5. Apply PFI eta vectors in forward order: E_1, ..., E_k
    for (const auto& eta : etas_) {
        const double s = x[static_cast<size_t>(eta.pivot_row)];
        if (std::abs(s) <= tol::kZeroDrop) continue;

        x[static_cast<size_t>(eta.pivot_row)] = s * eta.pivot_val;
        const size_t sz = eta.rows.size();
        for (size_t k = 0; k < sz; ++k) {
            x[static_cast<size_t>(eta.rows[k])] += s * eta.values[k];
        }
    }
}

void SparseLU::btran(std::span<double> y) const {
    if (m_ <= 0) return;

    // 1. Apply PFI eta vectors in reverse order
    for (auto it = etas_.rbegin(); it != etas_.rend(); ++it) {
        const auto& eta = *it;
        double dot = eta.pivot_val * y[static_cast<size_t>(eta.pivot_row)];
        const size_t sz = eta.rows.size();
        for (size_t k = 0; k < sz; ++k) {
            dot += eta.values[k] * y[static_cast<size_t>(eta.rows[k])];
        }
        y[static_cast<size_t>(eta.pivot_row)] = dot;
    }

    // 2. Permute by Q^T: work[k] = y[q_col_[k]]
    std::vector<double> work(static_cast<size_t>(m_));
    for (int k = 0; k < m_; ++k) {
        work[static_cast<size_t>(k)] = y[static_cast<size_t>(q_col_[static_cast<size_t>(k)])];
    }

    // 3. Solve U^T * w = Q^T * y
    solve_U_transpose_forward(work);

    // 4. Solve L^T * z = w
    solve_L_transpose_backward(work);

    // 5. Unpermute by P^T: y[p_row_[k]] = work[k]
    for (int k = 0; k < m_; ++k) {
        y[static_cast<size_t>(p_row_[static_cast<size_t>(k)])] = work[static_cast<size_t>(k)];
    }
}

void SparseLU::dfs_L(int node, std::vector<bool>& visited, std::vector<int>& order) const {
    visited[static_cast<size_t>(node)] = true;
    const auto rows = L_csc_.col_rows(node);
    for (int next : rows) {
        if (next > node && !visited[static_cast<size_t>(next)]) {
            dfs_L(next, visited, order);
        }
    }
    order.push_back(node);
}

void SparseLU::ftran_sparse(const std::vector<int>& rhs_indices, const std::vector<double>& rhs_values,
                            std::vector<int>& sol_indices, std::vector<double>& sol_values, 
                            std::vector<double>& dense_work) const {
    if (dense_work.size() < static_cast<size_t>(m_)) {
        dense_work.assign(static_cast<size_t>(m_), 0.0);
    } else {
        std::fill(dense_work.begin(), dense_work.end(), 0.0);
    }

    const size_t rhs_sz = rhs_indices.size();
    for (size_t k = 0; k < rhs_sz; ++k) {
        dense_work[static_cast<size_t>(rhs_indices[k])] = rhs_values[k];
    }

    ftran(dense_work);

    sol_indices.clear();
    sol_values.clear();
    for (int i = 0; i < m_; ++i) {
        if (std::abs(dense_work[static_cast<size_t>(i)]) > tol::kZeroDrop) {
            sol_indices.push_back(i);
            sol_values.push_back(dense_work[static_cast<size_t>(i)]);
        }
    }
}

void SparseLU::btran_sparse(const std::vector<int>& rhs_indices, const std::vector<double>& rhs_values,
                            std::vector<int>& sol_indices, std::vector<double>& sol_values,
                            std::vector<double>& dense_work) const {
    if (dense_work.size() < static_cast<size_t>(m_)) {
        dense_work.assign(static_cast<size_t>(m_), 0.0);
    } else {
        std::fill(dense_work.begin(), dense_work.end(), 0.0);
    }

    const size_t rhs_sz = rhs_indices.size();
    for (size_t k = 0; k < rhs_sz; ++k) {
        dense_work[static_cast<size_t>(rhs_indices[k])] = rhs_values[k];
    }

    btran(dense_work);

    sol_indices.clear();
    sol_values.clear();
    for (int i = 0; i < m_; ++i) {
        if (std::abs(dense_work[static_cast<size_t>(i)]) > tol::kZeroDrop) {
            sol_indices.push_back(i);
            sol_values.push_back(dense_work[static_cast<size_t>(i)]);
        }
    }
}

bool SparseLU::update_basis_pfi(int leaving_row, std::span<const double> alpha) {
    if (leaving_row < 0 || leaving_row >= m_) return false;
    const double pivot_elem = alpha[static_cast<size_t>(leaving_row)];
    if (std::abs(pivot_elem) <= tol::kPivotTolerance) {
        return false; // Pivot too small, requires refactorization
    }

    EtaVector eta;
    eta.pivot_row = leaving_row;
    eta.pivot_val = 1.0 / pivot_elem;

    for (int i = 0; i < m_; ++i) {
        if (i == leaving_row) continue;
        const double val = alpha[static_cast<size_t>(i)];
        if (std::abs(val) > tol::kZeroDrop) {
            eta.rows.push_back(i);
            eta.values.push_back(-val / pivot_elem);
        }
    }

    etas_.push_back(std::move(eta));
    return true;
}

} // namespace indus::la
