#include "indus/sparse.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace indus::la {

// ============================================================================
// SparseMatrixCSC Implementation
// ============================================================================

SparseMatrixCSC::SparseMatrixCSC(int rows, int cols)
    : m(rows), n(cols), col_ptr(static_cast<size_t>(cols + 1), 0) {}

void SparseMatrixCSC::clear() noexcept {
    m = 0;
    n = 0;
    col_ptr.clear();
    row_idx.clear();
    values.clear();
}

void SparseMatrixCSC::resize(int rows, int cols) {
    m = rows;
    n = cols;
    col_ptr.assign(static_cast<size_t>(cols + 1), 0);
    row_idx.clear();
    values.clear();
}

double SparseMatrixCSC::coeff(int r, int c) const {
    if (r < 0 || r >= m || c < 0 || c >= n || col_ptr.empty()) {
        return 0.0;
    }
    const auto rows = col_rows(c);
    const auto vals = col_vals(c);
    const auto it = std::lower_bound(rows.begin(), rows.end(), r);
    if (it != rows.end() && *it == r) {
        const auto idx = std::distance(rows.begin(), it);
        return vals[static_cast<size_t>(idx)];
    }
    return 0.0;
}

void SparseMatrixCSC::set_from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates) {
    m = rows;
    n = cols;
    col_ptr.assign(static_cast<size_t>(cols + 1), 0);
    row_idx.clear();
    values.clear();

    if (triplets.empty()) {
        return;
    }

    // Step 1: Count entries per column
    for (const auto& t : triplets) {
        if (t.row >= 0 && t.row < m && t.col >= 0 && t.col < n && std::abs(t.value) > tol::kZeroDrop) {
            col_ptr[static_cast<size_t>(t.col + 1)]++;
        }
    }

    // Step 2: Prefix sum for column pointers
    for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
        col_ptr[j + 1] += col_ptr[j];
    }

    const int64_t total_entries = col_ptr[static_cast<size_t>(n)];
    row_idx.resize(static_cast<size_t>(total_entries));
    values.resize(static_cast<size_t>(total_entries));

    // Temporary copy of col_ptr to track insertion positions
    std::vector<int64_t> cursor = col_ptr;

    // Step 3: Populate row_idx and values
    for (const auto& t : triplets) {
        if (t.row >= 0 && t.row < m && t.col >= 0 && t.col < n && std::abs(t.value) > tol::kZeroDrop) {
            const int64_t dest = cursor[static_cast<size_t>(t.col)]++;
            row_idx[static_cast<size_t>(dest)] = t.row;
            values[static_cast<size_t>(dest)] = t.value;
        }
    }

    // Step 4: Sort each column by row index and merge duplicates if requested
    std::vector<int> col_r;
    std::vector<double> col_v;
    std::vector<int> new_row_idx;
    std::vector<double> new_values;
    new_row_idx.reserve(static_cast<size_t>(total_entries));
    new_values.reserve(static_cast<size_t>(total_entries));

    std::vector<int64_t> new_col_ptr(static_cast<size_t>(n + 1), 0);

    for (int j = 0; j < n; ++j) {
        new_col_ptr[static_cast<size_t>(j)] = static_cast<int64_t>(new_row_idx.size());
        const int64_t start = col_ptr[static_cast<size_t>(j)];
        const int64_t end = col_ptr[static_cast<size_t>(j + 1)];
        const size_t count = static_cast<size_t>(end - start);

        if (count == 0) continue;

        // Pair sort
        std::vector<std::pair<int, double>> pairs(count);
        for (size_t k = 0; k < count; ++k) {
            pairs[k] = {row_idx[static_cast<size_t>(start + static_cast<int64_t>(k))],
                        values[static_cast<size_t>(start + static_cast<int64_t>(k))]};
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        for (size_t k = 0; k < count; ++k) {
            if (sum_duplicates && !new_row_idx.empty() && 
                static_cast<int64_t>(new_row_idx.size()) > new_col_ptr[static_cast<size_t>(j)] &&
                new_row_idx.back() == pairs[k].first) {
                new_values.back() += pairs[k].second;
            } else {
                new_row_idx.push_back(pairs[k].first);
                new_values.push_back(pairs[k].second);
            }
        }
    }
    new_col_ptr[static_cast<size_t>(n)] = static_cast<int64_t>(new_row_idx.size());

    col_ptr = std::move(new_col_ptr);
    row_idx = std::move(new_row_idx);
    values = std::move(new_values);
}

SparseMatrixCSC SparseMatrixCSC::from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates) {
    SparseMatrixCSC mat;
    mat.set_from_triplets(rows, cols, triplets, sum_duplicates);
    return mat;
}

void SparseMatrixCSC::transpose_into(SparseMatrixCSR& out) const {
    out.m = n;
    out.n = m;
    out.row_ptr = col_ptr;
    out.col_idx = row_idx;
    out.values = values;
}

SparseMatrixCSR SparseMatrixCSC::to_csr() const {
    SparseMatrixCSR csr(m, n);
    if (nnz() == 0) {
        return csr;
    }

    // Step 1: Count row frequencies
    std::vector<int64_t> row_counts(static_cast<size_t>(m), 0);
    for (const int r : row_idx) {
        row_counts[static_cast<size_t>(r)]++;
    }

    // Step 2: Prefix sum for row_ptr
    csr.row_ptr.assign(static_cast<size_t>(m + 1), 0);
    for (size_t i = 0; i < static_cast<size_t>(m); ++i) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + row_counts[i];
    }

    csr.col_idx.resize(static_cast<size_t>(nnz()));
    csr.values.resize(static_cast<size_t>(nnz()));

    std::vector<int64_t> cursor = csr.row_ptr;

    // Step 3: Distribute entries
    for (int j = 0; j < n; ++j) {
        const int64_t start = col_ptr[static_cast<size_t>(j)];
        const int64_t end = col_ptr[static_cast<size_t>(j + 1)];
        for (int64_t k = start; k < end; ++k) {
            const int r = row_idx[static_cast<size_t>(k)];
            const double val = values[static_cast<size_t>(k)];
            const int64_t dest = cursor[static_cast<size_t>(r)]++;
            csr.col_idx[static_cast<size_t>(dest)] = j;
            csr.values[static_cast<size_t>(dest)] = val;
        }
    }

    return csr;
}

SparseMatrixCSC SparseMatrixCSC::transpose() const {
    // Transpose of CSC(m, n) is a CSC matrix of dimension (n, m).
    // The CSR representation of A has row_ptr size m+1, col_idx, values.
    // Viewed as a CSC of A^T: col_ptr size m+1, row_idx, values.
    SparseMatrixCSR csr = to_csr();
    SparseMatrixCSC at(n, m);
    csr.transpose_into(at);
    return at;
}

void SparseMatrixCSC::multiply(std::span<const double> x, std::span<double> y, double alpha, double beta) const {
    if (beta == 0.0) {
        std::fill(y.begin(), y.end(), 0.0);
    } else if (beta != 1.0) {
        for (double& val : y) {
            val *= beta;
        }
    }

    for (int j = 0; j < n; ++j) {
        const double xj = x[static_cast<size_t>(j)];
        if (std::abs(xj) <= tol::kZeroDrop) continue;

        const double alpha_xj = alpha * xj;
        const auto rows = col_rows(j);
        const auto vals = col_vals(j);
        const size_t sz = rows.size();
        for (size_t k = 0; k < sz; ++k) {
            y[static_cast<size_t>(rows[k])] += alpha_xj * vals[k];
        }
    }
}

void SparseMatrixCSC::multiply_transpose(std::span<const double> x, std::span<double> y, double alpha, double beta) const {
    if (beta == 0.0) {
        std::fill(y.begin(), y.end(), 0.0);
    } else if (beta != 1.0) {
        for (double& val : y) {
            val *= beta;
        }
    }

    for (int j = 0; j < n; ++j) {
        const auto rows = col_rows(j);
        const auto vals = col_vals(j);
        double dot = 0.0;
        const size_t sz = rows.size();
        for (size_t k = 0; k < sz; ++k) {
            dot += vals[k] * x[static_cast<size_t>(rows[k])];
        }
        y[static_cast<size_t>(j)] += alpha * dot;
    }
}

double SparseMatrixCSC::norm_inf() const noexcept {
    if (m <= 0 || nnz() == 0) return 0.0;
    std::vector<double> row_sums(static_cast<size_t>(m), 0.0);
    for (size_t k = 0; k < static_cast<size_t>(nnz()); ++k) {
        row_sums[static_cast<size_t>(row_idx[k])] += std::abs(values[k]);
    }
    double max_norm = 0.0;
    for (double sum : row_sums) {
        if (sum > max_norm) max_norm = sum;
    }
    return max_norm;
}

double SparseMatrixCSC::norm_1() const noexcept {
    if (n <= 0 || nnz() == 0) return 0.0;
    double max_norm = 0.0;
    for (int j = 0; j < n; ++j) {
        const auto vals = col_vals(j);
        double sum = 0.0;
        for (double v : vals) {
            sum += std::abs(v);
        }
        if (sum > max_norm) max_norm = sum;
    }
    return max_norm;
}

double SparseMatrixCSC::norm_frobenius() const noexcept {
    double sum_sq = 0.0;
    for (double v : values) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

// ============================================================================
// SparseMatrixCSR Implementation
// ============================================================================

SparseMatrixCSR::SparseMatrixCSR(int rows, int cols)
    : m(rows), n(cols), row_ptr(static_cast<size_t>(rows + 1), 0) {}

void SparseMatrixCSR::clear() noexcept {
    m = 0;
    n = 0;
    row_ptr.clear();
    col_idx.clear();
    values.clear();
}

void SparseMatrixCSR::resize(int rows, int cols) {
    m = rows;
    n = cols;
    row_ptr.assign(static_cast<size_t>(rows + 1), 0);
    col_idx.clear();
    values.clear();
}

double SparseMatrixCSR::coeff(int r, int c) const {
    if (r < 0 || r >= m || c < 0 || c >= n || row_ptr.empty()) {
        return 0.0;
    }
    const auto cols = row_cols(r);
    const auto vals = row_vals(r);
    const auto it = std::lower_bound(cols.begin(), cols.end(), c);
    if (it != cols.end() && *it == c) {
        const auto idx = std::distance(cols.begin(), it);
        return vals[static_cast<size_t>(idx)];
    }
    return 0.0;
}

void SparseMatrixCSR::set_from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates) {
    m = rows;
    n = cols;
    row_ptr.assign(static_cast<size_t>(rows + 1), 0);
    col_idx.clear();
    values.clear();

    if (triplets.empty()) {
        return;
    }

    for (const auto& t : triplets) {
        if (t.row >= 0 && t.row < m && t.col >= 0 && t.col < n && std::abs(t.value) > tol::kZeroDrop) {
            row_ptr[static_cast<size_t>(t.row + 1)]++;
        }
    }

    for (size_t i = 0; i < static_cast<size_t>(m); ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    const int64_t total_entries = row_ptr[static_cast<size_t>(m)];
    col_idx.resize(static_cast<size_t>(total_entries));
    values.resize(static_cast<size_t>(total_entries));

    std::vector<int64_t> cursor = row_ptr;

    for (const auto& t : triplets) {
        if (t.row >= 0 && t.row < m && t.col >= 0 && t.col < n && std::abs(t.value) > tol::kZeroDrop) {
            const int64_t dest = cursor[static_cast<size_t>(t.row)]++;
            col_idx[static_cast<size_t>(dest)] = t.col;
            values[static_cast<size_t>(dest)] = t.value;
        }
    }

    std::vector<int> new_col_idx;
    std::vector<double> new_values;
    new_col_idx.reserve(static_cast<size_t>(total_entries));
    new_values.reserve(static_cast<size_t>(total_entries));

    std::vector<int64_t> new_row_ptr(static_cast<size_t>(m + 1), 0);

    for (int i = 0; i < m; ++i) {
        new_row_ptr[static_cast<size_t>(i)] = static_cast<int64_t>(new_col_idx.size());
        const int64_t start = row_ptr[static_cast<size_t>(i)];
        const int64_t end = row_ptr[static_cast<size_t>(i + 1)];
        const size_t count = static_cast<size_t>(end - start);

        if (count == 0) continue;

        std::vector<std::pair<int, double>> pairs(count);
        for (size_t k = 0; k < count; ++k) {
            pairs[k] = {col_idx[static_cast<size_t>(start + static_cast<int64_t>(k))],
                        values[static_cast<size_t>(start + static_cast<int64_t>(k))]};
        }
        std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        for (size_t k = 0; k < count; ++k) {
            if (sum_duplicates && !new_col_idx.empty() &&
                static_cast<int64_t>(new_col_idx.size()) > new_row_ptr[static_cast<size_t>(i)] &&
                new_col_idx.back() == pairs[k].first) {
                new_values.back() += pairs[k].second;
            } else {
                new_col_idx.push_back(pairs[k].first);
                new_values.push_back(pairs[k].second);
            }
        }
    }
    new_row_ptr[static_cast<size_t>(m)] = static_cast<int64_t>(new_col_idx.size());

    row_ptr = std::move(new_row_ptr);
    col_idx = std::move(new_col_idx);
    values = std::move(new_values);
}

SparseMatrixCSR SparseMatrixCSR::from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates) {
    SparseMatrixCSR mat;
    mat.set_from_triplets(rows, cols, triplets, sum_duplicates);
    return mat;
}

void SparseMatrixCSR::transpose_into(SparseMatrixCSC& out) const {
    out.m = n;
    out.n = m;
    out.col_ptr = row_ptr;
    out.row_idx = col_idx;
    out.values = values;
}

SparseMatrixCSC SparseMatrixCSR::to_csc() const {
    SparseMatrixCSC csc(m, n);
    if (nnz() == 0) {
        return csc;
    }

    std::vector<int64_t> col_counts(static_cast<size_t>(n), 0);
    for (const int c : col_idx) {
        col_counts[static_cast<size_t>(c)]++;
    }

    csc.col_ptr.assign(static_cast<size_t>(n + 1), 0);
    for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
        csc.col_ptr[j + 1] = csc.col_ptr[j] + col_counts[j];
    }

    csc.row_idx.resize(static_cast<size_t>(nnz()));
    csc.values.resize(static_cast<size_t>(nnz()));

    std::vector<int64_t> cursor = csc.col_ptr;

    for (int i = 0; i < m; ++i) {
        const int64_t start = row_ptr[static_cast<size_t>(i)];
        const int64_t end = row_ptr[static_cast<size_t>(i + 1)];
        for (int64_t k = start; k < end; ++k) {
            const int c = col_idx[static_cast<size_t>(k)];
            const double val = values[static_cast<size_t>(k)];
            const int64_t dest = cursor[static_cast<size_t>(c)]++;
            csc.row_idx[static_cast<size_t>(dest)] = i;
            csc.values[static_cast<size_t>(dest)] = val;
        }
    }

    return csc;
}

SparseMatrixCSR SparseMatrixCSR::transpose() const {
    SparseMatrixCSC csc = to_csc();
    SparseMatrixCSR at(n, m);
    csc.transpose_into(at);
    return at;
}

void SparseMatrixCSR::multiply(std::span<const double> x, std::span<double> y, double alpha, double beta) const {
    for (int i = 0; i < m; ++i) {
        const auto cols = row_cols(i);
        const auto vals = row_vals(i);
        double dot = 0.0;
        const size_t sz = cols.size();
        for (size_t k = 0; k < sz; ++k) {
            dot += vals[k] * x[static_cast<size_t>(cols[k])];
        }
        if (beta == 0.0) {
            y[static_cast<size_t>(i)] = alpha * dot;
        } else {
            y[static_cast<size_t>(i)] = alpha * dot + beta * y[static_cast<size_t>(i)];
        }
    }
}

void SparseMatrixCSR::multiply_transpose(std::span<const double> x, std::span<double> y, double alpha, double beta) const {
    if (beta == 0.0) {
        std::fill(y.begin(), y.end(), 0.0);
    } else if (beta != 1.0) {
        for (double& val : y) {
            val *= beta;
        }
    }

    for (int i = 0; i < m; ++i) {
        const double xi = x[static_cast<size_t>(i)];
        if (std::abs(xi) <= tol::kZeroDrop) continue;

        const double alpha_xi = alpha * xi;
        const auto cols = row_cols(i);
        const auto vals = row_vals(i);
        const size_t sz = cols.size();
        for (size_t k = 0; k < sz; ++k) {
            y[static_cast<size_t>(cols[k])] += alpha_xi * vals[k];
        }
    }
}

double SparseMatrixCSR::norm_inf() const noexcept {
    if (m <= 0 || nnz() == 0) return 0.0;
    double max_norm = 0.0;
    for (int i = 0; i < m; ++i) {
        const auto vals = row_vals(i);
        double sum = 0.0;
        for (double v : vals) {
            sum += std::abs(v);
        }
        if (sum > max_norm) max_norm = sum;
    }
    return max_norm;
}

double SparseMatrixCSR::norm_1() const noexcept {
    if (n <= 0 || nnz() == 0) return 0.0;
    std::vector<double> col_sums(static_cast<size_t>(n), 0.0);
    for (size_t k = 0; k < static_cast<size_t>(nnz()); ++k) {
        col_sums[static_cast<size_t>(col_idx[k])] += std::abs(values[k]);
    }
    double max_norm = 0.0;
    for (double sum : col_sums) {
        if (sum > max_norm) max_norm = sum;
    }
    return max_norm;
}

double SparseMatrixCSR::norm_frobenius() const noexcept {
    double sum_sq = 0.0;
    for (double v : values) {
        sum_sq += v * v;
    }
    return std::sqrt(sum_sq);
}

} // namespace indus::la
