#pragma once

#include <vector>
#include <cstdint>
#include <span>
#include <string>
#include "indus/types.hpp"
#include "indus/tolerances.hpp"

namespace indus::la {

struct SparseMatrixCSR;

struct Triplet {
    int row = 0;
    int col = 0;
    double value = 0.0;
};

struct SparseMatrixCSC {
    int m = 0;                          // Rows
    int n = 0;                          // Columns
    std::vector<int64_t> col_ptr;       // Size n + 1
    std::vector<int> row_idx;           // Size nnz
    std::vector<double> values;         // Size nnz

    SparseMatrixCSC() = default;
    SparseMatrixCSC(int rows, int cols);

    [[nodiscard]] int64_t nnz() const noexcept { 
        return static_cast<int64_t>(values.size()); 
    }

    [[nodiscard]] double coeff(int r, int c) const;

    [[nodiscard]] std::span<const int> col_rows(int c) const {
        if (c < 0 || c >= n || col_ptr.empty()) return {};
        const size_t start = static_cast<size_t>(col_ptr[c]);
        const size_t count = static_cast<size_t>(col_ptr[c + 1] - col_ptr[c]);
        return {&row_idx[start], count};
    }

    [[nodiscard]] std::span<const double> col_vals(int c) const {
        if (c < 0 || c >= n || col_ptr.empty()) return {};
        const size_t start = static_cast<size_t>(col_ptr[c]);
        const size_t count = static_cast<size_t>(col_ptr[c + 1] - col_ptr[c]);
        return {&values[start], count};
    }

    void clear() noexcept;
    void resize(int rows, int cols);
    void set_from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates = true);
    static SparseMatrixCSC from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates = true);

    void transpose_into(SparseMatrixCSR& out) const;
    [[nodiscard]] SparseMatrixCSC transpose() const;
    [[nodiscard]] SparseMatrixCSR to_csr() const;

    void multiply(std::span<const double> x, std::span<double> y, double alpha = 1.0, double beta = 0.0) const;
    void multiply_transpose(std::span<const double> x, std::span<double> y, double alpha = 1.0, double beta = 0.0) const;

    [[nodiscard]] double norm_inf() const noexcept;
    [[nodiscard]] double norm_1() const noexcept;
    [[nodiscard]] double norm_frobenius() const noexcept;
};

struct SparseMatrixCSR {
    int m = 0;                          // Rows
    int n = 0;                          // Columns
    std::vector<int64_t> row_ptr;       // Size m + 1
    std::vector<int> col_idx;           // Size nnz
    std::vector<double> values;         // Size nnz

    SparseMatrixCSR() = default;
    SparseMatrixCSR(int rows, int cols);

    [[nodiscard]] int64_t nnz() const noexcept { 
        return static_cast<int64_t>(values.size()); 
    }

    [[nodiscard]] double coeff(int r, int c) const;

    [[nodiscard]] std::span<const int> row_cols(int r) const {
        if (r < 0 || r >= m || row_ptr.empty()) return {};
        const size_t start = static_cast<size_t>(row_ptr[r]);
        const size_t count = static_cast<size_t>(row_ptr[r + 1] - row_ptr[r]);
        return {&col_idx[start], count};
    }

    [[nodiscard]] std::span<const double> row_vals(int r) const {
        if (r < 0 || r >= m || row_ptr.empty()) return {};
        const size_t start = static_cast<size_t>(row_ptr[r]);
        const size_t count = static_cast<size_t>(row_ptr[r + 1] - row_ptr[r]);
        return {&values[start], count};
    }

    void clear() noexcept;
    void resize(int rows, int cols);
    void set_from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates = true);
    static SparseMatrixCSR from_triplets(int rows, int cols, const std::vector<Triplet>& triplets, bool sum_duplicates = true);

    void transpose_into(SparseMatrixCSC& out) const;
    [[nodiscard]] SparseMatrixCSR transpose() const;
    [[nodiscard]] SparseMatrixCSC to_csc() const;

    void multiply(std::span<const double> x, std::span<double> y, double alpha = 1.0, double beta = 0.0) const;
    void multiply_transpose(std::span<const double> x, std::span<double> y, double alpha = 1.0, double beta = 0.0) const;

    [[nodiscard]] double norm_inf() const noexcept;
    [[nodiscard]] double norm_1() const noexcept;
    [[nodiscard]] double norm_frobenius() const noexcept;
};

} // namespace indus::la
