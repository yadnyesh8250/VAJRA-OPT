#include "src/linalg/scaling.hpp"
#include <algorithm>
#include <cmath>

namespace indus::la {

ScalingFactors RuizScaler::compute_and_scale(
    SparseMatrixCSC& A,
    std::vector<double>& c,
    std::vector<double>& row_lower,
    std::vector<double>& row_upper,
    std::vector<double>& col_lower,
    std::vector<double>& col_upper,
    int max_iters,
    double tolerance
) {
    ScalingFactors sf;
    const int m = A.m;
    const int n = A.n;
    if (m <= 0 || n <= 0) {
        return sf;
    }

    sf.row_scale.assign(static_cast<size_t>(m), 1.0);
    sf.col_scale.assign(static_cast<size_t>(n), 1.0);
    sf.inv_row_scale.assign(static_cast<size_t>(m), 1.0);
    sf.inv_col_scale.assign(static_cast<size_t>(n), 1.0);

    std::vector<double> r_k(static_cast<size_t>(m), 1.0);
    std::vector<double> c_k(static_cast<size_t>(n), 1.0);

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. Calculate row inf-norms
        std::vector<double> row_inf(static_cast<size_t>(m), 0.0);
        for (int j = 0; j < n; ++j) {
            const auto rows = A.col_rows(j);
            const auto vals = A.col_vals(j);
            const size_t sz = rows.size();
            for (size_t k = 0; k < sz; ++k) {
                const int r = rows[k];
                row_inf[static_cast<size_t>(r)] = std::max(row_inf[static_cast<size_t>(r)], std::abs(vals[k]));
            }
        }

        // 2. Calculate col inf-norms
        std::vector<double> col_inf(static_cast<size_t>(n), 0.0);
        for (int j = 0; j < n; ++j) {
            const auto vals = A.col_vals(j);
            for (double v : vals) {
                col_inf[static_cast<size_t>(j)] = std::max(col_inf[static_cast<size_t>(j)], std::abs(v));
            }
        }

        // Check convergence
        double max_row_err = 0.0;
        for (int i = 0; i < m; ++i) {
            if (row_inf[static_cast<size_t>(i)] > tol::kZeroDrop) {
                max_row_err = std::max(max_row_err, std::abs(1.0 - row_inf[static_cast<size_t>(i)]));
            }
        }
        double max_col_err = 0.0;
        for (int j = 0; j < n; ++j) {
            if (col_inf[static_cast<size_t>(j)] > tol::kZeroDrop) {
                max_col_err = std::max(max_col_err, std::abs(1.0 - col_inf[static_cast<size_t>(j)]));
            }
        }

        if (max_row_err <= tolerance && max_col_err <= tolerance) {
            break;
        }

        // Compute step scales: r_k = 1 / sqrt(row_inf), c_k = 1 / sqrt(col_inf)
        for (int i = 0; i < m; ++i) {
            if (row_inf[static_cast<size_t>(i)] > tol::kZeroDrop) {
                r_k[static_cast<size_t>(i)] = 1.0 / std::sqrt(row_inf[static_cast<size_t>(i)]);
            } else {
                r_k[static_cast<size_t>(i)] = 1.0;
            }
            sf.row_scale[static_cast<size_t>(i)] *= r_k[static_cast<size_t>(i)];
        }

        for (int j = 0; j < n; ++j) {
            if (col_inf[static_cast<size_t>(j)] > tol::kZeroDrop) {
                c_k[static_cast<size_t>(j)] = 1.0 / std::sqrt(col_inf[static_cast<size_t>(j)]);
            } else {
                c_k[static_cast<size_t>(j)] = 1.0;
            }
            sf.col_scale[static_cast<size_t>(j)] *= c_k[static_cast<size_t>(j)];
        }

        // Scale A in place: A_{ij} = r_k[i] * A_{ij} * c_k[j]
        for (int j = 0; j < n; ++j) {
            const double c_scale = c_k[static_cast<size_t>(j)];
            const int64_t start = A.col_ptr[static_cast<size_t>(j)];
            const int64_t end = A.col_ptr[static_cast<size_t>(j + 1)];
            for (int64_t k = start; k < end; ++k) {
                const int r = A.row_idx[static_cast<size_t>(k)];
                A.values[static_cast<size_t>(k)] *= r_k[static_cast<size_t>(r)] * c_scale;
            }
        }
    }

    // Compute inverses of cumulative scales
    for (int i = 0; i < m; ++i) {
        sf.inv_row_scale[static_cast<size_t>(i)] = 1.0 / sf.row_scale[static_cast<size_t>(i)];
    }
    for (int j = 0; j < n; ++j) {
        sf.inv_col_scale[static_cast<size_t>(j)] = 1.0 / sf.col_scale[static_cast<size_t>(j)];
    }

    // Scale vectors:
    // Objective: c_scaled = C * c
    if (c.size() == static_cast<size_t>(n)) {
        for (int j = 0; j < n; ++j) {
            c[static_cast<size_t>(j)] *= sf.col_scale[static_cast<size_t>(j)];
        }
    }

    // Row bounds: scaled_lower = R * row_lower, scaled_upper = R * row_upper
    if (row_lower.size() == static_cast<size_t>(m)) {
        for (int i = 0; i < m; ++i) {
            if (row_lower[static_cast<size_t>(i)] > -1e20) {
                row_lower[static_cast<size_t>(i)] *= sf.row_scale[static_cast<size_t>(i)];
            }
        }
    }
    if (row_upper.size() == static_cast<size_t>(m)) {
        for (int i = 0; i < m; ++i) {
            if (row_upper[static_cast<size_t>(i)] < 1e20) {
                row_upper[static_cast<size_t>(i)] *= sf.row_scale[static_cast<size_t>(i)];
            }
        }
    }

    // Col bounds: scaled_col_lower = C⁻¹ * col_lower, scaled_col_upper = C⁻¹ * col_upper
    if (col_lower.size() == static_cast<size_t>(n)) {
        for (int j = 0; j < n; ++j) {
            if (col_lower[static_cast<size_t>(j)] > -1e20) {
                col_lower[static_cast<size_t>(j)] *= sf.inv_col_scale[static_cast<size_t>(j)];
            }
        }
    }
    if (col_upper.size() == static_cast<size_t>(n)) {
        for (int j = 0; j < n; ++j) {
            if (col_upper[static_cast<size_t>(j)] < 1e20) {
                col_upper[static_cast<size_t>(j)] *= sf.inv_col_scale[static_cast<size_t>(j)];
            }
        }
    }

    sf.is_scaled = true;
    return sf;
}

void RuizScaler::unscale_primal(const ScalingFactors& sf, std::span<double> x) {
    if (!sf.is_scaled) return;
    const size_t sz = std::min(x.size(), sf.col_scale.size());
    for (size_t j = 0; j < sz; ++j) {
        x[j] *= sf.col_scale[j];
    }
}

void RuizScaler::unscale_dual(const ScalingFactors& sf, std::span<double> y) {
    if (!sf.is_scaled) return;
    const size_t sz = std::min(y.size(), sf.row_scale.size());
    for (size_t i = 0; i < sz; ++i) {
        y[i] *= sf.row_scale[i];
    }
}

void RuizScaler::unscale_reduced_costs(const ScalingFactors& sf, std::span<double> d) {
    if (!sf.is_scaled) return;
    const size_t sz = std::min(d.size(), sf.inv_col_scale.size());
    for (size_t j = 0; j < sz; ++j) {
        d[j] *= sf.inv_col_scale[j];
    }
}

void PockChambollePreconditioner::compute(
    const SparseMatrixCSC& A,
    std::vector<double>& d_row,
    std::vector<double>& d_col,
    double alpha
) {
    const int m = A.m;
    const int n = A.n;
    d_row.assign(static_cast<size_t>(m), 0.0);
    d_col.assign(static_cast<size_t>(n), 0.0);

    const double exp_row = 2.0 - alpha;
    const double exp_col = alpha;

    for (int j = 0; j < n; ++j) {
        const auto rows = A.col_rows(j);
        const auto vals = A.col_vals(j);
        const size_t sz = rows.size();
        for (size_t k = 0; k < sz; ++k) {
            const int r = rows[k];
            const double abs_val = std::abs(vals[k]);
            d_row[static_cast<size_t>(r)] += std::pow(abs_val, exp_row);
            d_col[static_cast<size_t>(j)] += std::pow(abs_val, exp_col);
        }
    }

    for (int i = 0; i < m; ++i) {
        if (d_row[static_cast<size_t>(i)] > tol::kZeroDrop) {
            d_row[static_cast<size_t>(i)] = 1.0 / d_row[static_cast<size_t>(i)];
        } else {
            d_row[static_cast<size_t>(i)] = 1.0;
        }
    }

    for (int j = 0; j < n; ++j) {
        if (d_col[static_cast<size_t>(j)] > tol::kZeroDrop) {
            d_col[static_cast<size_t>(j)] = 1.0 / d_col[static_cast<size_t>(j)];
        } else {
            d_col[static_cast<size_t>(j)] = 1.0;
        }
    }
}

} // namespace indus::la
