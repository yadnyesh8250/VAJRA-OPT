# CORE C++ HEADER INTERFACES: INDUS-OPT

This document provides the authoritative, exact C++20 header definitions for all foundational data structures in **INDUS-OPT**. Any developer or AI engine implementing the solver must use these exact struct and method signatures to ensure all 40+ implementation files compile and link cleanly without type mismatches.

---

## 1. Types & Tolerances (`include/indus/types.hpp` & `tolerances.hpp`)

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <limits>

namespace indus {

using Index = int;
using BigIndex = int64_t;
using Float = double;

inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

enum class ObjSense : uint8_t { kMinimize = 0, kMaximize = 1 };
enum class VarType : uint8_t { kContinuous = 0, kInteger = 1 };

enum class BasisStatus : uint8_t {
    kUnknown = 0,
    kBasic = 1,
    kAtLower = 2,
    kAtUpper = 3,
    kNonbasicFree = 4,
    kFixed = 5
};

enum class SolveStatus : uint8_t {
    kNotSolved = 0,
    kOptimal = 1,
    kInfeasible = 2,
    kUnbounded = 3,
    kInfeasibleOrUnbounded = 4,
    kFeasible = 5,
    kIterationLimit = 6,
    kTimeLimit = 7,
    kNodeLimit = 8,
    kNumericalError = 9,
    kModelError = 10
};

[[nodiscard]] inline constexpr bool claims_a_point(SolveStatus status) noexcept {
    switch (status) {
        case SolveStatus::kOptimal:
        case SolveStatus::kFeasible:
        case SolveStatus::kUnbounded:
        case SolveStatus::kIterationLimit:
        case SolveStatus::kTimeLimit:
        case SolveStatus::kNodeLimit: return true;
        default: return false;
    }
}

const char* to_string(SolveStatus status) noexcept;
const char* to_string(BasisStatus status) noexcept;
const char* to_string(VarType type) noexcept;

namespace tol {
    inline constexpr double kPrimalFeasibility = 1e-7;
    inline constexpr double kDualFeasibility = 1e-7;
    inline constexpr double kIntegrality = 1e-6;
    inline constexpr double kMipRelativeGap = 1e-4;
    inline constexpr double kMipAbsoluteGap = 1e-6;
    inline constexpr double kMarkowitzThreshold = 0.01;
    inline constexpr double kZeroDrop = 1e-11;
    inline constexpr int kPseudocostReliability = 8;
    inline constexpr int kStrongBranchingCandidates = 10;
    inline constexpr int kStrongBranchingIterations = 50;
}

} // namespace indus
```

---

## 2. Sparse Linear Algebra Views (`include/indus/sparse.hpp`)

```cpp
#pragma once
#include <vector>
#include <cstdint>
#include <span>
#include "indus/types.hpp"

namespace indus::la {

struct SparseMatrixCSC {
    int m = 0;                          // Rows
    int n = 0;                          // Columns
    std::vector<int64_t> col_ptr;       // Size n + 1
    std::vector<int> row_idx;           // Size nnz
    std::vector<double> values;         // Size nnz

    [[nodiscard]] int64_t nnz() const noexcept { return values.size(); }
    [[nodiscard]] double coeff(int r, int c) const;
    [[nodiscard]] std::span<const int> col_rows(int c) const {
        return {&row_idx[col_ptr[c]], static_cast<size_t>(col_ptr[c + 1] - col_ptr[c])};
    }
    [[nodiscard]] std::span<const double> col_vals(int c) const {
        return {&values[col_ptr[c]], static_cast<size_t>(col_ptr[c + 1] - col_ptr[c])};
    }
};

struct SparseMatrixCSR {
    int m = 0;
    int n = 0;
    std::vector<int64_t> row_ptr;       // Size m + 1
    std::vector<int> col_idx;           // Size nnz
    std::vector<double> values;         // Size nnz
    [[nodiscard]] int64_t nnz() const noexcept { return values.size(); }
};

} // namespace indus::la
```

---

## 3. The Core Model & Solution Specifications (`include/indus/model.hpp`)

```cpp
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

    // Linear Constraint Matrix A in CSC format
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
    int64_t iterations = 0;
    int64_t nodes = 0;
    double solve_time_seconds = 0.0;

    SolutionQuality quality;
    void recompute_quality(const Model& original_model);
};

// Master Dispatcher Seam
Solution solve(const Model& model, const Options& options);

} // namespace indus
```

---

## 4. Options Registry (`include/indus/options.hpp`)

```cpp
#pragma once
#include <string>
#include <unordered_map>
#include <variant>

namespace indus {

class Options {
public:
    Options();

    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, double value);
    void set(const std::string& key, int value);
    void set(const std::string& key, bool value);

    [[nodiscard]] double get_double(const std::string& key) const;
    [[nodiscard]] int get_int(const std::string& key) const;
    [[nodiscard]] bool get_bool(const std::string& key) const;
    [[nodiscard]] std::string get_string(const std::string& key) const;

    // Built-in shortcut accessors
    double time_limit = 1e20;
    int64_t iteration_limit = 1000000;
    int64_t node_limit = 500000;
    double mip_relative_gap = 1e-4;
    double mip_absolute_gap = 1e-6;
    bool enable_presolve = true;
    bool enable_scaling = true;
    bool enable_root_cuts = false;
    bool use_gpu = false;
    std::string algorithm = "auto"; // "auto", "simplex", "dual_simplex", "pdhg", "ipm"
};

} // namespace indus
```
