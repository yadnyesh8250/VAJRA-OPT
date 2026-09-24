#pragma once

#include "indus/model.hpp"
#include <vector>
#include <string>
#include <memory>
#include <utility>

namespace indus::presolve {

enum class ReductionType : uint8_t {
    kEmptyRow,
    kEmptyCol,
    kFixedCol,
    kSingletonRow,
    kForcingRow,
    kRedundantRow,
    kFreeColSingleton,
    kDoubletonRow
};

// Represents an individual reduction step pushed onto the reversible LIFO stack
struct ReductionRecord {
    ReductionType type = ReductionType::kEmptyRow;
    int row_idx = -1;
    int col_idx = -1;
    int other_col_idx = -1;
    std::string name;

    // Numerical metadata for postsolve reconstruction
    double coeff = 0.0;
    double other_coeff = 0.0;
    double old_lower = 0.0;
    double old_upper = 0.0;
    double row_lower = 0.0;
    double row_upper = 0.0;
    double fixed_value = 0.0;
    double obj_cost = 0.0;

    // Linear combinations for row/col substitutions:
    // For FixedCol: list of (row_idx, coeff) in original model
    // For FreeColSingleton: list of other columns in the row (col_idx, coeff)
    // For DoubletonRow: list of occurrences in other rows (row_idx, coeff)
    std::vector<std::pair<int, double>> entries;
};

// Reversible LIFO postsolve reduction stack
class PresolveStack {
public:
    void push(ReductionRecord rec) {
        records_.push_back(std::move(rec));
    }

    [[nodiscard]] size_t size() const noexcept { return records_.size(); }
    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

    [[nodiscard]] const std::vector<ReductionRecord>& records() const noexcept { return records_; }
    std::vector<ReductionRecord>& records() noexcept { return records_; }

    // Index mappings between reduced and original models
    std::vector<int> reduced_to_orig_row;
    std::vector<int> reduced_to_orig_col;
    std::vector<int> orig_to_reduced_row;
    std::vector<int> orig_to_reduced_col;

private:
    std::vector<ReductionRecord> records_;
};

// Summary of presolve transformation
struct PresolveResult {
    Model presolved_model;
    PresolveStack stack;
    bool is_infeasible = false;
    bool is_unbounded = false;
    std::string status_message;

    int num_empty_rows = 0;
    int num_empty_cols = 0;
    int num_fixed_cols = 0;
    int num_singleton_rows = 0;
    int num_forcing_rows = 0;
    int num_redundant_rows = 0;
    int num_free_col_singletons = 0;
    int num_doubleton_rows = 0;

    [[nodiscard]] int total_reductions() const noexcept {
        return num_empty_rows + num_empty_cols + num_fixed_cols +
               num_singleton_rows + num_forcing_rows + num_redundant_rows +
               num_free_col_singletons + num_doubleton_rows;
    }
};

class PresolveEngine {
public:
    // Applies reversible presolve reductions to fixed point
    static PresolveResult apply(const Model& original_model, int max_passes = 10);

    // Reconstructs original model solution from reduced model solution
    static Solution postsolve(const Solution& reduced_solution,
                              const PresolveStack& stack,
                              const Model& original_model);

    // Dual fixed-point refinement on original model
    static void refine_dual_solution(const Model& original_model,
                                     Solution& solution,
                                     double tol = 1e-6);
};

} // namespace indus::presolve
