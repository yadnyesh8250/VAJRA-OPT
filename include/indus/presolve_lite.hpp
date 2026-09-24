#pragma once

#include "indus/model.hpp"
#include <vector>
#include <string>

namespace indus::presolve {

struct EmptyRowStep {
    int orig_row_idx = -1;
    std::string row_name;
    double lower = 0.0;
    double upper = 0.0;
};

struct SingletonRowStep {
    int orig_row_idx = -1;
    std::string row_name;
    int col_idx = -1;
    double coeff = 0.0;
    double row_lower = 0.0;
    double row_upper = 0.0;
    double old_col_lb = 0.0;
    double old_col_ub = 0.0;
};

struct FixedColStep {
    int orig_col_idx = -1;
    std::string col_name;
    double fixed_val = 0.0;
    double obj_cost = 0.0;
    std::vector<std::pair<int, double>> entries; // (orig_row_idx, coeff)
};

struct PresolveStack {
    std::vector<EmptyRowStep> empty_rows;
    std::vector<SingletonRowStep> singleton_rows;
    std::vector<FixedColStep> fixed_cols;

    // Mapping from reduced model indices to original model indices
    std::vector<int> reduced_to_orig_row;
    std::vector<int> reduced_to_orig_col;
};

struct PresolveResult {
    Model presolved_model;
    PresolveStack stack;
    bool is_infeasible = false;
    std::string infeasibility_reason;
    int num_rows_removed = 0;
    int num_cols_removed = 0;
};

class PresolveLite {
public:
    static PresolveResult apply(const Model& original_model);
    static Solution postsolve(const Solution& reduced_solution,
                              const PresolveStack& stack,
                              const Model& original_model);
};

} // namespace indus::presolve
