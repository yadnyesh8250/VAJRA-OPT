#pragma once

#include "indus/model.hpp"
#include "indus/options.hpp"
#include <cstdint>
#include <string>

namespace indus::milp {

struct MilpStatistics {
    int64_t nodes_explored = 0;
    int64_t nodes_pruned_bound = 0;
    int64_t nodes_pruned_infeasible = 0;
    int64_t nodes_pruned_integral = 0;
    int64_t open_nodes_remaining = 0;
    double root_lp_objective = 0.0;
    double best_bound = 0.0;
    double incumbent_objective = 0.0;
    double absolute_gap = 0.0;
    double relative_gap = 0.0;
    bool has_incumbent = false;
    double time_taken_seconds = 0.0;
};

// Main Branch-and-Bound Native MILP Solver
Solution solve_milp(const Model& model, const Options& options = Options());

// Retrieve telemetry from the most recent MILP solve
MilpStatistics get_last_milp_statistics();

} // namespace indus::milp
