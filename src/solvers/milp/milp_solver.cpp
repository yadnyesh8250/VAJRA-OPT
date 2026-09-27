#include "indus/milp.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

namespace indus::milp {

namespace {

MilpStatistics g_last_milp_stats;

struct Node {
    int64_t id = 0;
    int depth = 0;
    std::vector<double> col_lower;
    std::vector<double> col_upper;
    double relaxation_bound = 0.0; // Lower bound (min) or Upper bound (max)
};

// Priority comparator for minimization: min-heap on relaxation_bound
struct CompareMin {
    bool operator()(const Node& a, const Node& b) const {
        if (std::abs(a.relaxation_bound - b.relaxation_bound) > 1e-9) {
            return a.relaxation_bound > b.relaxation_bound; // smaller bound has higher priority
        }
        return a.id > b.id; // deterministic tie-breaker
    }
};

// Priority comparator for maximization: max-heap on relaxation_bound
struct CompareMax {
    bool operator()(const Node& a, const Node& b) const {
        if (std::abs(a.relaxation_bound - b.relaxation_bound) > 1e-9) {
            return a.relaxation_bound < b.relaxation_bound; // larger bound has higher priority
        }
        return a.id > b.id; // deterministic tie-breaker
    }
};

// Check integer feasibility and find the most fractional variable
bool check_integrality(const Model& model, const std::vector<double>& x, double int_tol,
                      int& most_fractional_col, double& max_frac_distance) {
    most_fractional_col = -1;
    max_frac_distance = -1.0;
    bool all_integer = true;

    for (int j = 0; j < model.num_cols; ++j) {
        if (model.col_type.size() > static_cast<size_t>(j) &&
            model.col_type[static_cast<size_t>(j)] == VarType::kInteger) {
            const double val = x[static_cast<size_t>(j)];
            const double nearest = std::round(val);
            const double dist_to_nearest = std::abs(val - nearest);

            if (dist_to_nearest > int_tol) {
                all_integer = false;
                // Distance to integer bounds (fractional distance in [0, 0.5])
                const double dist = std::min(val - std::floor(val), std::ceil(val) - val);
                if (dist > max_frac_distance + 1e-9) {
                    max_frac_distance = dist;
                    most_fractional_col = j;
                }
            }
        }
    }
    return all_integer;
}

// Solve LP relaxation of a node using clean, unpresolved dual simplex (per Corrections #2 & #3)
Solution solve_relaxation(const Model& base_model,
                          const std::vector<double>& col_lower,
                          const std::vector<double>& col_upper,
                          const Options& global_options,
                          double remaining_seconds) {
    Model node_model = base_model;
    node_model.col_lower = col_lower;
    node_model.col_upper = col_upper;
    // Mark all variables continuous for node relaxation
    node_model.col_type.assign(static_cast<size_t>(base_model.num_cols), VarType::kContinuous);

    Options lp_opts;
    lp_opts.algorithm = "dual_simplex";
    lp_opts.enable_presolve = false; // per Correction #2: disable presolve for MILP nodes
    lp_opts.enable_scaling = global_options.enable_scaling;
    lp_opts.time_limit = remaining_seconds;
    lp_opts.iteration_limit = global_options.iteration_limit;
    lp_opts.set("tolerance", global_options.get_double("tolerance", 1e-6));

    return indus::solve(node_model, lp_opts);
}

} // namespace

Solution solve_milp(const Model& model, const Options& options) {
    const auto start_time = std::chrono::high_resolution_clock::now();
    try {
        model.validate();
    } catch (const std::exception& e) {
        Solution sol;
        sol.status = SolveStatus::kModelError;
        sol.status_message = std::string("Model validation error: ") + e.what();
        sol.solve_time_seconds = 0.0;
        return sol;
    }

    MilpStatistics stats;
    const bool is_max = (model.sense == ObjSense::kMaximize);
    const double int_tol = options.integer_tolerance;
    const double rel_gap_tol = options.mip_relative_gap;
    const double abs_gap_tol = options.mip_absolute_gap;
    const int64_t node_limit = options.node_limit;
    const double time_limit = options.time_limit;

    // Validate integer variables and binary bounds
    for (int j = 0; j < model.num_cols; ++j) {
        if (model.col_type.size() > static_cast<size_t>(j) &&
            model.col_type[static_cast<size_t>(j)] == VarType::kInteger) {
            const double lj = model.col_lower[static_cast<size_t>(j)];
            const double uj = model.col_upper[static_cast<size_t>(j)];
            if (lj > uj) {
                Solution sol;
                sol.status = SolveStatus::kModelError;
                sol.status_message = "Variable " + (model.col_names.empty() ? std::to_string(j) : model.col_names[static_cast<size_t>(j)]) +
                                     " has lower bound > upper bound";
                return sol;
            }
        }
    }

    // Solve Root LP Relaxation
    const auto time_elapsed = [&]() -> double {
        return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    };

    Solution root_sol = solve_relaxation(model, model.col_lower, model.col_upper, options, time_limit);
    stats.nodes_explored = 1;
    stats.root_lp_objective = root_sol.objective_value;

    if (root_sol.status == SolveStatus::kInfeasible) {
        Solution sol;
        sol.status = SolveStatus::kInfeasible;
        sol.status_message = "Root LP relaxation is infeasible";
        sol.solve_time_seconds = time_elapsed();
        sol.nodes = 1;
        sol.open_nodes = 0;
        sol.search_completed = true;
        sol.has_incumbent = false;
        sol.termination_reason = sol.status_message;
        g_last_milp_stats = stats;
        return sol;
    }

    if (root_sol.status == SolveStatus::kUnbounded) {
        Solution sol;
        sol.status = SolveStatus::kUnbounded;
        sol.status_message = "Root LP relaxation is unbounded";
        sol.solve_time_seconds = time_elapsed();
        sol.nodes = 1;
        sol.open_nodes = 0;
        sol.search_completed = true;
        sol.has_incumbent = false;
        sol.termination_reason = sol.status_message;
        g_last_milp_stats = stats;
        return sol;
    }

    // Requirement 1: Only an OPTIMAL LP relaxation may be used as a branch-and-bound bound.
    // If a node relaxation returns FEASIBLE, ITERATION_LIMIT, TIME_LIMIT, or NUMERICAL_ERROR:
    // terminate the MILP with an honest unverified/limit status.
    if (root_sol.status != SolveStatus::kOptimal || !root_sol.quality.is_primal_feasible) {
        Solution sol;
        sol.status = (root_sol.status == SolveStatus::kOptimal) ? SolveStatus::kNumericalError : root_sol.status;
        sol.status_message = "Root LP relaxation failed: " + std::string(to_string(root_sol.status)) + " (" + root_sol.status_message + ")";
        sol.solve_time_seconds = time_elapsed();
        sol.nodes = 1;
        sol.open_nodes = 1;
        sol.search_completed = false;
        sol.has_incumbent = false;
        sol.termination_reason = sol.status_message;
        stats.open_nodes_remaining = 1;
        g_last_milp_stats = stats;
        return sol;
    }

    // Check root integrality
    int most_frac_var = -1;
    double max_frac_dist = 0.0;
    const bool root_is_integral = check_integrality(model, root_sol.col_value, int_tol, most_frac_var, max_frac_dist);

    if (root_is_integral) {
        // Solved at root node!
        Solution sol = root_sol;
        sol.status = SolveStatus::kOptimal;
        sol.status_message = "Solved to optimality at root node";
        sol.best_dual_bound = root_sol.objective_value;
        sol.relative_gap = 0.0;
        sol.absolute_gap = 0.0;
        sol.nodes = 1;
        sol.open_nodes = 0;
        sol.pruned_nodes = 0;
        sol.has_incumbent = true;
        sol.search_completed = true;
        sol.solve_time_seconds = time_elapsed();
        sol.algorithm_used = "milp_branch_and_bound";
        sol.recompute_quality(model);

        stats.best_bound = root_sol.objective_value;
        stats.incumbent_objective = root_sol.objective_value;
        stats.has_incumbent = true;
        stats.time_taken_seconds = sol.solve_time_seconds;
        g_last_milp_stats = stats;
        return sol;
    }

    // Initialize Branch-and-Bound structures
    Solution incumbent;
    bool has_incumbent = false;
    double incumbent_obj = is_max ? -1e30 : 1e30;

    int64_t next_node_id = 1;
    Node root_node;
    root_node.id = next_node_id++;
    root_node.depth = 0;
    root_node.col_lower = model.col_lower;
    root_node.col_upper = model.col_upper;
    root_node.relaxation_bound = root_sol.objective_value;

    std::priority_queue<Node, std::vector<Node>, CompareMin> open_min;
    std::priority_queue<Node, std::vector<Node>, CompareMax> open_max;

    if (is_max) {
        open_max.push(root_node);
    } else {
        open_min.push(root_node);
    }

    SolveStatus final_status = SolveStatus::kOptimal;
    std::string term_reason = "Branch-and-bound search completed";

    // Branch-and-Bound Loop
    while (is_max ? !open_max.empty() : !open_min.empty()) {
        const double elapsed = time_elapsed();
        if (elapsed >= time_limit) {
            final_status = SolveStatus::kTimeLimit;
            term_reason = "Time limit exceeded";
            break;
        }

        if (stats.nodes_explored >= node_limit) {
            final_status = SolveStatus::kNodeLimit;
            term_reason = "Node limit reached (" + std::to_string(node_limit) + " nodes)";
            break;
        }

        // Pop best-bound node
        Node current;
        if (is_max) {
            current = open_max.top();
            open_max.pop();
        } else {
            current = open_min.top();
            open_min.pop();
        }

        // Bound pruning check
        if (has_incumbent) {
            if (!is_max && current.relaxation_bound >= incumbent_obj - abs_gap_tol) {
                stats.nodes_pruned_bound++;
                continue;
            }
            if (is_max && current.relaxation_bound <= incumbent_obj + abs_gap_tol) {
                stats.nodes_pruned_bound++;
                continue;
            }

            // Check current global MIP gap
            const double current_best = current.relaxation_bound;
            const double abs_gap = is_max ? std::max(0.0, current_best - incumbent_obj)
                                          : std::max(0.0, incumbent_obj - current_best);
            const double rel_gap = abs_gap / std::max(1.0, std::abs(incumbent_obj));

            if (rel_gap <= rel_gap_tol || abs_gap <= abs_gap_tol) {
                final_status = SolveStatus::kOptimal;
                term_reason = "Relative MIP gap tolerance reached (" + std::to_string(rel_gap) + " <= " + std::to_string(rel_gap_tol) + ")";
                break;
            }
        }

        // Solve LP relaxation at this node (unless it is root, which was already solved)
        Solution node_sol;
        if (current.id == 1) {
            node_sol = root_sol;
        } else {
            const double rem_time = std::max(0.001, time_limit - elapsed);
            node_sol = solve_relaxation(model, current.col_lower, current.col_upper, options, rem_time);
            stats.nodes_explored++;
        }

        // 1. Infeasibility Pruning
        if (node_sol.status == SolveStatus::kInfeasible) {
            stats.nodes_pruned_infeasible++;
            continue;
        }

        // 2. LP Relaxation Status Safety (Requirement 1)
        // Only an OPTIMAL LP relaxation may be used as a branch-and-bound bound.
        // If a node relaxation returns FEASIBLE, ITERATION_LIMIT, TIME_LIMIT, or NUMERICAL_ERROR:
        // - do not prune the node,
        // - do not update the global bound from that result,
        // - terminate the MILP with an honest unverified/limit status.
        if (node_sol.status != SolveStatus::kOptimal || !node_sol.quality.is_primal_feasible) {
            if (is_max) {
                open_max.push(current);
            } else {
                open_min.push(current);
            }
            final_status = (node_sol.status == SolveStatus::kOptimal) ? SolveStatus::kNumericalError : node_sol.status;
            term_reason = "LP relaxation at node " + std::to_string(current.id) + 
                          " returned non-optimal status: " + std::string(to_string(node_sol.status)) +
                          " (" + node_sol.status_message + ")";
            break;
        }

        // 3. Bound Pruning
        const double node_obj = node_sol.objective_value;
        if (has_incumbent) {
            if (!is_max && node_obj >= incumbent_obj - abs_gap_tol) {
                stats.nodes_pruned_bound++;
                continue;
            }
            if (is_max && node_obj <= incumbent_obj + abs_gap_tol) {
                stats.nodes_pruned_bound++;
                continue;
            }
        }

        // 4. Integrality Check
        int branch_col = -1;
        double frac_dist = 0.0;
        const bool is_int = check_integrality(model, node_sol.col_value, int_tol, branch_col, frac_dist);

        if (is_int) {
            // Integer feasible solution found!
            stats.nodes_pruned_integral++;
            bool is_better = false;
            if (!has_incumbent) {
                is_better = true;
            } else if (!is_max && node_obj < incumbent_obj - 1e-9) {
                is_better = true;
            } else if (is_max && node_obj > incumbent_obj + 1e-9) {
                is_better = true;
            }

            if (is_better) {
                incumbent = node_sol;
                has_incumbent = true;
                incumbent_obj = node_obj;
            }
            continue; // Pruned by integrality
        }

        // 5. Branching on Most Fractional Variable
        if (branch_col < 0) {
            continue;
        }

        const double val = node_sol.col_value[static_cast<size_t>(branch_col)];
        const double floor_val = std::floor(val);
        const double ceil_val = std::ceil(val);

        // Left Child: x_j <= floor(val)
        if (floor_val >= current.col_lower[static_cast<size_t>(branch_col)]) {
            Node left_child;
            left_child.id = next_node_id++;
            left_child.depth = current.depth + 1;
            left_child.col_lower = current.col_lower;
            left_child.col_upper = current.col_upper;
            left_child.col_upper[static_cast<size_t>(branch_col)] = floor_val;
            left_child.relaxation_bound = node_obj;

            if (is_max) {
                open_max.push(left_child);
            } else {
                open_min.push(left_child);
            }
        }

        // Right Child: x_j >= ceil(val)
        if (ceil_val <= current.col_upper[static_cast<size_t>(branch_col)]) {
            Node right_child;
            right_child.id = next_node_id++;
            right_child.depth = current.depth + 1;
            right_child.col_lower = current.col_lower;
            right_child.col_upper = current.col_upper;
            right_child.col_lower[static_cast<size_t>(branch_col)] = ceil_val;
            right_child.relaxation_bound = node_obj;

            if (is_max) {
                open_max.push(right_child);
            } else {
                open_min.push(right_child);
            }
        }
    }

    // Determine Best Bound
    const size_t open_count = is_max ? open_max.size() : open_min.size();
    double best_bound = incumbent_obj;
    if (open_count > 0) {
        best_bound = is_max ? open_max.top().relaxation_bound : open_min.top().relaxation_bound;
    } else if (has_incumbent) {
        best_bound = incumbent_obj; // Tree fully pruned -> best bound equals incumbent
    } else {
        best_bound = is_max ? -1e30 : 1e30;
    }

    // Calculate final MIP gaps
    double abs_gap = 0.0;
    double rel_gap = 0.0;
    if (has_incumbent && !std::isinf(best_bound) && !std::isnan(best_bound)) {
        abs_gap = is_max ? std::max(0.0, best_bound - incumbent_obj)
                         : std::max(0.0, incumbent_obj - best_bound);
        rel_gap = abs_gap / std::max(1.0, std::abs(incumbent_obj));
    }

    const bool tree_exhausted = (open_count == 0);
    const bool gap_closed = (has_incumbent && (rel_gap <= rel_gap_tol || abs_gap <= abs_gap_tol));

    stats.open_nodes_remaining = static_cast<int64_t>(open_count);
    stats.best_bound = best_bound;
    stats.incumbent_objective = incumbent_obj;
    stats.has_incumbent = has_incumbent;
    stats.absolute_gap = abs_gap;
    stats.relative_gap = rel_gap;
    stats.time_taken_seconds = time_elapsed();
    g_last_milp_stats = stats;

    // Assemble Return Solution
    Solution sol;
    sol.algorithm_used = "milp_branch_and_bound";
    sol.nodes = stats.nodes_explored;
    sol.open_nodes = stats.open_nodes_remaining;
    sol.pruned_nodes = stats.nodes_pruned_bound + stats.nodes_pruned_infeasible + stats.nodes_pruned_integral;
    sol.solve_time_seconds = stats.time_taken_seconds;
    sol.best_dual_bound = best_bound;
    sol.absolute_gap = abs_gap;
    sol.relative_gap = rel_gap;
    sol.has_incumbent = has_incumbent;
    sol.search_completed = (final_status == SolveStatus::kOptimal && (tree_exhausted || gap_closed));
    sol.termination_reason = term_reason;

    if (!has_incumbent) {
        if (final_status == SolveStatus::kTimeLimit || 
            final_status == SolveStatus::kNodeLimit || 
            final_status == SolveStatus::kIterationLimit ||
            final_status == SolveStatus::kNumericalError) {
            sol.status = final_status;
            sol.status_message = term_reason + " without finding any integer feasible point";
            sol.search_completed = false;
        } else {
            sol.status = SolveStatus::kInfeasible;
            sol.status_message = "Branch-and-bound proved integer infeasibility";
            sol.search_completed = true;
        }
        return sol;
    }

    // Incumbent exists
    sol.col_value = incumbent.col_value;
    sol.row_value = incumbent.row_value;
    sol.objective_value = incumbent_obj;

    if (final_status == SolveStatus::kTimeLimit || 
        final_status == SolveStatus::kNodeLimit || 
        final_status == SolveStatus::kIterationLimit ||
        final_status == SolveStatus::kNumericalError) {
        sol.status = final_status;
        sol.status_message = term_reason + "; returning best integer feasible incumbent";
        sol.search_completed = false;
    } else {
        sol.status = SolveStatus::kOptimal;
        sol.status_message = "Solved to optimality (MIP gap: " + std::to_string(rel_gap) + ")";
        sol.search_completed = true;
    }

    sol.recompute_quality(model);

    // MILP Status Guard
    if (sol.status == SolveStatus::kOptimal) {
        if (!sol.quality.is_primal_feasible || !sol.quality.is_integer_feasible) {
            sol.status = SolveStatus::kNumericalError;
            sol.status_message = "Status downgraded to NUMERICAL_ERROR: integrality or primal violation exceeds tolerance";
            sol.search_completed = false;
        }
    }
    return sol;
}

MilpStatistics get_last_milp_statistics() {
    return g_last_milp_stats;
}

} // namespace indus::milp
