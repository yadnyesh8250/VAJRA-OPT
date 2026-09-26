#pragma once

#include <vector>
#include <string>
#include <chrono>
#include "indus/types.hpp"
#include "indus/model.hpp"
#include "indus/options.hpp"
#include "indus/sparse.hpp"

namespace indus::pdhg {

using indus::Model;
using indus::Options;
using indus::Solution;

struct PdhgOptions {
    int64_t max_iterations = 50000;
    double time_limit = 1e20;
    double primal_tolerance = 1e-6;
    double dual_tolerance = 1e-6;
    double gap_tolerance = 1e-6;
    int check_frequency = 100;
    int restart_stagnation_limit = 200;
    double step_size_safety = 0.95;
    bool enable_scaling = true;
    bool verbose = false;
};

struct PdhgDiagnostics {
    int64_t iterations = 0;
    int restart_count = 0;
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    double duality_gap = 0.0;
    double spectral_norm_estimate = 0.0;
    double tau = 0.0;
    double sigma = 0.0;
    double setup_time_sec = 0.0;
    double iteration_time_sec = 0.0;
    double total_time_sec = 0.0;
    bool converged = false;
};

// Power iteration to compute spectral norm ||A||_2 of sparse matrix
double estimate_spectral_norm(const la::SparseMatrixCSC& A, int max_iter = 25, double tol = 1e-4);

// CPU Restarted PDHG Solver for continuous bounded LPs
class CpuPdhgSolver {
public:
    explicit CpuPdhgSolver(const Model& model, const Options& options = Options());

    Solution solve();
    [[nodiscard]] const PdhgDiagnostics& diagnostics() const noexcept { return diag_; }

private:
    const Model& original_model_;
    Options options_;
    PdhgOptions pdhg_opts_;
    PdhgDiagnostics diag_;
};

// Convenience entry point for CPU PDHG
Solution solve_pdhg_cpu(const Model& model, const Options& options = Options());

// Retrieve telemetry from the most recent CPU PDHG solve
[[nodiscard]] const PdhgDiagnostics& get_last_cpu_pdhg_diagnostics() noexcept;

} // namespace indus::pdhg
