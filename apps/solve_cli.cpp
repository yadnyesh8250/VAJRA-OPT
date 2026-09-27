#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/milp.hpp"
#include "indus/qp.hpp"
#include "indus/options.hpp"
#include "indus/verifier.hpp"
#include "indus/gpu.hpp"
#include "indus/pdhg.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --input <model.mps|model.lp> [options]\n\n"
              << "Options:\n"
              << "  -i, --input <file>        Input model path (.mps or .lp format) [required]\n"
              << "  -o, --output <file>       Output file path (.sol or .json). If no extension,\n"
              << "                            both <file>.sol and <file>.json are created.\n"
              << "      --sol <file>          Export solution in .sol text format\n"
              << "      --json <file>         Export solution and telemetry in JSON format\n"
              << "  -a, --algorithm <algo>    Algorithm: auto, dual_simplex, primal_simplex,\n"
              << "                            simplex, pdhg_cpu, pdhg_cuda, pdhg, milp, qp (default: auto)\n"
              << "      --time-limit <sec>    Time limit in seconds (default: 1e20 / unlimited)\n"
              << "      --iter-limit <N>      Iteration limit (default: 1000000; 50000 for PDHG)\n"
              << "      --node-limit <N>      Node limit for MILP branch-and-bound (default: 500000)\n"
              << "      --mip-gap <val>       Relative MIP gap tolerance (default: 1e-4)\n"
              << "      --abs-gap <val>       Absolute MIP gap tolerance (default: 1e-6)\n"
              << "      --integer-tol <val>   Integrality tolerance (default: 1e-5)\n"
              << "      --tol <val>           Feasibility and optimality tolerance (default: 1e-4)\n"
              << "      --presolve            Enable presolve reductions (default: enabled)\n"
              << "      --no-presolve         Disable presolve reductions\n"
              << "      --scaling             Enable matrix Ruiz equilibration scaling (default)\n"
              << "      --no-scaling          Disable matrix scaling\n"
              << "      --verify              Run independent solution verifier (default: enabled)\n"
              << "      --no-verify           Disable independent solution verification\n"
              << "  -h, --help                Show this help message and exit\n\n"
              << "Examples:\n"
              << "  " << prog << " --input model.mps --output solution.json\n"
              << "  " << prog << " --input model.lp --algorithm dual_simplex\n"
              << "  " << prog << " --input model.mps --algorithm milp --node-limit 1000\n"
              << "  " << prog << " --input model.mps --algorithm pdhg_cpu --time-limit 60\n"
              << "  " << prog << " --input model.mps --algorithm pdhg_cuda --output out.sol\n";
}

std::string to_lower_str(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    std::string input_path;
    std::string output_base;
    std::string sol_out;
    std::string json_out;
    std::string algorithm = "auto";
    double time_limit = 1e20;
    int64_t iter_limit = -1; // -1 means use solver default
    int64_t node_limit = 500000;
    double mip_relative_gap = 1e-4;
    double mip_absolute_gap = 1e-6;
    double integer_tol = 1e-5;
    double tol = 1e-4;
    bool enable_presolve = true;
    bool enable_scaling = true;
    bool verify = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_base = argv[++i];
        } else if (arg == "--sol" && i + 1 < argc) {
            sol_out = argv[++i];
        } else if (arg == "--json" && i + 1 < argc) {
            json_out = argv[++i];
        } else if ((arg == "-a" || arg == "--algorithm") && i + 1 < argc) {
            algorithm = to_lower_str(argv[++i]);
        } else if (arg == "--time-limit" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if ((arg == "--iter-limit" || arg == "--iteration-limit") && i + 1 < argc) {
            iter_limit = std::stoll(argv[++i]);
        } else if (arg == "--node-limit" && i + 1 < argc) {
            node_limit = std::stoll(argv[++i]);
        } else if ((arg == "--mip-gap" || arg == "--relative-gap") && i + 1 < argc) {
            mip_relative_gap = std::stod(argv[++i]);
        } else if ((arg == "--abs-gap" || arg == "--absolute-gap") && i + 1 < argc) {
            mip_absolute_gap = std::stod(argv[++i]);
        } else if (arg == "--integer-tol" && i + 1 < argc) {
            integer_tol = std::stod(argv[++i]);
        } else if ((arg == "--tol" || arg == "--tolerance") && i + 1 < argc) {
            tol = std::stod(argv[++i]);
        } else if (arg == "--presolve") {
            enable_presolve = true;
        } else if (arg == "--no-presolve") {
            enable_presolve = false;
        } else if (arg == "--scaling") {
            enable_scaling = true;
        } else if (arg == "--no-scaling") {
            enable_scaling = false;
        } else if (arg == "--verify") {
            verify = true;
        } else if (arg == "--no-verify") {
            verify = false;
        } else if (arg[0] != '-' && input_path.empty()) {
            input_path = arg;
        } else {
            std::cerr << "[ERROR] Unknown or misplaced argument: " << arg << "\n\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    if (input_path.empty()) {
        std::cerr << "[ERROR] Missing required --input argument.\n\n";
        print_usage(argv[0]);
        return 2;
    }

    if (!std::filesystem::exists(input_path)) {
        std::cerr << "[ERROR] Input model file not found: " << input_path << "\n";
        return 2;
    }

    // Configure outputs
    if (!output_base.empty()) {
        std::string ext = to_lower_str(std::filesystem::path(output_base).extension().string());
        if (ext == ".sol") {
            if (sol_out.empty()) sol_out = output_base;
        } else if (ext == ".json") {
            if (json_out.empty()) json_out = output_base;
        } else {
            if (sol_out.empty()) sol_out = output_base + ".sol";
            if (json_out.empty()) json_out = output_base + ".json";
        }
    }

    std::cout << "========================================================================\n";
    std::cout << "  SIDDHANTA (INDUS-OPT): SOVEREIGN HIGH-PERFORMANCE OPTIMIZATION SOLVER \n";
    std::cout << "  Smart India Hackathon SIH26119 | Indigenous LP Engine                  \n";
    std::cout << "========================================================================\n\n";

    // Hardware inspection
    auto dev_info = indus::gpu::probe_device();
    std::cout << "[HARDWARE CONFIGURATION]\n";
    std::cout << "  CPU Architecture: "
#if defined(__aarch64__) || defined(_M_ARM64)
              << "ARM64 (Apple Silicon / AArch64)\n";
#elif defined(__x86_64__) || defined(_M_X64)
              << "x86_64\n";
#else
              << "Generic Architecture\n";
#endif
    std::cout << "  GPU Hardware    : " << dev_info.device_name << "\n";
    std::cout << "  CUDA Status     : " << (dev_info.available ? "ACTIVE (Hardware Ready)" : "UNAVAILABLE (Pure CPU Fallback Active)") << "\n\n";

    // Load model
    std::cout << "[LOADING MODEL] " << input_path << "\n";
    indus::Model model;
    try {
        std::string ext = to_lower_str(std::filesystem::path(input_path).extension().string());
        if (ext == ".lp") {
            model = indus::io::read_lp(input_path);
        } else if (ext == ".mps") {
            model = indus::io::read_mps(input_path);
        } else {
            // Attempt MPS first, fallback to LP
            try {
                model = indus::io::read_mps(input_path);
            } catch (...) {
                model = indus::io::read_lp(input_path);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to parse model file: " << e.what() << "\n";
        return 2;
    }

    const auto pclass = model.classify();
    const bool is_miqp_problem = (pclass == indus::ProblemClass::kMiqp);
    const bool is_qp_problem = (pclass == indus::ProblemClass::kQp || algorithm == "qp");
    const bool is_milp_problem = (pclass == indus::ProblemClass::kMilp || (algorithm == "milp" && !is_qp_problem));

    if (is_miqp_problem) {
        std::cout << "  Model Name      : " << (model.name.empty() ? "(unnamed)" : model.name) << "\n";
        std::cout << "  Problem Class   : MIQP (Mixed-Integer Quadratic Program - UNSUPPORTED)\n";
        std::cout << "  Rows            : " << model.num_rows << "\n";
        std::cout << "  Columns         : " << model.num_cols << "\n";
        std::cout << "  A Nonzeros (NNZ): " << model.A.nnz() << "\n";
        std::cout << "  Q Nonzeros (NNZ): " << model.Q.nnz() << "\n";
        std::cerr << "\n[ERROR] MIQP is explicitly unsupported in Phase 8.\n";
        return 1;
    }

    if (is_qp_problem) {
        std::string conv_desc;
        auto conv = indus::qp::check_convexity(model, &conv_desc);
        std::cout << "  Model Name      : " << (model.name.empty() ? "(unnamed)" : model.name) << "\n";
        std::cout << "  Problem Class   : QP (Continuous Quadratic Program)\n";
        std::cout << "  Objective Form  : 0.5 * xᵀ Q x + cᵀ x + offset\n";
        std::cout << "  Rows            : " << model.num_rows << "\n";
        std::cout << "  Columns         : " << model.num_cols << "\n";
        std::cout << "  A Nonzeros (NNZ): " << model.A.nnz() << "\n";
        std::cout << "  Q Nonzeros (NNZ): " << model.Q.nnz() << "\n";
        std::cout << "  Convexity Status: " << indus::qp::to_string(conv) << " (" << conv_desc << ")\n";
        std::cout << "  Objective Sense : " << (model.sense == indus::ObjSense::kMaximize ? "MAXIMIZE" : "MINIMIZE") << "\n\n";

        if (conv == indus::qp::QpConvexity::kIndefinite || conv == indus::qp::QpConvexity::kNegativeDefinite || conv == indus::qp::QpConvexity::kInvalid) {
            std::cerr << "[ERROR] Nonconvex QP rejected: " << conv_desc << "\n";
            return 1;
        }
    } else {
        int num_integers = 0;
        int num_binaries = 0;
        for (int j = 0; j < model.num_cols; ++j) {
            if (model.col_type.size() > static_cast<size_t>(j) &&
                model.col_type[static_cast<size_t>(j)] == indus::VarType::kInteger) {
                num_integers++;
                if (model.col_lower[static_cast<size_t>(j)] >= 0.0 && model.col_upper[static_cast<size_t>(j)] <= 1.0) {
                    num_binaries++;
                }
            }
        }

        std::cout << "  Model Name      : " << (model.name.empty() ? "(unnamed)" : model.name) << "\n";
        std::cout << "  Problem Class   : " << (is_milp_problem ? "MILP (Mixed-Integer Linear Program)" : "LP (Continuous Linear Program)") << "\n";
        std::cout << "  Rows            : " << model.num_rows << "\n";
        std::cout << "  Columns         : " << model.num_cols << "\n";
        if (is_milp_problem) {
            std::cout << "  Integer Vars    : " << num_integers << " (Binary: " << num_binaries << ")\n";
        }
        std::cout << "  Nonzeros (NNZ)  : " << model.A.nnz() << "\n";
        std::cout << "  Objective Sense : " << (model.sense == indus::ObjSense::kMaximize ? "MAXIMIZE" : "MINIMIZE") << "\n\n";
    }

    // Setup options & algorithm dispatch
    indus::Options options;
    options.enable_presolve = enable_presolve;
    options.enable_scaling = enable_scaling;
    options.time_limit = time_limit;
    options.set("tolerance", tol);
    options.node_limit = node_limit;
    options.mip_relative_gap = mip_relative_gap;
    options.mip_absolute_gap = mip_absolute_gap;
    options.integer_tolerance = integer_tol;

    std::string backend_name;
    if (algorithm == "qp" || is_qp_problem) {
        options.algorithm = "qp";
        backend_name = "Native Primal Active-Set Convex QP";
        options.iteration_limit = (iter_limit > 0) ? iter_limit : 100000;
    } else if (algorithm == "milp" || is_milp_problem) {
        options.algorithm = "milp";
        backend_name = "Branch-and-Bound Native MILP";
    } else if (algorithm == "pdhg_cuda") {
        if (!dev_info.available) {
            std::cout << "[HARDWARE NOTICE] Requested algorithm 'pdhg_cuda', but no discrete CUDA device is available.\n"
                      << "                  Falling back cleanly to pure CPU PDHG solver (SIMD/Warp reference).\n\n";
            options.algorithm = "pdhg_cpu";
            options.use_gpu = false;
            backend_name = "PDHG_CPU (CUDA CPU Fallback)";
        } else {
            options.algorithm = "pdhg_cuda";
            options.use_gpu = true;
            backend_name = "PDHG_CUDA (" + dev_info.device_name + ")";
        }
        options.iteration_limit = (iter_limit > 0) ? iter_limit : 50000;
    } else if (algorithm == "pdhg_cpu" || algorithm == "pdhg") {
        options.algorithm = "pdhg_cpu";
        options.use_gpu = false;
        backend_name = "PDHG_CPU (Multi-Threaded CPU Engine)";
        options.iteration_limit = (iter_limit > 0) ? iter_limit : 50000;
    } else if (algorithm == "dual_simplex" || algorithm == "primal_simplex" || algorithm == "simplex") {
        options.algorithm = algorithm;
        backend_name = "Simplex (" + algorithm + ")";
        options.iteration_limit = (iter_limit > 0) ? iter_limit : 1000000;
    } else { // "auto"
        options.algorithm = "dual_simplex";
        backend_name = "Simplex (dual_simplex auto)";
        options.iteration_limit = (iter_limit > 0) ? iter_limit : 1000000;
    }

    std::cout << "[SOLVING] Backend: " << backend_name << "\n";
    std::cout << "  Presolve        : " << (enable_presolve ? "Enabled" : "Disabled") << "\n";
    std::cout << "  Scaling         : " << (enable_scaling ? "Enabled (Ruiz)" : "Disabled") << "\n";
    if (is_milp_problem) {
        std::cout << "  Node Limit      : " << options.node_limit << "\n";
        std::cout << "  MIP Rel Gap Tol : " << options.mip_relative_gap << "\n";
        std::cout << "  Integrality Tol : " << options.integer_tolerance << "\n";
    } else {
        std::cout << "  Iteration Limit : " << options.iteration_limit << "\n";
    }
    std::cout << "  Tolerance       : " << tol << "\n\n";

    const auto t_start = std::chrono::high_resolution_clock::now();
    indus::Solution sol;
    try {
        sol = indus::solve(model, options);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Solver threw exception during solve: " << e.what() << "\n";
        return 1;
    }
    const auto t_end = std::chrono::high_resolution_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();

    // Print results summary
    std::cout << "========================================================================\n";
    std::cout << "  OPTIMIZATION RESULT SUMMARY                                           \n";
    std::cout << "========================================================================\n";
    std::cout << std::left
              << std::setw(26) << "  Solve Status:"
              << indus::to_string(sol.status) << "\n"
              << std::setw(26) << "  Status Message:"
              << (sol.status_message.empty() ? "(none)" : sol.status_message) << "\n"
              << std::setw(26) << "  Objective Value:"
              << std::fixed << std::setprecision(12) << sol.objective_value << "\n"
              << std::setw(26) << "  Solve Time (s):"
              << std::fixed << std::setprecision(6) << sol.solve_time_seconds << " (wall clock: " << elapsed_sec << " s)\n";

    if (is_milp_problem) {
        std::cout << std::setw(26) << "  Nodes Explored:"
                  << sol.nodes << "\n"
                  << std::setw(26) << "  Open Nodes:"
                  << sol.open_nodes << "\n"
                  << std::setw(26) << "  Best Bound:"
                  << std::fixed << std::setprecision(12) << sol.best_dual_bound << "\n"
                  << std::setw(26) << "  Absolute Gap:"
                  << std::scientific << std::setprecision(4) << sol.absolute_gap << "\n"
                  << std::setw(26) << "  Relative MIP Gap:"
                  << std::scientific << std::setprecision(4) << sol.relative_gap << "\n";
    } else {
        std::cout << std::setw(26) << "  Iterations:"
                  << sol.iterations << "\n";
    }

    std::cout << std::setw(26) << "  Algorithm Used:"
              << sol.algorithm_used << "\n";

    if (enable_presolve && sol.presolve_num_rows >= 0) {
        std::cout << std::setw(26) << "  Presolve Dim (R/C):"
                  << sol.presolve_num_rows << " / " << sol.presolve_num_cols << "\n"
                  << std::setw(26) << "  Doubleton Reductions:"
                  << sol.presolve_doubleton_reductions << "\n";
    }

    // Solution verification
    bool verification_passed = true;
    indus::verifier::VerificationResult vres;
    if (verify && (sol.status == indus::SolveStatus::kOptimal || sol.status == indus::SolveStatus::kFeasible || sol.status == indus::SolveStatus::kNodeLimit)) {
        std::cout << "\n[INDEPENDENT SOLUTION VERIFICATION]\n";
        vres = indus::verifier::verify_solution(model, sol, tol);
        if (vres.is_milp) {
            std::cout << std::left
                      << std::setw(26) << "  Primal Feasibility:"
                      << (vres.primal_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_primal_violation << ")\n"
                      << std::setw(26) << "  Bounds Feasibility:"
                      << (vres.bounds_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_bound_violation << ")\n"
                      << std::setw(26) << "  Integrality:"
                      << (vres.integer_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_integrality_violation << ")\n"
                      << std::setw(26) << "  Recomputed Obj:"
                      << std::fixed << std::setprecision(12) << vres.recomputed_objective << "\n"
                      << std::setw(26) << "  Obj Discrepancy:"
                      << std::scientific << std::setprecision(2) << vres.objective_discrepancy << "\n"
                      << std::setw(26) << "  Global Bound Proof:"
                      << (vres.optimality_proven ? "VERIFIED (Gap <= tol)" : "UNPROVEN / GAP EXCEEDED") << "\n"
                      << std::setw(26) << "  Verification Status:"
                      << (vres.passed ? (vres.optimality_proven ? "PASSED (Optimal)" : "PASSED (Feasible Incumbent)") : "FAILED") << "\n";
            verification_passed = vres.passed;
        } else if (vres.is_qp) {
            std::cout << std::left
                      << std::setw(26) << "  Primal Feasibility:"
                      << (vres.primal_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_primal_violation << ")\n"
                      << std::setw(26) << "  Bounds Feasibility:"
                      << (vres.bounds_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_bound_violation << ")\n"
                      << std::setw(26) << "  Stationarity:"
                      << (vres.max_stationarity_residual <= tol ? "PASSED" : "FAILED")
                      << " (max residual: " << std::scientific << std::setprecision(2) << vres.max_stationarity_residual << ")\n"
                      << std::setw(26) << "  Dual Feasibility:"
                      << (vres.dual_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_dual_violation << ")\n"
                      << std::setw(26) << "  Complementarity:"
                      << (vres.max_complementarity_violation <= tol ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_complementarity_violation << ")\n"
                      << std::setw(26) << "  Convexity Status:"
                      << (vres.is_convex ? "PASSED (" + vres.convexity_status + ")" : "FAILED (NONCONVEX)") << "\n"
                      << std::setw(26) << "  Recomputed Obj:"
                      << std::fixed << std::setprecision(12) << vres.recomputed_objective << "\n"
                      << std::setw(26) << "  Obj Discrepancy:"
                      << std::scientific << std::setprecision(2) << vres.objective_discrepancy << "\n"
                      << std::setw(26) << "  Verification Status:"
                      << (vres.passed ? "PASSED" : "FAILED") << "\n";
            verification_passed = vres.passed;
        } else {
            std::cout << std::left
                      << std::setw(26) << "  Primal Feasibility:"
                      << (vres.primal_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_primal_violation << ")\n"
                      << std::setw(26) << "  Dual Feasibility:"
                      << (vres.dual_feasible ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_dual_violation << ")\n"
                      << std::setw(26) << "  Complementarity:"
                      << (vres.max_complementarity_violation <= tol ? "PASSED" : "FAILED")
                      << " (max viol: " << std::scientific << std::setprecision(2) << vres.max_complementarity_violation << ")\n"
                      << std::setw(26) << "  Recomputed Obj:"
                      << std::fixed << std::setprecision(12) << vres.recomputed_objective << "\n"
                      << std::setw(26) << "  Obj Discrepancy:"
                      << std::scientific << std::setprecision(2) << vres.objective_discrepancy << "\n"
                      << std::setw(26) << "  Verification Status:"
                      << (vres.passed ? "PASSED" : "FAILED") << "\n";
            verification_passed = vres.passed;
        }
    }

    // Export solution files
    if (!sol_out.empty()) {
        try {
            indus::io::write_solution(sol, model, sol_out);
            std::cout << "\n[EXPORT] Solution exported to: " << sol_out << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to write solution file: " << e.what() << "\n";
        }
    }
    if (!json_out.empty()) {
        try {
            indus::io::write_json(sol, model, json_out);
            std::cout << "[EXPORT] Telemetry JSON exported to: " << json_out << "\n";
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to write JSON file: " << e.what() << "\n";
        }
    }

    std::cout << "========================================================================\n";

    // Exit code determination
    if (sol.status == indus::SolveStatus::kOptimal) {
        if (!sol.quality.is_primal_feasible || !sol.quality.is_integer_feasible) {
            std::cerr << "\n[EXIT] Status is Optimal but feasibility check failed.\n";
            return 1;
        }
        if (verify && (!verification_passed || (vres.is_milp && !vres.optimality_proven))) {
            std::cerr << "\n[EXIT] Independent mathematical verification or optimality proof failed.\n";
            return 1;
        }
        std::cout << "\n[EXIT] Optimization successfully converged to verified optimal solution.\n";
        return 0;
    } else if (sol.status == indus::SolveStatus::kUnsupported) {
        std::cerr << "\n[EXIT] Solver terminated due to unsupported problem class or feature: " << sol.status_message << "\n";
        return 1;
    } else if (sol.status == indus::SolveStatus::kModelError) {
        std::cerr << "\n[EXIT] Solver terminated due to model error: " << sol.status_message << "\n";
        return 1;
    } else if (sol.status == indus::SolveStatus::kNodeLimit) {
        std::cerr << "\n[EXIT] Solver terminated due to branch-and-bound node limit (" << sol.nodes << " nodes explored).\n";
        return 1;
    } else if (sol.status == indus::SolveStatus::kIterationLimit || sol.status == indus::SolveStatus::kTimeLimit) {
        std::cerr << "\n[EXIT] Solver terminated due to iteration or time limit.\n";
        return 1;
    } else if (sol.status == indus::SolveStatus::kFeasible) {
        std::cerr << "\n[EXIT] Solver returned feasible point (did not reach optimality).\n";
        return 1;
    } else if (sol.status == indus::SolveStatus::kNumericalError) {
        std::cerr << "\n[EXIT] Solver encountered numerical error.\n";
        return 1;
    } else {
        std::cerr << "\n[EXIT] Solver terminated with non-optimal status: " << indus::to_string(sol.status) << "\n";
        return 1;
    }
}
