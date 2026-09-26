#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/verifier.hpp"
#include "indus/pdhg.hpp"
#include "indus/gpu.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

namespace {

struct BenchmarkInstance {
  std::string name;
  std::string filepath;
  double published_optimal;
  bool is_lp = false;
};

#ifndef INDUS_SOURCE_DIR
#define INDUS_SOURCE_DIR "."
#endif

std::string find_path(const std::string &rel) {
  std::filesystem::path p_src = std::filesystem::path(INDUS_SOURCE_DIR) / rel;
  if (std::filesystem::exists(p_src))
    return p_src.string();
  if (std::filesystem::exists(rel))
    return rel;
  if (std::filesystem::exists("../" + rel))
    return "../" + rel;
  if (std::filesystem::exists("../../" + rel))
    return "../../" + rel;
  return rel;
}

indus::Model generate_synthetic_sparse_lp(int m, int n, int nonzeros_per_col = 10, uint32_t seed = 42) {
  indus::Model model;
  model.name = "synthetic_sparse_" + std::to_string(m) + "x" + std::to_string(n);
  model.num_rows = m;
  model.num_cols = n;
  model.sense = indus::ObjSense::kMinimize;
  model.c.resize(static_cast<size_t>(n), 1.0);
  model.col_lower.resize(static_cast<size_t>(n), 0.0);
  model.col_upper.resize(static_cast<size_t>(n), 10.0);
  model.row_lower.resize(static_cast<size_t>(m), 1.0);
  model.row_upper.resize(static_cast<size_t>(m), 1e20);
  model.row_names.resize(static_cast<size_t>(m));
  model.col_names.resize(static_cast<size_t>(n));

  for (int i = 0; i < m; ++i) model.row_names[static_cast<size_t>(i)] = "r" + std::to_string(i);
  for (int j = 0; j < n; ++j) {
    model.col_names[static_cast<size_t>(j)] = "x" + std::to_string(j);
    model.c[static_cast<size_t>(j)] = 1.0 + (j % 7) * 0.5;
  }

  std::vector<indus::la::Triplet> triplets;
  triplets.reserve(static_cast<size_t>(n * nonzeros_per_col));

  uint32_t state = seed;
  auto lcg = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int i = 0; i < m; ++i) {
    int col = i % n;
    triplets.push_back({i, col, 1.0 + (lcg() % 5)});
  }

  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < nonzeros_per_col - 1; ++k) {
      int row = static_cast<int>(lcg() % static_cast<uint32_t>(m));
      double val = 1.0 + (lcg() % 9);
      triplets.push_back({row, j, val});
    }
  }

  model.A.set_from_triplets(m, n, triplets);
  return model;
}

} // namespace

int main(int argc, char *argv[]) {
  std::cout << "========================================================================\n";
  std::cout << "  SIDDHANTA (INDUS-OPT): SOVEREIGN BENCHMARK & MULTI-ENGINE HARNESS    \n";
  std::cout << "  Smart India Hackathon SIH26119 | MRPL Refinery & Netlib LP Suite     \n";
  std::cout << "========================================================================\n\n";

  auto dev_info = indus::gpu::probe_device();
  std::cout << "[HARDWARE CONFIGURATION]\n";
  std::cout << "  CPU Platform   : "
#if defined(__aarch64__) || defined(_M_ARM64)
            << "ARM64 (Apple Silicon / AArch64)\n";
#elif defined(__x86_64__) || defined(_M_X64)
            << "x86_64\n";
#else
            << "Generic CPU\n";
#endif
  std::cout << "  GPU Hardware   : " << dev_info.device_name << "\n";
  std::cout << "  CUDA Status    : " << (dev_info.available ? "ACTIVE (Hardware Ready)" : "UNAVAILABLE (Pure CPU Fallback Active)") << "\n";
  if (dev_info.available) {
    std::cout << "  VRAM Available : " << (dev_info.total_vram_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Compute Cap.   : " << dev_info.compute_capability_major << "." << dev_info.compute_capability_minor << "\n";
  }
  std::cout << "\n";

  std::string out_dir = "build/benchmark_artifacts";
  std::string csv_path = "build/benchmark_results.csv";
  double tol = 1e-4;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--csv" && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (arg == "--tol" && i + 1 < argc) {
      tol = std::stod(argv[++i]);
    }
  }

  std::vector<BenchmarkInstance> instances = {
      {"crude_blend_mps", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps", 214.145945946, false},
      {"crude_blend_lp", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp", 214.145945946, true},
      {"afiro", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps", -464.75314286, false},
      {"sc50a", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc50a.mps", -64.575077059, false},
      {"sc50b", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc50b.mps", -70.0, false},
      {"adlittle", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/adlittle.mps", 225494.96316, false},
      {"blend", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend.mps", -30.812149846, false},
      {"share2b", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/share2b.mps", -415.73224074, false},
      {"share1b", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/share1b.mps", -76589.318579, false},
      {"sc105", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc105.mps", -52.202061212, false},
      {"sc205", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc205.mps", -52.202061212, false},
      {"kb2", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/kb2.mps", -1749.9001299, false},
      {"recipe", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/recipe.mps", -266.616, false},
      {"stocfor1", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/stocfor1.mps", -41131.976219, false},
      {"beaconfd", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/beaconfd.mps", 33592.485807, false},
      {"brandy", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/brandy.mps", 1518.5098965, false},
      {"e226", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/e226.mps", -18.751929066, false},
      {"lotfi", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/lotfi.mps", -25.264706062, false},
      {"scagr7", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/scagr7.mps", -2331389.2548, false},
      {"scsd1", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/scsd1.mps", 8.6666666743, false},
      {"bandm", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/bandm.mps", -158.62801845, false},
      {"israel", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/israel.mps", -896644.82186, false}
  };

  std::filesystem::create_directories(out_dir);
  std::ofstream csv(csv_path);
  csv << "Suite,Instance,Rows,Cols,Nonzeros,Engine,GpuDevice,PresolveDim,Doubletons,"
         "Status,ComputedObj,PublishedObj,RelErr,Iterations,SetupTime_s,IterTime_s,"
         "TotalTime_s,PrimViol,DualViol,ObjDiscr,Speedup,Verified\n";

  std::cout << "[SUITE 1: 22 NETLIB & MRPL LP BASELINE - PRESOLVED REVISED SIMPLEX]\n";
  std::cout << std::left << std::setw(16) << "Instance" << std::setw(12)
            << "Orig(R/C)" << std::setw(12) << "Pres(R/C)" << std::setw(7)
            << "Dbltn" << std::setw(10) << "Status" << std::setw(18)
            << "Computed Obj" << std::setw(18) << "Published Obj"
            << std::setw(10) << "Rel Err" << std::setw(10) << "PrimViol"
            << std::setw(10) << "DualViol" << std::setw(10) << "ObjDiscr"
            << std::setw(8) << "Verified" << "\n";
  std::cout << std::string(141, '-') << "\n";

  int total_tested = 0;
  int total_passed = 0;

  for (const auto &inst : instances) {
    total_tested++;
    const std::string full_path = find_path(inst.filepath);

    indus::Model model;
    try {
      if (inst.is_lp) {
        model = indus::io::read_lp(full_path);
      } else {
        model = indus::io::read_mps(full_path);
      }
    } catch (const std::exception &e) {
      std::cout << std::left << std::setw(16) << inst.name
                << "ERROR LOADING: " << e.what() << "\n";
      continue;
    }

    indus::Options opts;
    opts.enable_presolve = true;
    opts.iteration_limit = 50000;

    indus::Solution sol = indus::solve(model, opts);

    const std::string sol_out = out_dir + "/" + inst.name + ".sol";
    const std::string json_out = out_dir + "/" + inst.name + ".json";
    indus::io::write_solution(sol, model, sol_out);
    indus::io::write_json(sol, model, json_out);

    indus::verifier::VerificationResult vres =
        indus::verifier::verify_files(full_path, sol_out, tol);

    const double rel_err =
        std::abs(sol.objective_value - inst.published_optimal) /
        std::max(1.0, std::abs(inst.published_optimal));

    const bool matches_published = (rel_err <= 1e-6);
    const bool overall_pass = (sol.status == indus::SolveStatus::kOptimal) &&
                              vres.passed && matches_published;

    if (overall_pass) {
      total_passed++;
    }

    std::string status_str =
        (sol.status == indus::SolveStatus::kOptimal)          ? "OPTIMAL"
        : (sol.status == indus::SolveStatus::kFeasible)       ? "FEASIBLE"
        : (sol.status == indus::SolveStatus::kNumericalError) ? "NUM_ERR"
                                                              : "OTHER";

    const std::string orig_dim =
        std::to_string(model.num_rows) + "/" + std::to_string(model.num_cols);
    const std::string pres_dim = std::to_string(sol.presolve_num_rows) + "/" +
                                 std::to_string(sol.presolve_num_cols);

    std::cout << std::left << std::setw(16) << inst.name << std::setw(12)
              << orig_dim << std::setw(12) << pres_dim << std::setw(7)
              << sol.presolve_doubleton_reductions << std::setw(10)
              << status_str << std::setw(18) << std::setprecision(8)
              << sol.objective_value << std::setw(18) << std::setprecision(8)
              << inst.published_optimal << std::setw(10) << std::scientific
              << std::setprecision(1) << rel_err << std::setw(10)
              << std::scientific << std::setprecision(1)
              << vres.max_primal_violation << std::setw(10) << std::scientific
              << std::setprecision(1) << vres.max_dual_violation
              << std::setw(10) << std::scientific << std::setprecision(1)
              << vres.objective_discrepancy << std::setw(8)
              << (overall_pass ? "PASSED" : "FAILED") << "\n";

    csv << "NetlibBaseline," << inst.name << "," << model.num_rows << "," << model.num_cols << ","
        << model.A.nnz() << ",Simplex," << dev_info.device_name << ","
        << pres_dim << "," << sol.presolve_doubleton_reductions << "," << status_str << ","
        << std::setprecision(17) << sol.objective_value << ","
        << inst.published_optimal << "," << rel_err << "," << sol.iterations << ",0.0,0.0,"
        << sol.solve_time_seconds << "," << vres.max_primal_violation << ","
        << vres.max_dual_violation << "," << vres.objective_discrepancy << ",1.00,"
        << (overall_pass ? "PASSED" : "FAILED") << "\n";
  }

  std::cout << std::string(141, '-') << "\n";
  std::cout << "BASELINE SUMMARY: " << total_passed << " / " << total_tested
            << " models passed independent verification.\n\n";

  // SUITE 2: MULTI-ENGINE SPEEDUP & COMPARATIVE BENCHMARK (Simplex vs CPU PDHG vs GPU PDHG)
  std::cout << "[SUITE 2: MULTI-ENGINE RESTART-PDHG VS SIMPLEX COMPARISON]\n";
  std::cout << "Categories: Small (Launch Overhead), Medium Sparse, Large Generated Expander, MRPL Blend\n";
  std::cout << std::left << std::setw(20) << "Model"
            << std::setw(14) << "Dimensions"
            << std::setw(10) << "Nonzeros"
            << std::setw(12) << "Simplex(s)"
            << std::setw(14) << "PDHG_CPU(s)"
            << std::setw(14) << "PDHG_GPU(s)"
            << std::setw(14) << "Speedup"
            << std::setw(10) << "ObjDiff"
            << std::setw(10) << "Verified"
            << "\n";
  std::cout << std::string(108, '-') << "\n";

  std::vector<std::pair<std::string, std::string>> comparison_models = {
      {"crude_blend (MRPL)", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"},
      {"afiro (Small)", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps"},
      {"sc50a (Medium)", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc50a.mps"},
      {"blend (Medium)", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend.mps"},
      {"beaconfd (Medium)", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/beaconfd.mps"}
  };

  for (const auto &[label, rel_path] : comparison_models) {
    const std::string full_path = find_path(rel_path);
    indus::Model model = indus::io::read_mps(full_path);

    // 1. Simplex run
    indus::Options s_opts;
    s_opts.set("algorithm", "dual_simplex");
    s_opts.enable_presolve = false;
    indus::Solution s_sol = indus::solve(model, s_opts);

    // 2. CPU PDHG run
    indus::Options cpu_opts;
    cpu_opts.set("algorithm", "pdhg_cpu");
    cpu_opts.enable_presolve = false;
    cpu_opts.iteration_limit = 50000;
    cpu_opts.set("tolerance", 1e-4);
    indus::Solution cpu_sol = indus::solve(model, cpu_opts);

    // 3. GPU PDHG run (or CPU fallback)
    indus::Options gpu_opts;
    gpu_opts.set("algorithm", "pdhg_cuda");
    gpu_opts.use_gpu = true;
    gpu_opts.enable_presolve = false;
    gpu_opts.iteration_limit = 50000;
    gpu_opts.set("tolerance", 1e-4);
    indus::Solution gpu_sol = indus::solve(model, gpu_opts);

    const double obj_diff = std::abs(s_sol.objective_value - cpu_sol.objective_value) /
                            std::max(1.0, std::abs(s_sol.objective_value));

    std::string speedup_str;
    if (dev_info.available && gpu_sol.solve_time_seconds > 0.0) {
      double sp = cpu_sol.solve_time_seconds / gpu_sol.solve_time_seconds;
      std::ostringstream ss;
      ss << std::fixed << std::setprecision(2) << sp << "x";
      speedup_str = ss.str();
    } else {
      speedup_str = "CPU Fallback";
    }

    const std::string dims = std::to_string(model.num_rows) + "x" + std::to_string(model.num_cols);
    const bool v_pass = (obj_diff < 1e-3);

    std::cout << std::left << std::setw(20) << label
              << std::setw(14) << dims
              << std::setw(10) << model.A.nnz()
              << std::setw(12) << std::fixed << std::setprecision(4) << s_sol.solve_time_seconds
              << std::setw(14) << std::fixed << std::setprecision(4) << cpu_sol.solve_time_seconds
              << std::setw(14) << std::fixed << std::setprecision(4) << gpu_sol.solve_time_seconds
              << std::setw(14) << speedup_str
              << std::setw(10) << std::scientific << std::setprecision(1) << obj_diff
              << std::setw(10) << (v_pass ? "PASSED" : "FAILED")
              << "\n";

    csv << "EngineComparison," << label << "," << model.num_rows << "," << model.num_cols << ","
        << model.A.nnz() << ",PDHG_CPU," << dev_info.device_name << ",N/A,0,"
        << indus::to_string(cpu_sol.status) << ","
        << std::setprecision(17) << cpu_sol.objective_value << "," << s_sol.objective_value << ","
        << obj_diff << "," << cpu_sol.iterations << ",0.0,0.0," << cpu_sol.solve_time_seconds << ","
        << cpu_sol.quality.max_primal_violation << "," << cpu_sol.quality.max_dual_violation << ",0.0,"
        << speedup_str << "," << (v_pass ? "PASSED" : "FAILED") << "\n";
  }

  // 4. Large Synthetic Expander Model (100,000+ Nonzeros)
  std::cout << "\n[LARGE GENERATED SPARSE LP BENCHMARK]\n";
  int large_rows = 5000;
  int large_cols = 10000;
  int nonzeros_per_col = 10;
  std::cout << "  Generating large sparse LP: " << large_rows << " constraints x "
            << large_cols << " variables (~100,000 nonzeros)...\n";
  indus::Model large_lp = generate_synthetic_sparse_lp(large_rows, large_cols, nonzeros_per_col, 12345);
  std::cout << "  Model created: " << large_lp.num_rows << " rows, " << large_lp.num_cols
            << " cols, " << large_lp.A.nnz() << " nonzeros.\n";

  indus::Options large_opts;
  large_opts.set("algorithm", "pdhg_cpu");
  large_opts.enable_presolve = false;
  large_opts.iteration_limit = 2000;
  large_opts.set("tolerance", 1e-4);

  const auto large_t0 = std::chrono::high_resolution_clock::now();
  indus::Solution large_sol = indus::solve(large_lp, large_opts);
  const double large_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - large_t0).count();

  std::cout << "  Large Sparse LP Solve completed in " << std::fixed << std::setprecision(3) << large_time
            << " s (" << large_sol.iterations << " iterations, status: " << indus::to_string(large_sol.status)
            << ", obj: " << large_sol.objective_value << ")\n";

  if (!dev_info.available) {
    std::cout << "  [HARDWARE NOTE] No NVIDIA GPU detected on this host. Hardware speedup is reported as: "
              << "'CUDA implementation compiled but hardware speedup not measured.'\n";
  }

  std::cout << "\nBenchmark results saved to: " << csv_path << "\n";
  return (total_passed == total_tested) ? 0 : 1;
}
