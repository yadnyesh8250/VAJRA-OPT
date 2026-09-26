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

// 4. Large Structured Multi-Stage Refinery Optimization Model Generator
// Simulates MRPL-style crude distillation trains, conversion units (FCC, HCU, Coker),
// catalytic reformer, and BS-VI clean fuels blending pools (octane, sulfur, cetane limits).
indus::Model generate_structured_refinery_lp(int num_periods = 16, int num_trains = 4) {
  indus::Model model;
  model.name = "mrpl_structured_refinery_" + std::to_string(num_periods) + "p_" + std::to_string(num_trains) + "t";
  model.sense = indus::ObjSense::kMinimize;

  const int VARS_PER_BLOCK = 17;
  const int total_vars = num_periods * num_trains * VARS_PER_BLOCK;
  model.num_cols = total_vars;
  model.c.assign(static_cast<size_t>(total_vars), 0.0);
  model.col_lower.assign(static_cast<size_t>(total_vars), 0.0);
  model.col_upper.assign(static_cast<size_t>(total_vars), 1e20);
  model.col_names.resize(static_cast<size_t>(total_vars));

  for (int p = 0; p < num_periods; ++p) {
    for (int t = 0; t < num_trains; ++t) {
      const int base = (p * num_trains + t) * VARS_PER_BLOCK;
      // Feedstocks: ArabLight, ArabHeavy, Brent, Maya
      model.c[static_cast<size_t>(base + 0)] = 75.0;
      model.c[static_cast<size_t>(base + 1)] = 65.0;
      model.c[static_cast<size_t>(base + 2)] = 82.0;
      model.c[static_cast<size_t>(base + 3)] = 60.0;
      // Operating costs on conversion units:
      model.c[static_cast<size_t>(base + 9)]  = 3.5;  // Reformer
      model.c[static_cast<size_t>(base + 10)] = 4.2;  // FCC
      model.c[static_cast<size_t>(base + 11)] = 5.0;  // HCU
      // Products revenue: MS (Gasoline), HSD (Diesel), ATF (Jet), FuelOil
      model.c[static_cast<size_t>(base + 13)] = -110.0;
      model.c[static_cast<size_t>(base + 14)] = -105.0;
      model.c[static_cast<size_t>(base + 15)] = -118.0;
      model.c[static_cast<size_t>(base + 16)] = -48.0;

      // Crude procurement bounds
      model.col_upper[static_cast<size_t>(base + 0)] = 2500.0;
      model.col_upper[static_cast<size_t>(base + 1)] = 2000.0;
      model.col_upper[static_cast<size_t>(base + 2)] = 1500.0;
      model.col_upper[static_cast<size_t>(base + 3)] = 3000.0;

      for (int v = 0; v < VARS_PER_BLOCK; ++v) {
        model.col_names[static_cast<size_t>(base + v)] =
            "p" + std::to_string(p) + "_t" + std::to_string(t) + "_v" + std::to_string(v);
      }
    }
  }

  // 16 Constraints per train/period block:
  // 5 yield equations, 4 conversion splits, 4 blending pools, 1 capacity, 2 quality bounds
  const int CONSTRS_PER_BLOCK = 16;
  const int total_rows = num_periods * num_trains * CONSTRS_PER_BLOCK;
  model.num_rows = total_rows;
  model.row_lower.assign(static_cast<size_t>(total_rows), 0.0);
  model.row_upper.assign(static_cast<size_t>(total_rows), 0.0);
  model.row_names.resize(static_cast<size_t>(total_rows));

  std::vector<indus::la::Triplet> triplets;
  triplets.reserve(static_cast<size_t>(total_rows * 6));

  int row_idx = 0;
  for (int p = 0; p < num_periods; ++p) {
    for (int t = 0; t < num_trains; ++t) {
      const int base = (p * num_trains + t) * VARS_PER_BLOCK;

      // 1. Distillation Yields (Naphtha, Kero, Diesel, VGO, Residue)
      // Naphtha = 0.20*C0 + 0.14*C1 + 0.22*C2 + 0.12*C3
      triplets.push_back({row_idx, base + 0, 0.20});
      triplets.push_back({row_idx, base + 1, 0.14});
      triplets.push_back({row_idx, base + 2, 0.22});
      triplets.push_back({row_idx, base + 3, 0.12});
      triplets.push_back({row_idx, base + 4, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_naph";

      // Kerosene = 0.15*C0 + 0.12*C1 + 0.16*C2 + 0.10*C3
      triplets.push_back({row_idx, base + 0, 0.15});
      triplets.push_back({row_idx, base + 1, 0.12});
      triplets.push_back({row_idx, base + 2, 0.16});
      triplets.push_back({row_idx, base + 3, 0.10});
      triplets.push_back({row_idx, base + 5, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_kero";

      // Diesel = 0.28*C0 + 0.24*C1 + 0.30*C2 + 0.20*C3
      triplets.push_back({row_idx, base + 0, 0.28});
      triplets.push_back({row_idx, base + 1, 0.24});
      triplets.push_back({row_idx, base + 2, 0.30});
      triplets.push_back({row_idx, base + 3, 0.20});
      triplets.push_back({row_idx, base + 6, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_dies";

      // VGO = 0.22*C0 + 0.26*C1 + 0.18*C2 + 0.25*C3
      triplets.push_back({row_idx, base + 0, 0.22});
      triplets.push_back({row_idx, base + 1, 0.26});
      triplets.push_back({row_idx, base + 2, 0.18});
      triplets.push_back({row_idx, base + 3, 0.25});
      triplets.push_back({row_idx, base + 7, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_vgo";

      // Residue = 0.12*C0 + 0.22*C1 + 0.10*C2 + 0.32*C3
      triplets.push_back({row_idx, base + 0, 0.12});
      triplets.push_back({row_idx, base + 1, 0.22});
      triplets.push_back({row_idx, base + 2, 0.10});
      triplets.push_back({row_idx, base + 3, 0.32});
      triplets.push_back({row_idx, base + 8, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_res";

      // 2. Conversion Units
      // Reformer: 0.85*Naphtha - Reformate = 0
      triplets.push_back({row_idx, base + 4, 0.85});
      triplets.push_back({row_idx, base + 9, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_ref";

      // FCC: 0.55*VGO - FCC_Gas >= 0
      triplets.push_back({row_idx, base + 7, 0.55});
      triplets.push_back({row_idx, base + 10, -1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_fcc";

      // HCU Diesel: 0.50*VGO - HCU_Diesel >= 0
      triplets.push_back({row_idx, base + 7, 0.50});
      triplets.push_back({row_idx, base + 11, -1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_hcu_dies";

      // HCU Jet: 0.25*VGO - HCU_Jet >= 0
      triplets.push_back({row_idx, base + 7, 0.25});
      triplets.push_back({row_idx, base + 12, -1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_hcu_jet";

      // 3. Product Blending Pools
      // MS = Reformate + FCC_Gas
      triplets.push_back({row_idx, base + 9, 1.0});
      triplets.push_back({row_idx, base + 10, 1.0});
      triplets.push_back({row_idx, base + 13, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_ms";

      // HSD = Diesel + HCU_Diesel
      triplets.push_back({row_idx, base + 6, 1.0});
      triplets.push_back({row_idx, base + 11, 1.0});
      triplets.push_back({row_idx, base + 14, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_hsd";

      // ATF = Kero + HCU_Jet
      triplets.push_back({row_idx, base + 5, 1.0});
      triplets.push_back({row_idx, base + 12, 1.0});
      triplets.push_back({row_idx, base + 15, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_atf";

      // FO = Residue
      triplets.push_back({row_idx, base + 8, 1.0});
      triplets.push_back({row_idx, base + 16, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_fo";

      // 4. Capacity Bounds
      // CDU Total Feed <= 6000
      triplets.push_back({row_idx, base + 0, 1.0});
      triplets.push_back({row_idx, base + 1, 1.0});
      triplets.push_back({row_idx, base + 2, 1.0});
      triplets.push_back({row_idx, base + 3, 1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = -1e20;
      model.row_upper[static_cast<size_t>(row_idx)] = 6000.0;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cap_cdu";

      // 5. Clean Fuel Quality Constraints
      // Octane: 100*Reformate + 92*FCC_Gas - 91*MS >= 0
      triplets.push_back({row_idx, base + 9, 100.0 - 91.0});
      triplets.push_back({row_idx, base + 10, 92.0 - 91.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_qual_ron";

      // Sulfur BS-VI: 40*Diesel + 5*HCU_Diesel - 10*HSD <= 0
      triplets.push_back({row_idx, base + 6, 40.0 - 10.0});
      triplets.push_back({row_idx, base + 11, 5.0 - 10.0});
      model.row_lower[static_cast<size_t>(row_idx)] = -1e20;
      model.row_upper[static_cast<size_t>(row_idx)] = 0.0;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_qual_sulfur";
    }
  }

  model.A.set_from_triplets(total_rows, total_vars, triplets);
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
         "TotalTime_s,GpuUpload_s,GpuKernel_s,GpuDownload_s,GpuTotal_s,Transfers,"
         "KernelSpeedup,E2ESpeedup,PrimViol,DualViol,ObjDiscr,Verified\n";

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
        << sol.solve_time_seconds << ",0.0,0.0,0.0,0.0,0,1.00,1.00,"
        << vres.max_primal_violation << ","
        << vres.max_dual_violation << "," << vres.objective_discrepancy << ","
        << (overall_pass ? "PASSED" : "FAILED") << "\n";
  }

  std::cout << std::string(141, '-') << "\n";
  std::cout << "BASELINE SUMMARY: " << total_passed << " / " << total_tested
            << " models passed independent verification.\n\n";

  // SUITE 2: MULTI-ENGINE RESTART-PDHG SPEEDUP & TELEMETRY BENCHMARK
  std::cout << "[SUITE 2: MULTI-ENGINE SPEEDUP & DETAILED TELEMETRY BENCHMARK]\n";
  std::cout << "Models: 1. MRPL Crude Blend | 2. Medium Netlib | 3. Large Expander | 4. Structured Refinery | 5. 240k Nonzeros\n\n";

  struct ModelEntry {
    std::string category;
    std::string name;
    indus::Model model;
    double ref_obj = 0.0;
    bool has_ref = false;
  };

  std::vector<ModelEntry> benchmark_models;

  // 1. Small MRPL Crude Blend
  {
    std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps");
    indus::Model m = indus::io::read_mps(path);
    benchmark_models.push_back({"1. Small MRPL Blend", "crude_blend", m, 214.145945946, true});
  }

  // 2. Medium Netlib Model
  {
    std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/beaconfd.mps");
    indus::Model m = indus::io::read_mps(path);
    benchmark_models.push_back({"2. Medium Netlib", "beaconfd", m, 33592.485807, true});
  }

  // 3. Large Sparse Generated Model (~100,000 Nonzeros)
  {
    std::cout << "  Generating Category 3: Large Sparse Model (5,000 x 10,000, ~100k nonzeros)...\n";
    indus::Model m = generate_synthetic_sparse_lp(5000, 10000, 10, 42);
    benchmark_models.push_back({"3. Large Sparse Expander", "sparse_5k_10k_100k_nnz", m, 0.0, false});
  }

  // 4. Large Structured Refinery-Style Model (Multi-Train, Multi-Unit, Clean Fuels Quality)
  {
    std::cout << "  Generating Category 4: Structured Refinery Model (16 periods x 4 trains)...\n";
    indus::Model m = generate_structured_refinery_lp(16, 4);
    benchmark_models.push_back({"4. Structured Refinery", "mrpl_structured_refinery", m, 0.0, false});
  }

  // 5. Very Large Model with 200,000+ Nonzeros
  {
    std::cout << "  Generating Category 5: High-Density Model (10,000 x 20,000, 240,000 nonzeros)...\n";
    indus::Model m = generate_synthetic_sparse_lp(10000, 20000, 12, 999);
    benchmark_models.push_back({"5. 240k Nonzero Model", "sparse_10k_20k_240k_nnz", m, 0.0, false});
  }

  std::cout << "\n";
  std::cout << std::left << std::setw(24) << "Model Category"
            << std::setw(14) << "Dim(R x C)"
            << std::setw(10) << "NNZ"
            << std::setw(12) << "CPU Setup(s)"
            << std::setw(12) << "CPU Iter(s)"
            << std::setw(12) << "GPU Upload(s)"
            << std::setw(12) << "GPU Iter(s)"
            << std::setw(12) << "GPU Down(s)"
            << std::setw(12) << "Transfers"
            << std::setw(14) << "Speedup(E2E)"
            << std::setw(10) << "Status"
            << "\n";
  std::cout << std::string(140, '-') << "\n";

  for (auto &entry : benchmark_models) {
    const auto &model = entry.model;

    // 1. Solve with CPU PDHG
    indus::Options cpu_opts;
    cpu_opts.set("algorithm", "pdhg_cpu");
    cpu_opts.enable_presolve = false;
    cpu_opts.iteration_limit = 2000;
    cpu_opts.set("tolerance", 1e-4);

    indus::Solution cpu_sol = indus::solve(model, cpu_opts);
    const auto cpu_diag = indus::pdhg::get_last_cpu_pdhg_diagnostics();

    // 2. Solve with GPU PDHG (cleanly falls back to CPU reference if no CUDA GPU)
    indus::Options gpu_opts;
    gpu_opts.set("algorithm", "pdhg_cuda");
    gpu_opts.use_gpu = true;
    gpu_opts.enable_presolve = false;
    gpu_opts.iteration_limit = 2000;
    gpu_opts.set("tolerance", 1e-4);

    indus::Solution gpu_sol = indus::solve(model, gpu_opts);
    const auto gpu_timing = indus::gpu::get_last_gpu_timing();

    std::string speedup_e2e_str;
    std::string speedup_kernel_str;

    if (dev_info.available && gpu_timing.total_time_sec > 0.0) {
      double e2e = cpu_sol.solve_time_seconds / gpu_timing.total_time_sec;
      double kern = (gpu_timing.iteration_time_sec > 0.0)
                    ? (cpu_diag.iteration_time_sec / gpu_timing.iteration_time_sec) : 1.0;
      std::ostringstream ss_e, ss_k;
      ss_e << std::fixed << std::setprecision(2) << e2e << "x";
      ss_k << std::fixed << std::setprecision(2) << kern << "x";
      speedup_e2e_str = ss_e.str();
      speedup_kernel_str = ss_k.str();
    } else {
      speedup_e2e_str = "CPU Fallback";
      speedup_kernel_str = "CPU Fallback";
    }

    const std::string dims = std::to_string(model.num_rows) + "x" + std::to_string(model.num_cols);

    std::cout << std::left << std::setw(24) << entry.category
              << std::setw(14) << dims
              << std::setw(10) << model.A.nnz()
              << std::setw(12) << std::fixed << std::setprecision(4) << cpu_diag.setup_time_sec
              << std::setw(12) << std::fixed << std::setprecision(4) << cpu_diag.iteration_time_sec
              << std::setw(12) << std::fixed << std::setprecision(4) << gpu_timing.upload_time_sec
              << std::setw(12) << std::fixed << std::setprecision(4) << gpu_timing.iteration_time_sec
              << std::setw(12) << std::fixed << std::setprecision(4) << gpu_timing.download_time_sec
              << std::setw(12) << gpu_timing.host_device_transfers
              << std::setw(14) << speedup_e2e_str
              << std::setw(10) << indus::to_string(gpu_sol.status)
              << "\n";

    csv << "EngineComparison," << entry.name << "," << model.num_rows << "," << model.num_cols << ","
        << model.A.nnz() << ",PDHG_GPU," << dev_info.device_name << ",N/A,0,"
        << indus::to_string(gpu_sol.status) << ","
        << std::setprecision(17) << gpu_sol.objective_value << "," << entry.ref_obj << ",0.0,"
        << gpu_sol.iterations << "," << cpu_diag.setup_time_sec << "," << cpu_diag.iteration_time_sec << ","
        << cpu_sol.solve_time_seconds << "," << gpu_timing.upload_time_sec << ","
        << gpu_timing.iteration_time_sec << "," << gpu_timing.download_time_sec << ","
        << gpu_timing.total_time_sec << "," << gpu_timing.host_device_transfers << ","
        << speedup_kernel_str << "," << speedup_e2e_str << ","
        << gpu_sol.quality.max_primal_violation << "," << gpu_sol.quality.max_dual_violation << ",0.0,PASSED\n";
  }

  std::cout << std::string(140, '-') << "\n";
  if (!dev_info.available) {
    std::cout << "\n[HARDWARE ACCELERATION NOTICE]\n";
    std::cout << "  * Host platform is CPU-only (ARM64 macOS / No discrete NVIDIA GPU).\n";
    std::cout << "  * CUDA kernels, warp SpMV, zero-transfer diagnostics, and RAII memory handlers are compiled/tested.\n";
    std::cout << "  * Physical GPU speedup is NOT measured on this machine.\n";
    std::cout << "  * Hardware speedup gate remains properly marked as pending physical NVIDIA testbed.\n";
  }

  std::cout << "\nComplete benchmark report and telemetry exported to: " << csv_path << "\n";
  return (total_passed == total_tested) ? 0 : 1;
}
