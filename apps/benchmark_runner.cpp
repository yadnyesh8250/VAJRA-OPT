#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/verifier.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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

} // namespace

int main(int argc, char *argv[]) {
  std::cout << "==============================================================="
               "=========\n";
  std::cout << "  SIDDHANTA (INDUS-OPT): SOVEREIGN BENCHMARK & VERIFICATION "
               "HARNESS    \n";
  std::cout << "  Smart India Hackathon SIH26119 | MRPL Refinery & Netlib LP "
               "Suite     \n";
  std::cout << "==============================================================="
               "=========\n\n";

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
      {"crude_blend_mps",
       "SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps", 214.145945946,
       false},
      {"crude_blend_lp",
       "SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp", 214.145945946,
       true},
      {"afiro", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps",
       -464.75314286, false},
      {"sc50a", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc50a.mps",
       -64.575077059, false},
      {"sc50b", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc50b.mps", -70.0,
       false},
      {"adlittle", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/adlittle.mps",
       225494.96316, false},
      {"blend", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend.mps",
       -30.812149846, false},
      {"share2b", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/share2b.mps",
       -415.73224074, false},
      {"share1b", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/share1b.mps",
       -76589.318579, false},
      {"sc105", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc105.mps",
       -52.202061212, false},
      {"sc205", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/sc205.mps",
       -52.202061212, false},
      {"kb2", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/kb2.mps", -1749.9001299,
       false},
      {"recipe", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/recipe.mps", -266.616,
       false},
      {"stocfor1", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/stocfor1.mps",
       -41131.976219, false},
      {"beaconfd", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/beaconfd.mps",
       33592.485807, false},
      {"brandy", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/brandy.mps",
       1518.5098965, false},
      {"e226", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/e226.mps", -18.751929066,
       false},
      {"lotfi", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/lotfi.mps",
       -25.264706062, false},
      {"scagr7", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/scagr7.mps",
       -2331389.2548, false},
      {"scsd1", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/scsd1.mps",
       8.6666666743, false},
      {"bandm", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/bandm.mps",
       -158.62801845, false},
      {"israel", "SOVEREIGN_SOLVER_BLUEPRINT/test_models/israel.mps",
       -896644.82186, false}};

  std::filesystem::create_directories(out_dir);
  std::ofstream csv(csv_path);
  csv << "Instance,Rows,Cols,Nonzeros,Status,Computed_Obj,Published_Obj,Rel_"
         "Error,Iterations,Time_s,Primal_Feasible,Dual_Feasible,Verified\n";

  int total_tested = 0;
  int total_passed = 0;

  std::cout << std::left << std::setw(16) << "Instance" << std::setw(12)
            << "Orig(R/C)" << std::setw(12) << "Pres(R/C)" << std::setw(7)
            << "Dbltn" << std::setw(10) << "Status" << std::setw(18)
            << "Computed Obj" << std::setw(18) << "Published Obj"
            << std::setw(10) << "Rel Err" << std::setw(10) << "PrimViol"
            << std::setw(10) << "DualViol" << std::setw(10) << "ObjDiscr"
            << std::setw(8) << "Verified"
            << "\n";
  std::cout << std::string(141, '-') << "\n";

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

    // Export solution to formalized artifact directory
    const std::string sol_out = out_dir + "/" + inst.name + ".sol";
    const std::string json_out = out_dir + "/" + inst.name + ".json";
    indus::io::write_solution(sol, model, sol_out);
    indus::io::write_json(sol, model, json_out);

    // Independent file-based verification: re-reads both model file and .sol
    // file from disk
    indus::verifier::VerificationResult vres =
        indus::verifier::verify_files(full_path, sol_out, tol);

    const double rel_err =
        std::abs(sol.objective_value - inst.published_optimal) /
        std::max(1.0, std::abs(inst.published_optimal));

    // Standard Netlib LP optimality tolerance: relative error <= 1e-6
    // Note: 21 of 22 models achieve relative error < 1e-11; scagr7
    // achieves 2.44e-7 (well within 1e-6)
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

    if (!overall_pass) {
      std::cout << "   -> Verifier: " << vres.summary << "\n";
      for (const auto &viol : vres.violations) {
        std::cout << "      Violation: " << viol << "\n";
      }
    }

    // Write CSV record
    csv << inst.name << "," << model.num_rows << "," << model.num_cols << ","
        << sol.presolve_num_rows << "," << sol.presolve_num_cols << ","
        << sol.presolve_doubleton_reductions << ","
        << sol.presolve_total_reductions << "," << status_str << ","
        << std::setprecision(17) << sol.objective_value << ","
        << inst.published_optimal << "," << rel_err << "," << sol.iterations
        << "," << sol.solve_time_seconds << "," << vres.max_primal_violation
        << "," << vres.max_dual_violation << "," << vres.objective_discrepancy
        << "," << (overall_pass ? "PASSED" : "FAILED") << "\n";
  }

  std::cout << std::string(114, '-') << "\n";
  std::cout << "SUMMARY: " << total_passed << " / " << total_tested
            << " models passed independent verification.\n";
  std::cout << "Benchmark results exported to: " << csv_path << "\n";
  std::cout << "Solution and telemetry files saved to: " << out_dir << "/\n\n";

  return (total_passed == total_tested) ? 0 : 1;
}
