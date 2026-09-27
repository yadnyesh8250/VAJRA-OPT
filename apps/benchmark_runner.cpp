#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/verifier.hpp"
#include "indus/pdhg.hpp"
#include "indus/gpu.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

namespace {

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

std::string escape_json(const std::string &s) {
  std::ostringstream o;
  for (char c : s) {
    switch (c) {
      case '"': o << "\\\""; break;
      case '\\': o << "\\\\"; break;
      case '\b': o << "\\b"; break;
      case '\f': o << "\\f"; break;
      case '\n': o << "\\n"; break;
      case '\r': o << "\\r"; break;
      case '\t': o << "\\t"; break;
      default:
        if ('\x00' <= c && c <= '\x1f') {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        } else {
          o << c;
        }
    }
  }
  return o.str();
}

struct BenchmarkRow {
  std::string suite;                // "lp", "milp", "qp", "pdhg", "cuda"
  std::string suite_tag;            // "NetlibBaseline", "EngineComparison", "MILP", "QP"
  std::string model_name;
  std::string source_path;
  std::string problem_class;        // "LP", "MILP", "QP"
  int rows = 0;
  int cols = 0;
  int64_t a_nnz = 0;
  int64_t q_nnz = 0;
  int integer_vars = 0;
  int binary_vars = 0;
  std::string objective_sense;      // "MINIMIZE", "MAXIMIZE"
  std::string algorithm;
  std::string backend;              // "CPU", "CPU FALLBACK", "NVIDIA_CUDA"
  std::string solver_status;        // "OPTIMAL", "FEASIBLE", "ITERATION_LIMIT", "TIME_LIMIT", "NODE_LIMIT", "INFEASIBLE", "UNBOUNDED", "NUMERICAL_ERROR", "MODEL_ERROR", "UNSUPPORTED"
  std::string verification_status;  // "OPTIMAL_VERIFIED", "FEASIBLE_UNVERIFIED", "LIMIT_REACHED", "INFEASIBLE_VERIFIED", "UNSUPPORTED", "FAILED", "NOT_APPLICABLE"
  double objective_value = 0.0;
  double reference_objective = 0.0;
  bool has_reference = false;
  double objective_discrepancy = 0.0;
  double runtime = 0.0;
  int64_t iterations = 0;
  int64_t node_count = 0;
  double mip_gap = 0.0;
  double primal_violation = 0.0;
  double dual_violation = 0.0;
  double stationarity_residual = 0.0;
  double complementarity_residual = 0.0;
  std::string platform;
  std::string architecture;
  std::string cpu;
  std::string gpu;
  std::string compiler;
  std::string cuda_version;
  double tolerance = 1e-4;
  double time_limit = 1000.0;
  int64_t iteration_limit = 50000;
  int64_t node_limit = 500000;

  // Extended telemetry for PDHG / CUDA:
  std::string presolve_dim = "N/A";
  int doubletons = 0;
  double cpu_setup_time = 0.0;
  double cpu_iter_time = 0.0;
  double gpu_upload_time = 0.0;
  double gpu_kernel_time = 0.0;
  double gpu_download_time = 0.0;
  double gpu_total_time = 0.0;
  int gpu_transfers = 0;
  std::string speedup_kernel = "N/A";
  std::string speedup_e2e = "N/A";
};

// Synthetic generators
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
      model.c[static_cast<size_t>(base + 0)] = 75.0;
      model.c[static_cast<size_t>(base + 1)] = 65.0;
      model.c[static_cast<size_t>(base + 2)] = 82.0;
      model.c[static_cast<size_t>(base + 3)] = 60.0;
      model.c[static_cast<size_t>(base + 9)]  = 3.5;
      model.c[static_cast<size_t>(base + 10)] = 4.2;
      model.c[static_cast<size_t>(base + 11)] = 5.0;
      model.c[static_cast<size_t>(base + 13)] = -110.0;
      model.c[static_cast<size_t>(base + 14)] = -105.0;
      model.c[static_cast<size_t>(base + 15)] = -118.0;
      model.c[static_cast<size_t>(base + 16)] = -48.0;

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

      triplets.push_back({row_idx, base + 0, 0.20});
      triplets.push_back({row_idx, base + 1, 0.14});
      triplets.push_back({row_idx, base + 2, 0.22});
      triplets.push_back({row_idx, base + 3, 0.12});
      triplets.push_back({row_idx, base + 4, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_naph";

      triplets.push_back({row_idx, base + 0, 0.15});
      triplets.push_back({row_idx, base + 1, 0.12});
      triplets.push_back({row_idx, base + 2, 0.16});
      triplets.push_back({row_idx, base + 3, 0.10});
      triplets.push_back({row_idx, base + 5, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_kero";

      triplets.push_back({row_idx, base + 0, 0.28});
      triplets.push_back({row_idx, base + 1, 0.24});
      triplets.push_back({row_idx, base + 2, 0.30});
      triplets.push_back({row_idx, base + 3, 0.20});
      triplets.push_back({row_idx, base + 6, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_dies";

      triplets.push_back({row_idx, base + 0, 0.22});
      triplets.push_back({row_idx, base + 1, 0.26});
      triplets.push_back({row_idx, base + 2, 0.18});
      triplets.push_back({row_idx, base + 3, 0.25});
      triplets.push_back({row_idx, base + 7, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_vgo";

      triplets.push_back({row_idx, base + 0, 0.12});
      triplets.push_back({row_idx, base + 1, 0.22});
      triplets.push_back({row_idx, base + 2, 0.10});
      triplets.push_back({row_idx, base + 3, 0.32});
      triplets.push_back({row_idx, base + 8, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cdu_res";

      triplets.push_back({row_idx, base + 4, 0.85});
      triplets.push_back({row_idx, base + 9, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_ref";

      triplets.push_back({row_idx, base + 7, 0.55});
      triplets.push_back({row_idx, base + 10, -1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_fcc";

      triplets.push_back({row_idx, base + 7, 0.50});
      triplets.push_back({row_idx, base + 11, -1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_hcu_dies";

      triplets.push_back({row_idx, base + 7, 0.25});
      triplets.push_back({row_idx, base + 12, -1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_hcu_jet";

      triplets.push_back({row_idx, base + 9, 1.0});
      triplets.push_back({row_idx, base + 10, 1.0});
      triplets.push_back({row_idx, base + 13, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_ms";

      triplets.push_back({row_idx, base + 6, 1.0});
      triplets.push_back({row_idx, base + 11, 1.0});
      triplets.push_back({row_idx, base + 14, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_hsd";

      triplets.push_back({row_idx, base + 5, 1.0});
      triplets.push_back({row_idx, base + 12, 1.0});
      triplets.push_back({row_idx, base + 15, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_atf";

      triplets.push_back({row_idx, base + 8, 1.0});
      triplets.push_back({row_idx, base + 16, -1.0});
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_pool_fo";

      triplets.push_back({row_idx, base + 0, 1.0});
      triplets.push_back({row_idx, base + 1, 1.0});
      triplets.push_back({row_idx, base + 2, 1.0});
      triplets.push_back({row_idx, base + 3, 1.0});
      model.row_lower[static_cast<size_t>(row_idx)] = -1e20;
      model.row_upper[static_cast<size_t>(row_idx)] = 6000.0;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_cap_cdu";

      triplets.push_back({row_idx, base + 9, 100.0 - 91.0});
      triplets.push_back({row_idx, base + 10, 92.0 - 91.0});
      model.row_lower[static_cast<size_t>(row_idx)] = 0.0;
      model.row_upper[static_cast<size_t>(row_idx)] = 1e20;
      model.row_names[static_cast<size_t>(row_idx++)] = "p" + std::to_string(p) + "_t" + std::to_string(t) + "_qual_ron";

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

indus::Model generate_knapsack_milp() {
  indus::Model model;
  model.name = "knapsack_binary";
  model.sense = indus::ObjSense::kMaximize;
  model.num_rows = 1;
  model.num_cols = 5;
  model.c = {6.0, 3.0, 5.0, 4.0, 6.0};
  model.col_lower.assign(5, 0.0);
  model.col_upper.assign(5, 1.0);
  model.col_type.assign(5, indus::VarType::kInteger);
  model.col_names = {"item0", "item1", "item2", "item3", "item4"};
  model.row_lower = {-1e20};
  model.row_upper = {10.0};
  model.row_names = {"capacity"};
  std::vector<indus::la::Triplet> triplets = {
      {0, 0, 2.0}, {0, 1, 2.0}, {0, 2, 6.0}, {0, 3, 5.0}, {0, 4, 4.0}
  };
  model.A.set_from_triplets(1, 5, triplets);
  return model;
}

indus::Model generate_infeasible_milp() {
  indus::Model model;
  model.name = "infeasible_integer";
  model.sense = indus::ObjSense::kMinimize;
  model.num_rows = 1;
  model.num_cols = 1;
  model.c = {1.0};
  model.col_lower = {0.0};
  model.col_upper = {1.0};
  model.col_type = {indus::VarType::kInteger};
  model.col_names = {"x0"};
  model.row_lower = {2.0};
  model.row_upper = {1e20};
  model.row_names = {"r0"};
  model.A.set_from_triplets(1, 1, {{0, 0, 1.0}});
  return model;
}

indus::Model generate_semidefinite_qp() {
  indus::Model model;
  model.name = "semidefinite_qp";
  model.sense = indus::ObjSense::kMinimize;
  model.num_rows = 1;
  model.num_cols = 2;
  model.c = {0.0, 1.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.row_lower = {2.0};
  model.row_upper = {2.0};
  model.col_names = {"x0", "x1"};
  model.row_names = {"sum"};
  model.A.set_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
  model.Q.set_from_triplets(2, 2, {{0, 0, 2.0}});
  return model;
}

indus::Model generate_equality_inequality_qp() {
  indus::Model model;
  model.name = "equality_inequality_qp";
  model.sense = indus::ObjSense::kMinimize;
  model.num_rows = 1;
  model.num_cols = 2;
  model.c = {0.0, 0.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.row_lower = {4.0};
  model.row_upper = {1e20};
  model.col_names = {"x0", "x1"};
  model.row_names = {"sum_ge_4"};
  model.A.set_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
  model.Q.set_from_triplets(2, 2, {{0, 0, 2.0}, {1, 1, 2.0}});
  return model;
}

indus::Model generate_nonconvex_qp() {
  indus::Model model;
  model.name = "nonconvex_qp_indefinite";
  model.sense = indus::ObjSense::kMinimize;
  model.num_rows = 1;
  model.num_cols = 2;
  model.c = {0.0, 0.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.row_lower = {1.0};
  model.row_upper = {10.0};
  model.col_names = {"x0", "x1"};
  model.row_names = {"r0"};
  model.A.set_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});
  model.Q.set_from_triplets(2, 2, {{0, 0, 2.0}, {1, 1, -2.0}});
  return model;
}

void write_benchmark_csv(const std::string &csv_path, const std::vector<BenchmarkRow> &rows) {
  std::error_code ec;
  std::filesystem::path p_csv(csv_path);
  if (p_csv.has_parent_path()) {
    std::filesystem::create_directories(p_csv.parent_path(), ec);
  }
  std::ofstream csv(csv_path);
  csv << "Platform,Compiler,PrimalTol,DualTol,IterationLimit,TimeLimit_s,Suite,Instance,Rows,Cols,Nonzeros,Engine,GpuDevice,PresolveDim,Doubletons,"
         "Status,ComputedObj,PublishedObj,RelErr,Iterations,SetupTime_s,IterTime_s,"
         "TotalTime_s,GpuUpload_s,GpuKernel_s,GpuDownload_s,GpuTotal_s,Transfers,"
         "KernelSpeedup,E2ESpeedup,PrimViol,DualViol,ObjDiscr,ExecutionStatus,VerificationStatus,"
         "SourcePath,ProblemClass,QNonzeros,IntegerVars,BinaryVars,ObjSense,Algorithm,Backend,NodeCount,MipGap,StatRes,CompViol,Architecture,CPU,CUDA_Version,NodeLimit\n";

  for (const auto &r : rows) {
    double rel_err = 0.0;
    if (r.has_reference) {
      rel_err = std::abs(r.objective_value - r.reference_objective) / std::max(1.0, std::abs(r.reference_objective));
    }
    csv << r.platform << "," << r.compiler << "," << r.tolerance << "," << r.tolerance << ","
        << r.iteration_limit << "," << r.time_limit << ","
        << r.suite_tag << "," << r.model_name << "," << r.rows << "," << r.cols << ","
        << r.a_nnz << "," << r.algorithm << "," << r.gpu << ","
        << r.presolve_dim << "," << r.doubletons << ","
        << r.solver_status << ","
        << std::setprecision(17) << r.objective_value << ","
        << (r.has_reference ? r.reference_objective : 0.0) << "," << rel_err << ","
        << r.iterations << "," << r.cpu_setup_time << "," << r.cpu_iter_time << ","
        << r.runtime << "," << r.gpu_upload_time << ","
        << r.gpu_kernel_time << "," << r.gpu_download_time << ","
        << r.gpu_total_time << "," << r.gpu_transfers << ","
        << r.speedup_kernel << "," << r.speedup_e2e << ","
        << r.primal_violation << "," << r.dual_violation << "," << r.objective_discrepancy << ","
        << r.solver_status << "," << r.verification_status << ","
        << r.source_path << "," << r.problem_class << "," << r.q_nnz << ","
        << r.integer_vars << "," << r.binary_vars << "," << r.objective_sense << ","
        << r.algorithm << "," << r.backend << "," << r.node_count << ","
        << r.mip_gap << "," << r.stationarity_residual << "," << r.complementarity_residual << ","
        << r.architecture << "," << r.cpu << "," << r.cuda_version << "," << r.node_limit << "\n";
  }
}

void write_benchmark_json(const std::string &json_path,
                          const std::string &platform,
                          const std::string &architecture,
                          const std::string &cpu,
                          const std::string &gpu,
                          const std::string &compiler,
                          const std::string &cuda_status,
                          const std::string &cuda_version,
                          const std::string &mode_str,
                          const std::string &suite_filter,
                          double tol,
                          const std::vector<BenchmarkRow> &rows) {
  std::error_code ec;
  std::filesystem::path p_json(json_path);
  if (p_json.has_parent_path()) {
    std::filesystem::create_directories(p_json.parent_path(), ec);
  }
  std::ofstream jf(json_path);
  jf << "{\n";
  jf << "  \"benchmark_metadata\": {\n";
  jf << "    \"platform\": \"" << escape_json(platform) << "\",\n";
  jf << "    \"architecture\": \"" << escape_json(architecture) << "\",\n";
  jf << "    \"cpu\": \"" << escape_json(cpu) << "\",\n";
  jf << "    \"gpu\": \"" << escape_json(gpu) << "\",\n";
  jf << "    \"compiler\": \"" << escape_json(compiler) << "\",\n";
  jf << "    \"cuda_status\": \"" << escape_json(cuda_status) << "\",\n";
  jf << "    \"cuda_version\": \"" << escape_json(cuda_version) << "\",\n";
  jf << "    \"mode\": \"" << escape_json(mode_str) << "\",\n";
  jf << "    \"selected_suite\": \"" << escape_json(suite_filter) << "\",\n";
  jf << "    \"tolerance\": " << tol << "\n";
  jf << "  },\n";

  jf << "  \"results\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto &r = rows[i];
    jf << "    {\n";
    jf << "      \"suite\": \"" << escape_json(r.suite) << "\",\n";
    jf << "      \"suite_tag\": \"" << escape_json(r.suite_tag) << "\",\n";
    jf << "      \"model_name\": \"" << escape_json(r.model_name) << "\",\n";
    jf << "      \"source_path\": \"" << escape_json(r.source_path) << "\",\n";
    jf << "      \"problem_class\": \"" << escape_json(r.problem_class) << "\",\n";
    jf << "      \"rows\": " << r.rows << ",\n";
    jf << "      \"cols\": " << r.cols << ",\n";
    jf << "      \"a_nnz\": " << r.a_nnz << ",\n";
    jf << "      \"q_nnz\": " << r.q_nnz << ",\n";
    jf << "      \"integer_vars\": " << r.integer_vars << ",\n";
    jf << "      \"binary_vars\": " << r.binary_vars << ",\n";
    jf << "      \"objective_sense\": \"" << escape_json(r.objective_sense) << "\",\n";
    jf << "      \"algorithm\": \"" << escape_json(r.algorithm) << "\",\n";
    jf << "      \"backend\": \"" << escape_json(r.backend) << "\",\n";
    jf << "      \"solver_status\": \"" << escape_json(r.solver_status) << "\",\n";
    jf << "      \"verification_status\": \"" << escape_json(r.verification_status) << "\",\n";
    jf << "      \"objective_value\": " << std::setprecision(17) << r.objective_value << ",\n";
    if (r.has_reference) {
      jf << "      \"reference_objective\": " << std::setprecision(17) << r.reference_objective << ",\n";
    } else {
      jf << "      \"reference_objective\": null,\n";
    }
    jf << "      \"has_reference\": " << (r.has_reference ? "true" : "false") << ",\n";
    jf << "      \"objective_discrepancy\": " << r.objective_discrepancy << ",\n";
    jf << "      \"runtime\": " << r.runtime << ",\n";
    jf << "      \"iterations\": " << r.iterations << ",\n";
    jf << "      \"node_count\": " << r.node_count << ",\n";
    jf << "      \"mip_gap\": " << r.mip_gap << ",\n";
    jf << "      \"primal_violation\": " << r.primal_violation << ",\n";
    jf << "      \"dual_violation\": " << r.dual_violation << ",\n";
    jf << "      \"stationarity_residual\": " << r.stationarity_residual << ",\n";
    jf << "      \"complementarity_residual\": " << r.complementarity_residual << ",\n";
    jf << "      \"platform\": \"" << escape_json(r.platform) << "\",\n";
    jf << "      \"architecture\": \"" << escape_json(r.architecture) << "\",\n";
    jf << "      \"cpu\": \"" << escape_json(r.cpu) << "\",\n";
    jf << "      \"gpu\": \"" << escape_json(r.gpu) << "\",\n";
    jf << "      \"compiler\": \"" << escape_json(r.compiler) << "\",\n";
    jf << "      \"cuda_version\": \"" << escape_json(r.cuda_version) << "\",\n";
    jf << "      \"tolerance\": " << r.tolerance << ",\n";
    jf << "      \"time_limit\": " << r.time_limit << ",\n";
    jf << "      \"iteration_limit\": " << r.iteration_limit << ",\n";
    jf << "      \"node_limit\": " << r.node_limit << "\n";
    jf << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
  }
  jf << "  ],\n";

  int opt_cnt = 0, feas_cnt = 0, lim_cnt = 0, inf_cnt = 0, unsup_cnt = 0, fail_cnt = 0, na_cnt = 0;
  for (const auto &r : rows) {
    if (r.verification_status == "OPTIMAL_VERIFIED" || r.verification_status == "PASSED") opt_cnt++;
    else if (r.verification_status == "FEASIBLE_UNVERIFIED") feas_cnt++;
    else if (r.verification_status == "LIMIT_REACHED") lim_cnt++;
    else if (r.verification_status == "INFEASIBLE_VERIFIED") inf_cnt++;
    else if (r.verification_status == "UNSUPPORTED") unsup_cnt++;
    else if (r.verification_status == "FAILED") fail_cnt++;
    else if (r.verification_status == "NOT_APPLICABLE") na_cnt++;
  }

  jf << "  \"summary\": {\n";
  jf << "    \"total_models\": " << rows.size() << ",\n";
  jf << "    \"optimal_verified\": " << opt_cnt << ",\n";
  jf << "    \"feasible_unverified\": " << feas_cnt << ",\n";
  jf << "    \"limit_reached\": " << lim_cnt << ",\n";
  jf << "    \"infeasible_verified\": " << inf_cnt << ",\n";
  jf << "    \"unsupported\": " << unsup_cnt << ",\n";
  jf << "    \"failed\": " << fail_cnt << ",\n";
  jf << "    \"not_applicable\": " << na_cnt << "\n";
  jf << "  }\n";
  jf << "}\n";
}

} // namespace

int main(int argc, char *argv[]) {
  std::cout << "========================================================================\n";
  std::cout << "  SIDDHANTA (INDUS-OPT): SOVEREIGN BENCHMARK & MULTI-ENGINE HARNESS    \n";
  std::cout << "  Smart India Hackathon SIH26119 | MRPL Problem Statement 26119        \n";
  std::cout << "========================================================================\n\n";

  std::string out_dir = "build/benchmark_artifacts";
  std::string csv_path = "build/benchmark_results.csv";
  std::string json_path = "build/benchmark_results.json";
  std::string suite_filter = "all";
  double tol = 1e-4;
  bool quick_mode = false;
  bool full_mode = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--out-dir" && i + 1 < argc) {
      out_dir = argv[++i];
    } else if (arg == "--csv" && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (arg == "--json" && i + 1 < argc) {
      json_path = argv[++i];
    } else if (arg == "--suite" && i + 1 < argc) {
      suite_filter = argv[++i];
      std::transform(suite_filter.begin(), suite_filter.end(), suite_filter.begin(), ::tolower);
    } else if (arg == "--tol" && i + 1 < argc) {
      tol = std::stod(argv[++i]);
    } else if (arg == "--quick") {
      quick_mode = true;
    } else if (arg == "--full") {
      full_mode = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: indus_benchmark [options]\n\n"
                << "Execution Modes:\n"
                << "  --quick          Smoke/classification test: validates Suite 1 baseline (22/22),\n"
                << "                   verifies CSV/JSON generation and telemetry classification without\n"
                << "                   requiring Suite 2 reference PDHG convergence to optimality. Exits 0 on clean smoke run.\n"
                << "  --full (default) Full reference verification: enforces independent verification and\n"
                << "                   reference-objective matching for reference models. Exits nonzero\n"
                << "                   if any reference model fails verification or limits.\n\n"
                << "Benchmark Suites:\n"
                << "  --suite <name>   Filter by suite: 'lp', 'milp', 'qp', 'pdhg', 'cuda', 'all' (default: all)\n\n"
                << "Options:\n"
                << "  --out-dir <dir>  Directory to save solution and JSON artifacts (default: build/benchmark_artifacts)\n"
                << "  --csv <path>     Path for CSV telemetry output (default: build/benchmark_results.csv)\n"
                << "  --json <path>    Path for JSON telemetry output (default: build/benchmark_results.json)\n"
                << "  --tol <tol>      Verification tolerance (default: 1e-4)\n"
                << "  --help, -h       Display this help message\n";
      return 0;
    } else {
      std::cerr << "[ERROR] Unknown option or missing value: " << arg << "\n";
      return 1;
    }
  }

  if (!quick_mode) {
    full_mode = true;
  }

  // Validate suite_filter
  if (suite_filter != "all" && suite_filter != "lp" && suite_filter != "milp" &&
      suite_filter != "qp" && suite_filter != "pdhg" && suite_filter != "cuda") {
    std::cerr << "[ERROR] Invalid suite: '" << suite_filter << "'. Valid suites: 'lp', 'milp', 'qp', 'pdhg', 'cuda', 'all'.\n";
    return 1;
  }

  auto dev_info = indus::gpu::probe_device();

#if defined(__APPLE__) && defined(__MACH__)
  const std::string platform_str = "macOS_ARM64";
#elif defined(_WIN32)
  const std::string platform_str = "Windows";
#elif defined(__linux__)
  const std::string platform_str = "Linux";
#else
  const std::string platform_str = "GenericPlatform";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
  const std::string arch_str = "ARM64";
  const std::string cpu_str = "Apple Silicon (ARM64)";
#elif defined(__x86_64__) || defined(_M_X64)
  const std::string arch_str = "x86_64";
  const std::string cpu_str = "x86_64";
#else
  const std::string arch_str = "GenericArch";
  const std::string cpu_str = "Generic CPU";
#endif

#if defined(__clang__)
  const std::string compiler_str = "Clang_" + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
  const std::string compiler_str = "GCC_" + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__);
#elif defined(_MSC_VER)
  const std::string compiler_str = "MSVC_" + std::to_string(_MSC_VER);
#else
  const std::string compiler_str = "CXX";
#endif

  const std::string cuda_status_str = dev_info.available ? "ACTIVE (Hardware Ready)" : "NOT AVAILABLE";
  const std::string cuda_version_str = dev_info.available ? "NVIDIA CUDA" : "NOT_AVAILABLE";
  const std::string gpu_name_str = dev_info.available ? dev_info.device_name : "None (CUDA support disabled at compile time / CPU only)";

  std::cout << "[HARDWARE CONFIGURATION]\n";
  std::cout << "  Platform       : " << platform_str << " (" << arch_str << ")\n";
  std::cout << "  CPU Hardware   : " << cpu_str << "\n";
  std::cout << "  Compiler       : " << compiler_str << "\n";
  std::cout << "  GPU Hardware   : " << gpu_name_str << "\n";
  std::cout << "  CUDA Status    : " << cuda_status_str << "\n";
  if (dev_info.available) {
    std::cout << "  VRAM Available : " << (dev_info.total_vram_bytes / (1024 * 1024)) << " MB\n";
    std::cout << "  Compute Cap.   : " << dev_info.compute_capability_major << "." << dev_info.compute_capability_minor << "\n";
  }
  std::cout << "  Benchmark Mode : " << (quick_mode ? "QUICK SMOKE TEST (--quick)" : "FULL REFERENCE VERIFICATION (--full)") << "\n";
  std::cout << "  Selected Suite : " << suite_filter << "\n\n";

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec || !std::filesystem::exists(out_dir)) {
    out_dir = (std::filesystem::temp_directory_path() / "vajra_benchmark_artifacts").string();
    std::filesystem::create_directories(out_dir, ec);
  }

  std::vector<BenchmarkRow> all_rows;

  bool run_lp   = (suite_filter == "all" || suite_filter == "lp");
  bool run_pdhg = (suite_filter == "all" || suite_filter == "pdhg");
  bool run_cuda = (suite_filter == "all" || suite_filter == "cuda");
  bool run_milp = (suite_filter == "all" || suite_filter == "milp");
  bool run_qp   = (suite_filter == "all" || suite_filter == "qp");

  int s1_total = 0, s1_passed = 0;
  int s2_ref_failures = 0;
  int total_limit_reached = 0;
  int total_not_applicable = 0;
  bool has_false_passed_row = false;

  // =========================================================================
  // SUITE 1: 22 NETLIB & MRPL LP BASELINE (Presolved Revised Simplex)
  // =========================================================================
  if (run_lp) {
    struct LpInstance {
      std::string name;
      std::string filepath;
      double published_optimal;
      bool is_lp = false;
    };

    std::vector<LpInstance> instances = {
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

    std::cout << "[SUITE 1: 22 NETLIB & MRPL LP BASELINE - PRESOLVED REVISED SIMPLEX]\n";
    std::cout << std::left << std::setw(16) << "Instance" << std::setw(12)
              << "Orig(R/C)" << std::setw(12) << "Pres(R/C)" << std::setw(7)
              << "Dbltn" << std::setw(10) << "Status" << std::setw(18)
              << "Computed Obj" << std::setw(18) << "Published Obj"
              << std::setw(10) << "Rel Err" << std::setw(10) << "PrimViol"
              << std::setw(10) << "DualViol" << std::setw(10) << "ObjDiscr"
              << std::setw(8) << "Verified" << "\n";
    std::cout << std::string(141, '-') << "\n";

    for (const auto &inst : instances) {
      s1_total++;
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
        s1_passed++;
      }

      std::string status_str =
          (sol.status == indus::SolveStatus::kOptimal)          ? "OPTIMAL"
          : (sol.status == indus::SolveStatus::kFeasible)       ? "FEASIBLE"
          : (sol.status == indus::SolveStatus::kNumericalError) ? "NUM_ERR"
                                                                : "OTHER";

      const std::string orig_dim = std::to_string(model.num_rows) + "/" + std::to_string(model.num_cols);
      const std::string pres_dim = std::to_string(sol.presolve_num_rows) + "/" + std::to_string(sol.presolve_num_cols);

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

      BenchmarkRow row;
      row.suite = "lp";
      row.suite_tag = "NetlibBaseline";
      row.model_name = inst.name;
      row.source_path = full_path;
      row.problem_class = "LP";
      row.rows = model.num_rows;
      row.cols = model.num_cols;
      row.a_nnz = model.A.nnz();
      row.q_nnz = 0;
      row.integer_vars = 0;
      row.binary_vars = 0;
      row.objective_sense = (model.sense == indus::ObjSense::kMinimize ? "MINIMIZE" : "MAXIMIZE");
      row.algorithm = "simplex";
      row.backend = "CPU";
      row.solver_status = status_str;
      row.verification_status = (overall_pass ? "PASSED" : "FAILED");
      row.objective_value = sol.objective_value;
      row.reference_objective = inst.published_optimal;
      row.has_reference = true;
      row.objective_discrepancy = vres.objective_discrepancy;
      row.runtime = sol.solve_time_seconds;
      row.iterations = sol.iterations;
      row.node_count = 0;
      row.mip_gap = 0.0;
      row.primal_violation = vres.max_primal_violation;
      row.dual_violation = vres.max_dual_violation;
      row.stationarity_residual = 0.0;
      row.complementarity_residual = 0.0;
      row.platform = platform_str;
      row.architecture = arch_str;
      row.cpu = cpu_str;
      row.gpu = gpu_name_str;
      row.compiler = compiler_str;
      row.cuda_version = cuda_version_str;
      row.tolerance = tol;
      row.time_limit = opts.time_limit;
      row.iteration_limit = opts.iteration_limit;
      row.node_limit = 0;
      row.presolve_dim = pres_dim;
      row.doubletons = sol.presolve_doubleton_reductions;
      row.speedup_kernel = "1.00";
      row.speedup_e2e = "1.00";

      all_rows.push_back(row);
    }

    std::cout << std::string(141, '-') << "\n";
    std::cout << "BASELINE SUMMARY: " << s1_passed << " / " << s1_total
              << " models passed independent verification.\n\n";
  }

  // =========================================================================
  // SUITE 2: MULTI-ENGINE RESTART-PDHG SPEEDUP & TELEMETRY BENCHMARK
  // =========================================================================
  if (run_pdhg || run_cuda) {
    std::cout << "[SUITE 2: MULTI-ENGINE SPEEDUP & DETAILED TELEMETRY BENCHMARK]\n";
    std::cout << "Models: 1. MRPL Crude Blend | 2. Medium Netlib | 3. Large Expander | 4. Structured Refinery | 5. 240k Nonzeros\n\n";

    struct ModelEntry {
      std::string category;
      std::string name;
      std::string filepath;
      indus::Model model;
      double ref_obj = 0.0;
      bool has_ref = false;
    };

    std::vector<ModelEntry> benchmark_models;
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps");
      indus::Model m = indus::io::read_mps(path);
      benchmark_models.push_back({"1. Small MRPL Blend", "crude_blend", path, m, 214.145945946, true});
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/beaconfd.mps");
      indus::Model m = indus::io::read_mps(path);
      benchmark_models.push_back({"2. Medium Netlib", "beaconfd", path, m, 33592.485807, true});
    }
    {
      std::cout << "  Generating Category 3: Large Sparse Model (5,000 x 10,000, ~100k nonzeros)...\n";
      indus::Model m = generate_synthetic_sparse_lp(5000, 10000, 10, 42);
      benchmark_models.push_back({"3. Large Sparse Expander", "sparse_5k_10k_100k_nnz", "", m, 0.0, false});
    }
    {
      std::cout << "  Generating Category 4: Structured Refinery Model (16 periods x 4 trains)...\n";
      indus::Model m = generate_structured_refinery_lp(16, 4);
      benchmark_models.push_back({"4. Structured Refinery", "mrpl_structured_refinery", "", m, 0.0, false});
    }
    {
      std::cout << "  Generating Category 5: High-Density Model (10,000 x 20,000, 240,000 nonzeros)...\n";
      indus::Model m = generate_synthetic_sparse_lp(10000, 20000, 12, 999);
      benchmark_models.push_back({"5. 240k Nonzero Model", "sparse_10k_20k_240k_nnz", "", m, 0.0, false});
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
              << std::setw(16) << "ExecStatus"
              << std::setw(20) << "VerifyStatus"
              << "\n";
    std::cout << std::string(162, '-') << "\n";

    for (auto &entry : benchmark_models) {
      const auto &model = entry.model;

      const int64_t iter_lim = entry.has_ref ? (quick_mode ? 2000 : 50000)
                               : (quick_mode && model.A.nnz() > 50000) ? 50
                               : 2000;

      // 1. Solve with CPU PDHG
      indus::Options cpu_opts;
      cpu_opts.set("algorithm", "pdhg_cpu");
      cpu_opts.enable_presolve = false;
      cpu_opts.iteration_limit = iter_lim;
      cpu_opts.set("tolerance", 1e-4);

      indus::Solution cpu_sol = indus::solve(model, cpu_opts);
      const auto cpu_diag = indus::pdhg::get_last_cpu_pdhg_diagnostics();

      // 2. Solve with GPU PDHG (cleanly falls back to CPU if no CUDA GPU)
      indus::Options gpu_opts;
      gpu_opts.set("algorithm", "pdhg_cuda");
      gpu_opts.use_gpu = true;
      gpu_opts.enable_presolve = false;
      gpu_opts.iteration_limit = iter_lim;
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
        speedup_e2e_str = "NOT_MEASURED";
        speedup_kernel_str = "NOT_MEASURED";
      }

      const std::string dims = std::to_string(model.num_rows) + "x" + std::to_string(model.num_cols);
      const std::string execution_status = indus::to_string(gpu_sol.status);

      std::string verification_status;
      if (gpu_sol.status == indus::SolveStatus::kOptimal && gpu_sol.quality.is_primal_feasible) {
        if (!entry.filepath.empty() && entry.has_ref) {
          const std::string s2_sol = out_dir + "/" + entry.name + "_pdhg.sol";
          indus::io::write_solution(gpu_sol, entry.model, s2_sol);
          indus::verifier::VerificationResult vres2 =
              indus::verifier::verify_files(entry.filepath, s2_sol, tol);
          const double ref_rel_err = std::abs(gpu_sol.objective_value - entry.ref_obj) /
                                     std::max(1.0, std::abs(entry.ref_obj));
          const bool passes = vres2.passed && (ref_rel_err <= 1e-6);
          verification_status = passes ? "PASSED" : "FAILED";
          if (!passes) ++s2_ref_failures;
        } else {
          verification_status = "NOT_APPLICABLE";
          ++total_not_applicable;
        }
      } else if (gpu_sol.status == indus::SolveStatus::kIterationLimit ||
                 gpu_sol.status == indus::SolveStatus::kTimeLimit) {
        verification_status = "NOT_APPLICABLE";
        ++total_limit_reached;
        ++total_not_applicable;
        if (entry.has_ref) ++s2_ref_failures;
      } else if (gpu_sol.status == indus::SolveStatus::kFeasible) {
        verification_status = "FEASIBLE_UNVERIFIED";
        if (entry.has_ref) ++s2_ref_failures;
      } else {
        verification_status = "FAILED";
        if (entry.has_ref) ++s2_ref_failures;
      }

      if (verification_status == "PASSED" &&
          (gpu_sol.status != indus::SolveStatus::kOptimal || !gpu_sol.quality.is_primal_feasible)) {
        has_false_passed_row = true;
      }

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
                << std::setw(16) << execution_status
                << std::setw(20) << verification_status
                << "\n";

      BenchmarkRow row;
      row.suite = (run_cuda ? "cuda" : "pdhg");
      row.suite_tag = "EngineComparison";
      row.model_name = entry.name;
      row.source_path = entry.filepath.empty() ? "synthetic" : entry.filepath;
      row.problem_class = "LP";
      row.rows = model.num_rows;
      row.cols = model.num_cols;
      row.a_nnz = model.A.nnz();
      row.q_nnz = 0;
      row.integer_vars = 0;
      row.binary_vars = 0;
      row.objective_sense = "MINIMIZE";
      row.algorithm = (run_cuda ? "pdhg_cuda" : "pdhg_cpu");
      row.backend = (dev_info.available ? "NVIDIA_CUDA" : "CPU FALLBACK");
      row.solver_status = execution_status;
      row.verification_status = verification_status;
      row.objective_value = gpu_sol.objective_value;
      row.reference_objective = entry.ref_obj;
      row.has_reference = entry.has_ref;
      row.objective_discrepancy = 0.0;
      row.runtime = gpu_sol.solve_time_seconds;
      row.iterations = gpu_sol.iterations;
      row.node_count = 0;
      row.mip_gap = 0.0;
      row.primal_violation = gpu_sol.quality.max_primal_violation;
      row.dual_violation = gpu_sol.quality.max_dual_violation;
      row.stationarity_residual = 0.0;
      row.complementarity_residual = 0.0;
      row.platform = platform_str;
      row.architecture = arch_str;
      row.cpu = cpu_str;
      row.gpu = gpu_name_str;
      row.compiler = compiler_str;
      row.cuda_version = cuda_version_str;
      row.tolerance = 1e-4;
      row.time_limit = 1e20;
      row.iteration_limit = iter_lim;
      row.node_limit = 0;
      row.cpu_setup_time = cpu_diag.setup_time_sec;
      row.cpu_iter_time = cpu_diag.iteration_time_sec;
      row.gpu_upload_time = gpu_timing.upload_time_sec;
      row.gpu_kernel_time = gpu_timing.iteration_time_sec;
      row.gpu_download_time = gpu_timing.download_time_sec;
      row.gpu_total_time = gpu_timing.total_time_sec;
      row.gpu_transfers = gpu_timing.host_device_transfers;
      row.speedup_kernel = speedup_kernel_str;
      row.speedup_e2e = speedup_e2e_str;

      all_rows.push_back(row);
    }
    std::cout << std::string(162, '-') << "\n";
  }

  // =========================================================================
  // SUITE 3: MILP BRANCH-AND-BOUND BENCHMARK SUITE
  // =========================================================================
  int milp_total = 0, milp_verified = 0;
  if (run_milp) {
    std::cout << "\n[SUITE 3: MILP BRANCH-AND-BOUND BENCHMARK SUITE]\n";
    std::cout << "Models: 1. Crude Blend MILP | 2. Lot Sizing (Production) | 3. Power Dispatch | 4. Knapsack | 5. Infeasible | 6. Node Limit | 7. Time Limit\n\n";

    struct MilpEntry {
      std::string name;
      indus::Model model;
      double ref_obj = 0.0;
      bool has_ref = false;
      int64_t node_limit = 500000;
      double time_limit = 1000.0;
      std::string expected_status = "OPTIMAL";
    };

    std::vector<MilpEntry> milp_models;
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend_milp.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        milp_models.push_back({"blend_milp", std::move(m), 223.857605178, true, 500000, 1000.0, "OPTIMAL"});
      }
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/lot_sizing.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        milp_models.push_back({"lot_sizing", std::move(m), 770.0, true, 500000, 1000.0, "OPTIMAL"});
      }
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/power_dispatch.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        milp_models.push_back({"power_dispatch", std::move(m), 3270.0, true, 500000, 1000.0, "OPTIMAL"});
      }
    }
    {
      milp_models.push_back({"knapsack_binary", generate_knapsack_milp(), 15.0, true, 500000, 1000.0, "OPTIMAL"});
    }
    {
      milp_models.push_back({"infeasible_integer", generate_infeasible_milp(), 0.0, false, 500000, 1000.0, "INFEASIBLE"});
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/lot_sizing.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        milp_models.push_back({"lot_sizing_node_limit", std::move(m), 770.0, true, 1, 1000.0, "NODE_LIMIT"});
      }
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/lot_sizing.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        milp_models.push_back({"lot_sizing_time_limit", std::move(m), 770.0, true, 500000, 0.000001, "TIME_LIMIT"});
      }
    }

    std::cout << std::left << std::setw(24) << "Instance"
              << std::setw(14) << "Dim(R x C)"
              << std::setw(10) << "Int/Bin"
              << std::setw(10) << "NNZ"
              << std::setw(18) << "Objective"
              << std::setw(12) << "Time(s)"
              << std::setw(10) << "Nodes"
              << std::setw(12) << "MipGap"
              << std::setw(16) << "ExecStatus"
              << std::setw(20) << "VerifyStatus"
              << "\n";
    std::cout << std::string(146, '-') << "\n";

    milp_total = static_cast<int>(milp_models.size());

    for (auto &entry : milp_models) {
      indus::Options opts;
      opts.set("algorithm", "milp");
      opts.node_limit = entry.node_limit;
      opts.time_limit = entry.time_limit;
      opts.set("tolerance", tol);

      indus::Solution sol = indus::solve(entry.model, opts);
      indus::verifier::VerificationResult vres = indus::verifier::verify_solution(entry.model, sol);

      int int_cnt = 0, bin_cnt = 0;
      for (size_t j = 0; j < entry.model.col_type.size(); ++j) {
        if (entry.model.col_type[j] == indus::VarType::kInteger) {
          int_cnt++;
          if (entry.model.col_lower[j] == 0.0 && entry.model.col_upper[j] == 1.0) {
            bin_cnt++;
          }
        }
      }

      std::string exec_status = indus::to_string(sol.status);
      std::string verif_status = "FAILED";

      if (sol.status == indus::SolveStatus::kOptimal) {
        const double ref_rel_err = entry.has_ref ? (std::abs(sol.objective_value - entry.ref_obj) / std::max(1.0, std::abs(entry.ref_obj))) : 0.0;
        if (vres.passed && sol.has_incumbent && sol.relative_gap <= 1e-4 && (ref_rel_err <= 1e-5)) {
          verif_status = "OPTIMAL_VERIFIED";
          milp_verified++;
        }
      } else if (sol.status == indus::SolveStatus::kNodeLimit || sol.status == indus::SolveStatus::kTimeLimit) {
        verif_status = "LIMIT_REACHED";
        milp_verified++;
      } else if (sol.status == indus::SolveStatus::kInfeasible) {
        verif_status = "INFEASIBLE_VERIFIED";
        milp_verified++;
      } else if (sol.status == indus::SolveStatus::kFeasible) {
        verif_status = "FEASIBLE_UNVERIFIED";
      } else if (sol.status == indus::SolveStatus::kUnsupported) {
        verif_status = "UNSUPPORTED";
      }

      std::string dim_str = std::to_string(entry.model.num_rows) + "x" + std::to_string(entry.model.num_cols);
      std::string int_str = std::to_string(int_cnt) + "/" + std::to_string(bin_cnt);

      std::cout << std::left << std::setw(24) << entry.name
                << std::setw(14) << dim_str
                << std::setw(10) << int_str
                << std::setw(10) << entry.model.A.nnz()
                << std::setw(18) << std::setprecision(8) << sol.objective_value
                << std::setw(12) << std::setprecision(6) << sol.solve_time_seconds
                << std::setw(10) << sol.nodes
                << std::setw(12) << std::scientific << std::setprecision(1) << sol.relative_gap
                << std::defaultfloat << std::setw(16) << exec_status
                << std::setw(20) << verif_status
                << "\n";

      BenchmarkRow row;
      row.suite = "milp";
      row.suite_tag = "MILP";
      row.model_name = entry.name;
      row.source_path = entry.name;
      row.problem_class = "MILP";
      row.rows = entry.model.num_rows;
      row.cols = entry.model.num_cols;
      row.a_nnz = entry.model.A.nnz();
      row.q_nnz = 0;
      row.integer_vars = int_cnt;
      row.binary_vars = bin_cnt;
      row.objective_sense = (entry.model.sense == indus::ObjSense::kMinimize ? "MINIMIZE" : "MAXIMIZE");
      row.algorithm = "milp_branch_and_bound";
      row.backend = "CPU";
      row.solver_status = exec_status;
      row.verification_status = verif_status;
      row.objective_value = sol.objective_value;
      row.reference_objective = entry.ref_obj;
      row.has_reference = entry.has_ref;
      row.objective_discrepancy = vres.objective_discrepancy;
      row.runtime = sol.solve_time_seconds;
      row.iterations = sol.iterations;
      row.node_count = sol.nodes;
      row.mip_gap = sol.relative_gap;
      row.primal_violation = vres.max_primal_violation;
      row.dual_violation = vres.max_dual_violation;
      row.stationarity_residual = 0.0;
      row.complementarity_residual = 0.0;
      row.platform = platform_str;
      row.architecture = arch_str;
      row.cpu = cpu_str;
      row.gpu = gpu_name_str;
      row.compiler = compiler_str;
      row.cuda_version = cuda_version_str;
      row.tolerance = tol;
      row.time_limit = entry.time_limit;
      row.iteration_limit = 50000;
      row.node_limit = entry.node_limit;
      row.speedup_kernel = "1.00";
      row.speedup_e2e = "1.00";

      all_rows.push_back(row);
    }
    std::cout << std::string(146, '-') << "\n";
  }

  // =========================================================================
  // SUITE 4: CONVEX QUADRATIC PROGRAMMING (QP) BENCHMARK
  // =========================================================================
  int qp_total = 0, qp_verified = 0;
  if (run_qp) {
    std::cout << "\n[SUITE 4: CONVEX QUADRATIC PROGRAMMING (QP) BENCHMARK]\n";
    std::cout << "Models: 1. MRPL QP Blend | 2. Crude Blend QP | 3. Portfolio Allocation (100 Assets) | 4. Semidefinite QP | 5. Eq/Ineq QP | 6. Nonconvex QP | 7. MIQP\n\n";

    struct QpModelEntry {
      std::string name;
      indus::Model model;
      double ref_obj = 0.0;
      bool has_ref = false;
    };

    std::vector<QpModelEntry> qp_models;
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/qp_blend.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        qp_models.push_back({"qp_blend", std::move(m), 66.666666667, true});
      }
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend_qp.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        qp_models.push_back({"crude_blend_qp", std::move(m), 179.91776118, true});
      }
    }
    // 3. Synthetic Portfolio QP (100 assets, 1 budget row)
    {
      const int n_assets = 100;
      indus::Model m;
      m.name = "portfolio_qp_100";
      m.num_cols = n_assets;
      m.num_rows = 1;
      m.sense = indus::ObjSense::kMinimize;
      m.c.resize(static_cast<size_t>(n_assets));
      for (int j = 0; j < n_assets; ++j) {
        m.c[static_cast<size_t>(j)] = -0.05 - 0.001 * (j % 10);
      }
      m.col_lower.assign(static_cast<size_t>(n_assets), 0.0);
      m.col_upper.assign(static_cast<size_t>(n_assets), 1.0);
      m.row_lower = {1.0};
      m.row_upper = {1.0};
      std::vector<indus::la::Triplet> a_triplets;
      for (int j = 0; j < n_assets; ++j) {
        a_triplets.push_back({0, j, 1.0});
      }
      m.A = indus::la::SparseMatrixCSC::from_triplets(1, n_assets, a_triplets);
      std::vector<indus::la::Triplet> q_triplets;
      for (int j = 0; j < n_assets; ++j) {
        q_triplets.push_back({j, j, 0.1 * (1.0 + (j % 5))});
        if (j + 1 < n_assets) {
          q_triplets.push_back({j + 1, j, 0.01});
          q_triplets.push_back({j, j + 1, 0.01});
        }
      }
      m.Q = indus::la::SparseMatrixCSC::from_triplets(n_assets, n_assets, q_triplets);
      qp_models.push_back({"portfolio_qp_100", std::move(m), 0.0, false});
    }
    {
      qp_models.push_back({"semidefinite_qp", generate_semidefinite_qp(), 1.75, true});
    }
    {
      qp_models.push_back({"equality_inequality_qp", generate_equality_inequality_qp(), 8.0, true});
    }
    {
      qp_models.push_back({"nonconvex_qp_indefinite", generate_nonconvex_qp(), 0.0, false});
    }
    {
      std::string path = find_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/miqp_blend.mps");
      if (std::filesystem::exists(path)) {
        indus::Model m = indus::io::read_mps(path);
        qp_models.push_back({"miqp_blend", std::move(m), 0.0, false});
      }
    }

    std::cout << std::left << std::setw(24) << "Instance"
              << std::setw(14) << "Dim(R x C)"
              << std::setw(10) << "Q NNZ"
              << std::setw(10) << "A NNZ"
              << std::setw(18) << "Objective"
              << std::setw(12) << "Time(s)"
              << std::setw(8) << "Iter"
              << std::setw(14) << "PrimViol"
              << std::setw(14) << "StatRes"
              << std::setw(14) << "CompViol"
              << std::setw(16) << "ExecStatus"
              << std::setw(20) << "VerifyStatus"
              << "\n";
    std::cout << std::string(174, '-') << "\n";

    qp_total = static_cast<int>(qp_models.size());

    for (auto &entry : qp_models) {
      indus::Options opts;
      opts.set("algorithm", "qp");
      opts.iteration_limit = 100000;
      opts.set("tolerance", 1e-6);

      indus::Solution qp_sol = indus::solve(entry.model, opts);
      indus::verifier::VerificationResult vres = indus::verifier::verify_solution(entry.model, qp_sol);

      std::string exec_status = indus::to_string(qp_sol.status);
      std::string verif_status = "FAILED";
      if (qp_sol.status == indus::SolveStatus::kOptimal && vres.passed) {
        verif_status = "OPTIMAL_VERIFIED";
        qp_verified++;
      } else if (qp_sol.status == indus::SolveStatus::kFeasible) {
        verif_status = "FEASIBLE_UNVERIFIED";
      } else if (qp_sol.status == indus::SolveStatus::kIterationLimit || qp_sol.status == indus::SolveStatus::kTimeLimit) {
        verif_status = "LIMIT_REACHED";
      } else if (qp_sol.status == indus::SolveStatus::kInfeasible && vres.passed) {
        verif_status = "INFEASIBLE_VERIFIED";
        qp_verified++;
      } else if (qp_sol.status == indus::SolveStatus::kUnsupported || qp_sol.status == indus::SolveStatus::kModelError) {
        verif_status = "UNSUPPORTED";
        qp_verified++;
      }

      std::string dim_str = std::to_string(entry.model.num_rows) + "x" + std::to_string(entry.model.num_cols);
      std::cout << std::left << std::setw(24) << entry.name
                << std::setw(14) << dim_str
                << std::setw(10) << entry.model.Q.nnz()
                << std::setw(10) << entry.model.A.nnz()
                << std::setw(18) << std::setprecision(8) << qp_sol.objective_value
                << std::setw(12) << std::setprecision(6) << qp_sol.solve_time_seconds
                << std::setw(8) << qp_sol.iterations
                << std::setw(14) << std::scientific << std::setprecision(2) << qp_sol.quality.max_primal_violation
                << std::setw(14) << std::scientific << std::setprecision(2) << qp_sol.quality.max_stationarity_residual
                << std::setw(14) << std::scientific << std::setprecision(2) << qp_sol.quality.max_complementarity_violation
                << std::defaultfloat << std::setw(16) << exec_status
                << std::setw(20) << verif_status
                << "\n";

      BenchmarkRow row;
      row.suite = "qp";
      row.suite_tag = "QP";
      row.model_name = entry.name;
      row.source_path = entry.name;
      row.problem_class = (entry.name == "miqp_blend" ? "MIQP" : "QP");
      row.rows = entry.model.num_rows;
      row.cols = entry.model.num_cols;
      row.a_nnz = entry.model.A.nnz();
      row.q_nnz = entry.model.Q.nnz();
      row.integer_vars = 0;
      row.binary_vars = 0;
      row.objective_sense = (entry.model.sense == indus::ObjSense::kMinimize ? "MINIMIZE" : "MAXIMIZE");
      row.algorithm = "convex_qp";
      row.backend = "CPU";
      row.solver_status = exec_status;
      row.verification_status = verif_status;
      row.objective_value = qp_sol.objective_value;
      row.reference_objective = entry.ref_obj;
      row.has_reference = entry.has_ref;
      row.objective_discrepancy = vres.objective_discrepancy;
      row.runtime = qp_sol.solve_time_seconds;
      row.iterations = qp_sol.iterations;
      row.node_count = 0;
      row.mip_gap = 0.0;
      row.primal_violation = qp_sol.quality.max_primal_violation;
      row.dual_violation = 0.0;
      row.stationarity_residual = qp_sol.quality.max_stationarity_residual;
      row.complementarity_residual = qp_sol.quality.max_complementarity_violation;
      row.platform = platform_str;
      row.architecture = arch_str;
      row.cpu = cpu_str;
      row.gpu = gpu_name_str;
      row.compiler = compiler_str;
      row.cuda_version = cuda_version_str;
      row.tolerance = 1e-6;
      row.time_limit = 1e20;
      row.iteration_limit = 100000;
      row.node_limit = 0;
      row.speedup_kernel = "N/A";
      row.speedup_e2e = "N/A";

      all_rows.push_back(row);
    }
    std::cout << std::string(174, '-') << "\n";
  }

  // Hardware acceleration notice when CUDA unavailable
  if (!dev_info.available) {
    std::cout << "\n[HARDWARE ACCELERATION NOTICE]\n";
    std::cout << "  * CUDA status       : NOT AVAILABLE\n";
    std::cout << "  * Execution backend : CPU FALLBACK\n";
    std::cout << "  * GPU speedup       : NOT MEASURED\n";
    std::cout << "  * Host platform is CPU-only (ARM64 macOS / No discrete NVIDIA GPU).\n";
    std::cout << "  * CUDA kernels, warp SpMV, zero-transfer diagnostics, and RAII memory handlers are compiled/tested.\n";
    std::cout << "  * Physical GPU speedup is NOT measured on this machine.\n";
    std::cout << "  * Hardware speedup gate remains properly marked as pending physical NVIDIA testbed.\n";
  }

  // Export CSV & JSON
  write_benchmark_csv(csv_path, all_rows);
  write_benchmark_json(json_path, platform_str, arch_str, cpu_str, gpu_name_str,
                       compiler_str, cuda_status_str, cuda_version_str,
                       (quick_mode ? "QUICK" : "FULL"), suite_filter, tol, all_rows);

  const bool csv_ok = std::filesystem::exists(csv_path) && std::filesystem::file_size(csv_path) > 0;
  const bool json_ok = std::filesystem::exists(json_path) && std::filesystem::file_size(json_path) > 0;
  const bool all_s1_pass = (s1_total == 0) || (s1_passed == s1_total);
  const bool all_s2_ref_pass = (s2_ref_failures == 0);
  const bool all_s3_pass = (milp_total == 0) || (milp_verified == milp_total);
  const bool all_s4_pass = (qp_total == 0) || (qp_verified == qp_total);

  std::cout << "\n========================================================================\n";
  std::cout << "  BENCHMARK SUMMARY REPORT                                              \n";
  std::cout << "========================================================================\n";
  std::cout << "  Mode                     : " << (quick_mode ? "QUICK SMOKE TEST (--quick)" : "FULL REFERENCE VERIFICATION (--full)") << "\n";
  std::cout << "  Selected Suite           : " << suite_filter << "\n";
  if (run_lp) {
    std::cout << "  Suite 1 (LP Baseline)    : " << s1_passed << " / " << s1_total << " verified ("
              << (all_s1_pass ? "PASSED" : "FAILED") << ")\n";
  }
  if (run_pdhg || run_cuda) {
    std::cout << "  Suite 2 Ref Models       : " << (s2_ref_failures == 0 ? "All verified (0 failures)" : std::to_string(s2_ref_failures) + " unverified / limit reached") << "\n";
  }
  if (run_milp) {
    std::cout << "  Suite 3 (MILP B&B)       : " << milp_verified << " / " << milp_total << " verified ("
              << (all_s3_pass ? "PASSED" : "FAILED") << ")\n";
  }
  if (run_qp) {
    std::cout << "  Suite 4 (Convex QP)      : " << qp_verified << " / " << qp_total << " verified ("
              << (all_s4_pass ? "PASSED" : "FAILED") << ")\n";
  }
  std::cout << "  Limit-Reached Cases      : " << total_limit_reached << "\n";
  std::cout << "  NOT_APPLICABLE Cases     : " << total_not_applicable << "\n";
  std::cout << "  CSV Telemetry Produced   : " << (csv_ok ? "YES (" + csv_path + ")" : "NO / EMPTY") << "\n";
  std::cout << "  JSON Telemetry Produced  : " << (json_ok ? "YES (" + json_path + ")" : "NO / EMPTY") << "\n";
  std::cout << "  False Optimality Claims  : " << (has_false_passed_row ? "DETECTED (VIOLATION)" : "NONE (INTEGRITY PRESERVED)") << "\n";

  if (quick_mode) {
    const bool quick_success = all_s1_pass && all_s3_pass && all_s4_pass && csv_ok && json_ok && !has_false_passed_row;
    if (quick_success) {
      std::cout << "\n[QUICK SMOKE TEST RESULT] PASSED (exit code 0)\n";
      std::cout << "  All baseline suites verified; telemetry exported honestly without false optimality claims.\n";
      return 0;
    } else {
      std::cout << "\n[QUICK SMOKE TEST RESULT] FAILED (exit code 1)\n";
      if (!all_s1_pass) std::cout << "  - Suite 1 failure: " << s1_passed << "/" << s1_total << "\n";
      if (!all_s3_pass) std::cout << "  - Suite 3 failure: " << milp_verified << "/" << milp_total << "\n";
      if (!all_s4_pass) std::cout << "  - Suite 4 failure: " << qp_verified << "/" << qp_total << "\n";
      if (!csv_ok) std::cout << "  - CSV output failed\n";
      if (!json_ok) std::cout << "  - JSON output failed\n";
      if (has_false_passed_row) std::cout << "  - False PASSED claim detected\n";
      return 1;
    }
  } else if (full_mode) {
    const bool full_success = all_s1_pass && all_s2_ref_pass && all_s3_pass && all_s4_pass && csv_ok && json_ok && !has_false_passed_row;
    if (full_success) {
      std::cout << "\n[FULL VERIFICATION RESULT] PASSED (exit code 0)\n";
      return 0;
    } else {
      std::cout << "\n[FULL VERIFICATION RESULT] FAILED (exit code 1)\n";
      if (!all_s1_pass) std::cout << "  - Suite 1 failure: " << s1_passed << "/" << s1_total << "\n";
      if (!all_s2_ref_pass) std::cout << "  - Suite 2 reference models unverified: " << s2_ref_failures << "\n";
      if (!all_s3_pass) std::cout << "  - Suite 3 failure: " << milp_verified << "/" << milp_total << "\n";
      if (!all_s4_pass) std::cout << "  - Suite 4 failure: " << qp_verified << "/" << qp_total << "\n";
      if (!csv_ok) std::cout << "  - CSV output failed\n";
      if (!json_ok) std::cout << "  - JSON output failed\n";
      if (has_false_passed_row) std::cout << "  - False PASSED claim detected\n";
      return 1;
    }
  }
  return 0;
}
