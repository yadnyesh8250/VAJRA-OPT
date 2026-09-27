#include "indus/model.hpp"
#include "indus/options.hpp"
#include "indus/gpu.hpp"
#include "indus/verifier.hpp"
#include "indus/types.hpp"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef INDUS_SOURCE_DIR
#define INDUS_SOURCE_DIR "."
#endif

namespace {

fs::path find_benchmark_binary(int argc = 0, char* argv[] = nullptr) {
  if (argc > 1 && argv && argv[1]) {
    fs::path p(argv[1]);
    if (fs::exists(p)) return p;
  }
#ifdef INDUS_BENCHMARK_BIN
  if (fs::exists(INDUS_BENCHMARK_BIN)) return INDUS_BENCHMARK_BIN;
#endif
  const char* env_bin = std::getenv("INDUS_BENCHMARK_BIN");
  if (env_bin && fs::exists(env_bin)) return env_bin;

  const std::vector<fs::path> candidates = {
#ifdef INDUS_BINARY_DIR
      fs::path(INDUS_BINARY_DIR) / "indus_benchmark",
      fs::path(INDUS_BINARY_DIR) / "indus_benchmark.exe",
#endif
      fs::path(INDUS_SOURCE_DIR) / "build" / "indus_benchmark",
      fs::current_path() / "indus_benchmark"
  };
  for (const auto& c : candidates) {
    if (!c.empty() && fs::exists(c)) return c;
  }
  return {};
}

// Portable unique temporary directory creation
fs::path make_unique_tmpdir(const std::string& prefix = "vajra_harness") {
  const long long ts = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::string dirname = prefix + "_" + std::to_string(ts);
  fs::path dir = fs::temp_directory_path() / dirname;
  fs::create_directories(dir);
  return dir;
}

// RAII directory cleanup
struct TmpDirGuard {
  fs::path path;
  explicit TmpDirGuard(fs::path p) : path(std::move(p)) {}
  ~TmpDirGuard() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  TmpDirGuard(const TmpDirGuard&) = delete;
  TmpDirGuard& operator=(const TmpDirGuard&) = delete;
};

#define HARNESS_ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr << "HARNESS FAILURE: " << msg << " (" << #cond << ")\n"; \
      std::abort(); \
    } \
  } while (0)

// 1. Missing CUDA behavior & CPU fallback test
void test_missing_cuda_behavior() {
  std::cout << "[TEST 1] Missing CUDA behavior & CPU fallback detection...\n";
  auto dev = indus::gpu::probe_device();
#ifndef INDUS_HAS_CUDA
  HARNESS_ASSERT(!dev.available, "In CPU-only build, CUDA device must be reported as unavailable");
#endif

  indus::Model model;
  model.name = "gpu_fallback_test";
  model.num_rows = 1;
  model.num_cols = 2;
  model.c = {1.0, 2.0};
  model.col_lower = {0.0, 0.0};
  model.col_upper = {10.0, 10.0};
  model.row_lower = {1.0};
  model.row_upper = {1e20};
  model.A.set_from_triplets(1, 2, {{0, 0, 1.0}, {0, 1, 1.0}});

  indus::Options opts;
  opts.set("algorithm", "pdhg_cuda");
  opts.use_gpu = true;
  opts.iteration_limit = 500;
  opts.set("tolerance", 1e-4);

  indus::Solution sol = indus::solve(model, opts);
  HARNESS_ASSERT(sol.status == indus::SolveStatus::kOptimal || sol.status == indus::SolveStatus::kFeasible, "Solver must return optimal or feasible on fallback");
  std::cout << "  [PASS] Missing CUDA detection & CPU fallback verified.\n";
}

// 2. Suite filtering CLI test
void test_suite_filtering(const fs::path& bench_bin, const fs::path& tmp_dir) {
  std::cout << "[TEST 2] CLI Suite filtering (--suite lp, milp, qp)...\n";
  HARNESS_ASSERT(!bench_bin.empty() && fs::exists(bench_bin), "indus_benchmark executable must exist");

  // Test --suite lp
  {
    fs::path out_dir = tmp_dir / "lp_artifacts";
    fs::path csv_path = tmp_dir / "lp.csv";
    fs::path json_path = tmp_dir / "lp.json";
    fs::create_directories(out_dir);

    std::string cmd = "\"" + bench_bin.string() + "\" --suite lp --quick --out-dir \"" + out_dir.string() + "\" --csv \"" + csv_path.string() + "\" --json \"" + json_path.string() + "\"";
    int ret = std::system(cmd.c_str());
    HARNESS_ASSERT(ret == 0, "--suite lp must exit 0 in quick mode");

    std::ifstream f(csv_path);
    std::string line;
    int lp_count = 0, other_count = 0;
    while (std::getline(f, line)) {
      if (line.find("NetlibBaseline") != std::string::npos) lp_count++;
      if (line.find("EngineComparison") != std::string::npos || line.find(",MILP,") != std::string::npos || line.find(",QP,") != std::string::npos) {
        other_count++;
      }
    }
    HARNESS_ASSERT(lp_count == 22, "LP suite must contain exactly 22 NetlibBaseline models");
    HARNESS_ASSERT(other_count == 0, "LP suite filter must not run other suites");
  }

  // Test --suite milp
  {
    fs::path out_dir = tmp_dir / "milp_artifacts";
    fs::path csv_path = tmp_dir / "milp.csv";
    fs::path json_path = tmp_dir / "milp.json";
    fs::create_directories(out_dir);

    std::string cmd = "\"" + bench_bin.string() + "\" --suite milp --quick --out-dir \"" + out_dir.string() + "\" --csv \"" + csv_path.string() + "\" --json \"" + json_path.string() + "\"";
    int ret = std::system(cmd.c_str());
    HARNESS_ASSERT(ret == 0, "--suite milp must exit 0 in quick mode");

    std::ifstream f(csv_path);
    std::string line;
    int milp_count = 0, lp_count = 0;
    while (std::getline(f, line)) {
      if (line.find("MILP") != std::string::npos && line.find("Platform") == std::string::npos) milp_count++;
      if (line.find("NetlibBaseline") != std::string::npos) lp_count++;
    }
    HARNESS_ASSERT(milp_count >= 5, "MILP suite must contain at least 5 models");
    HARNESS_ASSERT(lp_count == 0, "MILP suite filter must not run LP baseline");
  }

  // Test --suite qp
  {
    fs::path out_dir = tmp_dir / "qp_artifacts";
    fs::path csv_path = tmp_dir / "qp.csv";
    fs::path json_path = tmp_dir / "qp.json";
    fs::create_directories(out_dir);

    std::string cmd = "\"" + bench_bin.string() + "\" --suite qp --quick --out-dir \"" + out_dir.string() + "\" --csv \"" + csv_path.string() + "\" --json \"" + json_path.string() + "\"";
    int ret = std::system(cmd.c_str());
    HARNESS_ASSERT(ret == 0, "--suite qp must exit 0 in quick mode");

    std::ifstream f(csv_path);
    std::string line;
    int qp_count = 0, lp_count = 0;
    while (std::getline(f, line)) {
      if (line.find(",QP,") != std::string::npos && line.find("Platform") == std::string::npos) qp_count++;
      if (line.find("NetlibBaseline") != std::string::npos) lp_count++;
    }
    HARNESS_ASSERT(qp_count >= 5, "QP suite must contain at least 5 models");
    HARNESS_ASSERT(lp_count == 0, "QP suite filter must not run LP baseline");
  }

  std::cout << "  [PASS] CLI suite filtering verified.\n";
}

// 3. JSON validity and Python parse test
void test_json_validity_and_python_parse(const fs::path& bench_bin, const fs::path& tmp_dir, fs::path& out_json_path) {
  std::cout << "[TEST 3] JSON schema validity & Python json parser test...\n";
  fs::path out_dir = tmp_dir / "all_artifacts";
  fs::path json_path = tmp_dir / "all_results.json";
  fs::path csv_path = tmp_dir / "all_results.csv";
  fs::create_directories(out_dir);

  std::string cmd = "\"" + bench_bin.string() + "\" --quick --out-dir \"" + out_dir.string() + "\" --json \"" + json_path.string() + "\" --csv \"" + csv_path.string() + "\"";
  int ret = std::system(cmd.c_str());
  HARNESS_ASSERT(ret == 0, "Benchmark quick mode must exit 0");
  HARNESS_ASSERT(fs::exists(json_path), "Benchmark JSON output file must exist");
  HARNESS_ASSERT(fs::exists(csv_path), "Benchmark CSV output file must exist");

  std::string py_cmd = "python3 -c \"import json\n"
                       "d = json.load(open('" + json_path.string() + "'))\n"
                       "assert 'benchmark_metadata' in d\n"
                       "assert 'results' in d\n"
                       "assert 'summary' in d\n"
                       "assert len(d['results']) >= 20\n"
                       "for r in d['results']:\n"
                       "    for k in ['suite', 'model_name', 'problem_class', 'algorithm', 'solver_status', 'verification_status', 'objective_value', 'runtime']:\n"
                       "        assert k in r, f'Missing key {k}'\n"
                       "print('JSON schema validation passed')\n\"";
  int py_ret = std::system(py_cmd.c_str());
  HARNESS_ASSERT(py_ret == 0, "Python standard json module must parse benchmark JSON without errors");
  out_json_path = json_path;
  std::cout << "  [PASS] JSON schema and Python parsing verified.\n";
}

// 4. Anti-fabrication: No false optimality claims
void test_no_false_passed_contract(const fs::path& json_path) {
  std::cout << "[TEST 4] Anti-fabrication: No non-optimal row may report PASSED...\n";
  HARNESS_ASSERT(fs::exists(json_path), "Benchmark JSON file must exist");

  std::string py_cmd = "python3 -c \"import json\n"
                       "d = json.load(open('" + json_path.string() + "'))\n"
                       "for r in d['results']:\n"
                       "    st = r['solver_status']\n"
                       "    vst = r['verification_status']\n"
                       "    if st in ['FEASIBLE', 'ITERATION_LIMIT', 'TIME_LIMIT', 'NODE_LIMIT', 'UNSUPPORTED', 'MODEL_ERROR']:\n"
                       "        assert vst not in ['OPTIMAL_VERIFIED', 'PASSED'], f'Non-optimal status {st} claimed {vst} on {r[\"model_name\"]}'\n"
                       "print('Anti-fabrication contract verified')\n\"";
  int py_ret = std::system(py_cmd.c_str());
  HARNESS_ASSERT(py_ret == 0, "Anti-fabrication check must pass");
  std::cout << "  [PASS] Anti-fabrication contract verified.\n";
}

// 5. Unsupported models classification
void test_unsupported_models(const fs::path& json_path) {
  std::cout << "[TEST 5] Unsupported models classified as UNSUPPORTED...\n";
  HARNESS_ASSERT(fs::exists(json_path), "Benchmark JSON file must exist");

  std::string py_cmd = "python3 -c \"import json\n"
                       "d = json.load(open('" + json_path.string() + "'))\n"
                       "unsupported = [r for r in d['results'] if r['verification_status'] == 'UNSUPPORTED']\n"
                       "assert len(unsupported) >= 2, 'Must have at least 2 unsupported models (e.g. nonconvex QP, MIQP)'\n"
                       "names = [r['model_name'] for r in unsupported]\n"
                       "assert 'nonconvex_qp_indefinite' in names\n"
                       "assert 'miqp_blend' in names\n"
                       "print('Unsupported models properly identified')\n\"";
  int py_ret = std::system(py_cmd.c_str());
  HARNESS_ASSERT(py_ret == 0, "Unsupported models must be flagged honestly");
  std::cout << "  [PASS] Unsupported model classification verified.\n";
}

// 6. Limit-reached classification
void test_limit_reached_classification(const fs::path& json_path) {
  std::cout << "[TEST 6] Limit-reached models classified as LIMIT_REACHED...\n";
  HARNESS_ASSERT(fs::exists(json_path), "Benchmark JSON file must exist");

  std::string py_cmd = "python3 -c \"import json\n"
                       "d = json.load(open('" + json_path.string() + "'))\n"
                       "limits = [r for r in d['results'] if r['verification_status'] == 'LIMIT_REACHED']\n"
                       "assert len(limits) >= 2, 'Must have at least 2 limit-reached models (node limit and time limit)'\n"
                       "for r in limits:\n"
                       "    assert r['solver_status'] in ['NODE_LIMIT', 'TIME_LIMIT', 'ITERATION_LIMIT']\n"
                       "print('Limit reached properly verified')\n\"";
  int py_ret = std::system(py_cmd.c_str());
  HARNESS_ASSERT(py_ret == 0, "Limit-reached models must be classified honestly");
  std::cout << "  [PASS] Limit-reached classification verified.\n";
}

// 7. Invalid CLI args rejection
void test_invalid_cli_args(const fs::path& bench_bin) {
  std::cout << "[TEST 7] Invalid CLI arguments rejection...\n";
  std::string cmd = "\"" + bench_bin.string() + "\" --suite invalid_suite_name >/dev/null 2>&1";
  int ret = std::system(cmd.c_str());
  int exit_code = (ret >> 8);
  HARNESS_ASSERT(exit_code != 0, "Invalid suite option must exit nonzero");
  std::cout << "  [PASS] Invalid CLI arguments properly rejected.\n";
}

} // namespace

int main(int argc, char* argv[]) {
  std::cout << "========================================================\n";
  std::cout << "VAJRA-OPT Benchmark Harness & Verification Suite (Phase 9)\n";
  std::cout << "========================================================\n\n";

  fs::path bench_bin = find_benchmark_binary(argc, argv);
  HARNESS_ASSERT(!bench_bin.empty() && fs::exists(bench_bin),
                 "indus_benchmark executable could not be resolved");

  fs::path tmp_dir = make_unique_tmpdir();
  TmpDirGuard guard(tmp_dir);

  test_missing_cuda_behavior();
  test_suite_filtering(bench_bin, tmp_dir);

  fs::path generated_json;
  test_json_validity_and_python_parse(bench_bin, tmp_dir, generated_json);
  test_no_false_passed_contract(generated_json);
  test_unsupported_models(generated_json);
  test_limit_reached_classification(generated_json);
  test_invalid_cli_args(bench_bin);

  std::cout << "\n========================================================\n";
  std::cout << "ALL PHASE 9 BENCHMARK HARNESS TESTS PASSED SUCCESSFULLY!\n";
  std::cout << "========================================================\n";
  return 0;
}
