#include "indus/io.hpp"
#include "indus/model.hpp"
#include "indus/sparse.hpp"
#include "indus/tolerances.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ─── Source-path resolution ──────────────────────────────────────────────────
#ifndef INDUS_SOURCE_DIR
#define INDUS_SOURCE_DIR "."
#endif

std::string find_model_path(const std::string& rel) {
    std::filesystem::path p_src = std::filesystem::path(INDUS_SOURCE_DIR) / rel;
    if (std::filesystem::exists(p_src)) return p_src.string();
    if (std::filesystem::exists(rel))        return rel;
    if (std::filesystem::exists("../" + rel))   return "../" + rel;
    if (std::filesystem::exists("../../" + rel)) return "../../" + rel;
    return rel;
}

// ─── Portable unique temporary directory ─────────────────────────────────────
// Uses high_resolution_clock nanoseconds as a suffix. Standard C++17 only;
// does NOT use std::filesystem::unique_path() (Boost-only / non-standard).
std::filesystem::path make_unique_tmpdir(const std::string& prefix = "vajra_test") {
    namespace fs = std::filesystem;
    const long long ts = std::chrono::high_resolution_clock::now()
                             .time_since_epoch().count();
    const std::string dirname = prefix + "_" + std::to_string(ts);
    fs::path dir = fs::temp_directory_path() / dirname;
    fs::create_directories(dir);
    return dir;
}

// RAII guard: removes the directory tree on destruction (errors silently ignored).
struct TmpDirGuard {
    std::filesystem::path path;
    explicit TmpDirGuard(std::filesystem::path p) : path(std::move(p)) {}
    ~TmpDirGuard() { std::error_code ec; std::filesystem::remove_all(path, ec); }
    TmpDirGuard(const TmpDirGuard&) = delete;
    TmpDirGuard& operator=(const TmpDirGuard&) = delete;
};

// Helper: read a file into a std::string.
std::string read_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
}

// ─── Tests ───────────────────────────────────────────────────────────────────

void test_mps_parser_crude_blend() {
    std::cout << "--- [Test 1] MPS Reader: crude_blend.mps ---" << std::endl;
    const std::string mps_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps");
    indus::Model model = indus::io::read_mps(mps_path);

    std::cout << "Model Name: " << model.name << std::endl;
    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    assert(model.num_rows == 3);
    assert(model.num_cols == 3);
    assert(model.sense == indus::ObjSense::kMaximize);

    assert(model.col_names.size() == 3);
    assert(model.col_names[0] == "AL");
    assert(model.col_names[1] == "BN");
    assert(model.col_names[2] == "MU");

    assert(std::abs(model.c[0] - 2.40) < 1e-6);
    assert(std::abs(model.c[1] - 1.60) < 1e-6);
    assert(std::abs(model.c[2] - 1.64) < 1e-6);

    assert(std::abs(model.col_lower[0] - 10.0) < 1e-6);
    assert(model.col_upper[0] > 1e19);

    assert(std::abs(model.col_lower[1] - 0.0) < 1e-6);
    assert(std::abs(model.col_upper[1] - 45.0) < 1e-6);

    assert(std::abs(model.col_lower[2] - 0.0) < 1e-6);
    assert(std::abs(model.col_upper[2] - 60.0) < 1e-6);

    assert(model.row_names.size() == 3);
    assert(model.row_names[0] == "THRUPUT");
    assert(std::abs(model.row_lower[0] - 90.0) < 1e-6);
    assert(std::abs(model.row_upper[0] - 120.0) < 1e-6);

    assert(model.row_names[1] == "DIESEL");
    assert(std::abs(model.row_lower[1] - 40.0) < 1e-6);
    assert(std::abs(model.row_upper[1] - 40.0) < 1e-6);

    assert(model.row_names[2] == "SULPHUR");
    assert(model.row_lower[2] < -1e19);
    assert(std::abs(model.row_upper[2] - 0.0) < 1e-6);

    std::cout << "PASS: crude_blend.mps parsed correctly with RANGES, BOUNDS, and OBJSENSE." << std::endl;
}

void test_lp_parser_crude_blend() {
    std::cout << "--- [Test 2] LP Reader: crude_blend.lp ---" << std::endl;
    const std::string lp_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp");
    indus::Model model = indus::io::read_lp(lp_path);

    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    assert(model.num_rows == 3);
    assert(model.num_cols == 3);
    assert(model.sense == indus::ObjSense::kMaximize);

    indus::Model mps_model = indus::io::read_mps(find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"));
    assert(model.num_rows == mps_model.num_rows);
    assert(model.num_cols == mps_model.num_cols);

    for (int j = 0; j < model.num_cols; ++j) {
        assert(std::abs(model.c[j] - mps_model.c[j]) < 1e-6);
        assert(std::abs(model.col_lower[j] - mps_model.col_lower[j]) < 1e-6);
        if (mps_model.col_upper[j] < 1e19) {
            assert(std::abs(model.col_upper[j] - mps_model.col_upper[j]) < 1e-6);
        } else {
            assert(model.col_upper[j] > 1e19);
        }
    }

    for (int i = 0; i < model.num_rows; ++i) {
        if (mps_model.row_lower[i] > -1e19) {
            assert(std::abs(model.row_lower[i] - mps_model.row_lower[i]) < 1e-6);
        } else {
            assert(model.row_lower[i] < -1e19);
        }
        if (mps_model.row_upper[i] < 1e19) {
            assert(std::abs(model.row_upper[i] - mps_model.row_upper[i]) < 1e-6);
        } else {
            assert(model.row_upper[i] > 1e19);
        }
    }

    std::cout << "PASS: crude_blend.lp matches crude_blend.mps exactly." << std::endl;
}

void test_solve_crude_blend_and_export() {
    std::cout << "--- [Test 3] End-to-End Solve: crude_blend.mps & IO Writers ---" << std::endl;
    indus::Model model = indus::io::read_mps(find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"));

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    std::cout << "Solve Status: " << static_cast<int>(sol.status) << " (" << sol.status_message << ")" << std::endl;
    std::cout << "Iterations: " << sol.iterations << std::endl;
    std::cout << "Optimal Objective: " << sol.objective_value << std::endl;
    for (size_t j = 0; j < model.col_names.size(); ++j) {
        std::cout << "  " << model.col_names[j] << " = " << sol.col_value[j]
                  << " (dual: " << sol.col_dual[j] << ")" << std::endl;
    }
    for (size_t i = 0; i < model.row_names.size(); ++i) {
        std::cout << "  Row " << model.row_names[i] << " = " << sol.row_value[i]
                  << " (shadow price: " << sol.row_dual[i] << ")" << std::endl;
    }

    assert(sol.status == indus::SolveStatus::kOptimal);
    assert(sol.quality.is_primal_feasible);
    assert(sol.quality.is_dual_feasible);

    // Write to a unique temp directory — not the source tree, and cleaned up after.
    auto tmp_dir = make_unique_tmpdir("vajra_test3");
    TmpDirGuard guard(tmp_dir);
    const std::string sol_path  = (tmp_dir / "crude_blend.sol").string();
    const std::string json_path = (tmp_dir / "crude_blend.json").string();

    indus::io::write_solution(sol, model, sol_path);
    indus::io::write_json(sol, model, json_path);

    assert(std::filesystem::exists(sol_path));
    assert(std::filesystem::file_size(sol_path) > 0);
    assert(std::filesystem::exists(json_path));
    assert(std::filesystem::file_size(json_path) > 0);

    // Validate JSON with Python's json module when Python3 is available.
    if (std::system("python3 --version > /dev/null 2>&1") == 0) {
        // Write a tiny script to avoid shell-quoting issues with paths.
        const std::string py_script = (tmp_dir / "parse.py").string();
        {
            std::ofstream pf(py_script);
            pf << "import json, sys\n"
               << "with open(sys.argv[1]) as fh: json.load(fh)\n"
               << "sys.exit(0)\n";
        }
        const std::string cmd = "python3 \"" + py_script + "\" \"" + json_path + "\"";
        assert(std::system(cmd.c_str()) == 0
               && "Python json.load() must accept the generated crude_blend.json");
    }

    std::cout << "PASS: crude_blend solved to optimality and exported to .sol and .json." << std::endl;
}

void test_netlib_afiro() {
    std::cout << "--- [Test 4] Netlib Benchmark: afiro.mps ---" << std::endl;
    const std::string afiro_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps");
    indus::Model model = indus::io::read_mps(afiro_path);

    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    assert(model.num_rows == 27);
    assert(model.num_cols == 32);

    indus::Options opts;
    indus::Solution sol = indus::solve(model, opts);

    std::cout << "Solve Status: " << static_cast<int>(sol.status) << " (" << sol.status_message << ")" << std::endl;
    std::cout << "Iterations: " << sol.iterations << std::endl;
    std::cout << "Computed Objective: " << sol.objective_value << std::endl;

    const double published_optimal = -464.75314286;
    const double rel_err = std::abs(sol.objective_value - published_optimal) / std::abs(published_optimal);
    std::cout << "Published Optimal:  " << published_optimal << std::endl;
    std::cout << "Relative Error:     " << rel_err << std::endl;

    assert(sol.status == indus::SolveStatus::kOptimal);
    assert(rel_err < 1e-5);
    assert(sol.quality.is_primal_feasible);

    std::cout << "PASS: Netlib AFIRO solved and matches published benchmark (-464.75314286)." << std::endl;
}

void test_netlib_blend_parse() {
    std::cout << "--- [Test 5] Netlib Benchmark Parsing: blend.mps ---" << std::endl;
    const std::string blend_path = find_model_path("SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend.mps");
    indus::Model model = indus::io::read_mps(blend_path);

    std::cout << "Dimensions: " << model.num_rows << " rows, " << model.num_cols << " cols, "
              << model.A.nnz() << " nonzeros" << std::endl;

    int obj_nnz = 0;
    for (double cval : model.c) {
        if (std::abs(cval) > 1e-12) ++obj_nnz;
    }
    std::cout << "A nnz: " << model.A.nnz() << ", obj nnz: " << obj_nnz
              << ", total MPS entries: " << (model.A.nnz() + obj_nnz) << std::endl;

    assert(model.num_rows == 74);
    assert(model.num_cols == 83);
    assert(model.A.nnz() + obj_nnz == 521);

    std::cout << "PASS: Netlib BLEND parsed correctly (521 total matrix + objective entries)." << std::endl;
}

// ─── Test 6: Unnamed model ───────────────────────────────────────────────────
// num_cols/num_rows > 0 but col_names/row_names are empty.
// write_solution/write_json must produce entries for every column and row,
// using fallback names c0, c1, … and r0, r1, …
void test_unnamed_model_write() {
    std::cout << "--- [Test 6] Unnamed Model: write_solution / write_json ---" << std::endl;

    indus::Model model;
    model.num_rows = 2;
    model.num_cols = 2;
    model.name     = "";    // explicitly unnamed
    model.sense    = indus::ObjSense::kMinimize;
    model.c         = {1.0, 2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {1e20, 1e20};
    model.row_lower = {0.0, 0.0};
    model.row_upper = {1e20, 1e20};
    // col_names and row_names intentionally left empty

    {
        std::vector<indus::la::Triplet> trips = {{0, 0, 1.0}, {1, 1, 1.0}};
        model.A.set_from_triplets(2, 2, trips);
    }

    indus::Solution sol;
    sol.status          = indus::SolveStatus::kOptimal;
    sol.status_message  = "OK";
    sol.algorithm_used  = "test";
    sol.certificate_type = "none";
    sol.objective_value  = 0.0;
    sol.best_dual_bound  = 0.0;
    sol.col_value = {0.0, 0.0};
    sol.col_dual  = {1.0, 2.0};
    sol.row_value = {0.0, 0.0};
    sol.row_dual  = {0.0, 0.0};

    auto tmp_dir = make_unique_tmpdir("vajra_unnamed");
    TmpDirGuard guard(tmp_dir);
    const std::string sol_path  = (tmp_dir / "unnamed.sol").string();
    const std::string json_path = (tmp_dir / "unnamed.json").string();

    // Must not throw; must write entries for all num_cols / num_rows.
    indus::io::write_solution(sol, model, sol_path);
    indus::io::write_json(sol, model, json_path);

    const std::string sol_txt  = read_file(sol_path);
    const std::string json_txt = read_file(json_path);

    assert(!sol_txt.empty()  && ".sol file is empty");
    assert(!json_txt.empty() && ".json file is empty");

    // Fallback names in .sol
    assert(sol_txt.find("c0") != std::string::npos && "c0 fallback missing from .sol");
    assert(sol_txt.find("c1") != std::string::npos && "c1 fallback missing from .sol");
    assert(sol_txt.find("r0") != std::string::npos && "r0 fallback missing from .sol");
    assert(sol_txt.find("r1") != std::string::npos && "r1 fallback missing from .sol");

    // Fallback names and "unnamed" marker in .json
    assert(json_txt.find("\"c0\"")     != std::string::npos && "c0 fallback missing from .json");
    assert(json_txt.find("\"c1\"")     != std::string::npos && "c1 fallback missing from .json");
    assert(json_txt.find("\"r0\"")     != std::string::npos && "r0 fallback missing from .json");
    assert(json_txt.find("\"r1\"")     != std::string::npos && "r1 fallback missing from .json");
    assert(json_txt.find("\"unnamed\"") != std::string::npos && "\"unnamed\" marker missing from .json");

    std::cout << "PASS: Unnamed model (num_cols=2, num_rows=2, empty name vectors) "
                 "writes fallback names c0/c1/r0/r1 correctly." << std::endl;
}

// ─── Test 7: Partially named model ───────────────────────────────────────────
// col_names has fewer entries than num_cols; row_names has fewer than num_rows.
// Named entries must appear with their names; remaining entries use fallback names.
void test_partial_names_model_write() {
    std::cout << "--- [Test 7] Partial Names Model: write_solution / write_json ---" << std::endl;

    indus::Model model;
    model.num_rows = 3;
    model.num_cols = 3;
    model.name  = "partial_names_model";
    model.sense = indus::ObjSense::kMinimize;
    model.c         = {1.0, 2.0, 3.0};
    model.col_lower = {0.0, 0.0, 0.0};
    model.col_upper = {1e20, 1e20, 1e20};
    model.row_lower = {0.0, 0.0, 0.0};
    model.row_upper = {1e20, 1e20, 1e20};
    // Provide only 2 of 3 column names, and 1 of 3 row names.
    model.col_names = {"x0", "x1"};   // "x2" missing → should fall back to "c2"
    model.row_names = {"con_A"};       // "con_B", "con_C" missing → "r1", "r2"

    {
        std::vector<indus::la::Triplet> trips = {{0,0,1.0},{1,1,1.0},{2,2,1.0}};
        model.A.set_from_triplets(3, 3, trips);
    }

    indus::Solution sol;
    sol.status          = indus::SolveStatus::kOptimal;
    sol.status_message  = "OK";
    sol.algorithm_used  = "test";
    sol.certificate_type = "none";
    sol.objective_value  = 0.0;
    sol.best_dual_bound  = 0.0;
    sol.col_value = {0.0, 0.0, 0.0};
    sol.col_dual  = {1.0, 2.0, 3.0};
    sol.row_value = {0.0, 0.0, 0.0};
    sol.row_dual  = {0.0, 0.0, 0.0};

    auto tmp_dir = make_unique_tmpdir("vajra_partial");
    TmpDirGuard guard(tmp_dir);
    const std::string sol_path  = (tmp_dir / "partial.sol").string();
    const std::string json_path = (tmp_dir / "partial.json").string();

    indus::io::write_solution(sol, model, sol_path);
    indus::io::write_json(sol, model, json_path);

    const std::string sol_txt  = read_file(sol_path);
    const std::string json_txt = read_file(json_path);

    // Named entries present
    assert(sol_txt.find("x0")    != std::string::npos && "x0 missing from .sol");
    assert(sol_txt.find("x1")    != std::string::npos && "x1 missing from .sol");
    assert(sol_txt.find("con_A") != std::string::npos && "con_A missing from .sol");

    // Fallback for missing entries
    assert(sol_txt.find("c2") != std::string::npos && "c2 fallback missing for 3rd column");
    assert(sol_txt.find("r1") != std::string::npos && "r1 fallback missing for 2nd row");
    assert(sol_txt.find("r2") != std::string::npos && "r2 fallback missing for 3rd row");

    // Same check in JSON
    assert(json_txt.find("\"x0\"")    != std::string::npos);
    assert(json_txt.find("\"c2\"")    != std::string::npos);
    assert(json_txt.find("\"con_A\"") != std::string::npos);
    assert(json_txt.find("\"r1\"")    != std::string::npos);
    assert(json_txt.find("\"r2\"")    != std::string::npos);

    std::cout << "PASS: Partial names model (3 cols/rows, 2/1 names supplied) "
                 "uses fallback names correctly." << std::endl;
}

// ─── Test 8: JSON escape characters + Python json.load() ─────────────────────
// Model name, variable names, row names, and status_message all contain
// double-quotes, backslashes, tabs, and newlines. The generated JSON must be
// syntactically valid (parseable by Python's json module) and must contain the
// correctly escaped sequences (no raw control characters inside string values).
void test_json_escape_and_python_parse() {
    std::cout << "--- [Test 8] JSON Escape: special chars + Python json.load() ---" << std::endl;

    indus::Model model;
    model.num_rows = 1;
    model.num_cols = 1;
    model.name  = "model_with\"quotes\"and\\backslash";  // raw: model_with"quotes"and\backslash
    model.sense = indus::ObjSense::kMinimize;
    model.c         = {1.0};
    model.col_lower = {0.0};
    model.col_upper = {1e20};
    model.row_lower = {0.0};
    model.row_upper = {1e20};
    model.col_names = {"var\twith\ttabs"};    // embedded tabs
    model.row_names = {"con\nnewline"};        // embedded newline

    {
        std::vector<indus::la::Triplet> trips = {{0, 0, 1.0}};
        model.A.set_from_triplets(1, 1, trips);
    }

    indus::Solution sol;
    sol.status          = indus::SolveStatus::kOptimal;
    sol.status_message  = "Solved\twith\ttabs\nand newlines\"";   // tabs, newline, quote
    sol.algorithm_used  = "test\\algo";                            // backslash
    sol.certificate_type = "none";
    sol.objective_value  = 0.0;
    sol.best_dual_bound  = 0.0;
    sol.col_value = {0.0};
    sol.col_dual  = {1.0};
    sol.row_value = {0.0};
    sol.row_dual  = {0.0};

    auto tmp_dir = make_unique_tmpdir("vajra_escape");
    TmpDirGuard guard(tmp_dir);
    const std::string json_path = (tmp_dir / "escape_test.json").string();

    indus::io::write_json(sol, model, json_path);

    const std::string json_txt = read_file(json_path);
    assert(!json_txt.empty() && "escape_test.json should not be empty");

    // Verify that the raw special characters are properly escaped:
    // (1) Embedded double-quotes must appear as \"  (not raw ")
    assert(json_txt.find("\\\"quotes\\\"") != std::string::npos
           && "Embedded quotes must be escaped as \\\"");
    // (2) Embedded backslash must appear as \\ (not raw \)
    assert(json_txt.find("\\\\backslash") != std::string::npos
           && "Embedded backslash must be escaped as \\\\");
    // (3) Embedded tab must appear as \t
    assert(json_txt.find("\\t") != std::string::npos
           && "Tab must be escaped as \\t");
    // (4) Embedded newline must appear as \n
    assert(json_txt.find("\\n") != std::string::npos
           && "Newline must be escaped as \\n");

    // Validate with Python's json module (when python3 is available).
    if (std::system("python3 --version > /dev/null 2>&1") == 0) {
        // Write a helper script so path quoting is not an issue.
        const std::string py_script = (tmp_dir / "parse_json.py").string();
        {
            std::ofstream pf(py_script);
            pf << "import json, sys\n"
               << "with open(sys.argv[1]) as fh:\n"
               << "    json.load(fh)\n"
               << "sys.exit(0)\n";
        }
        const std::string cmd = "python3 \"" + py_script + "\" \"" + json_path + "\"";
        const int py_ret = std::system(cmd.c_str());
        if (py_ret != 0) {
            std::cerr << "FAIL: python3 json.load() rejected the generated JSON.\n"
                      << "File: " << json_path << "\nContent:\n" << json_txt << "\n";
            assert(py_ret == 0 && "Generated JSON must be parseable by Python's json module");
        }
        std::cout << "  Python json.load() accepted the output." << std::endl;
    } else {
        std::cout << "  (python3 not found; Python JSON parse check skipped)" << std::endl;
    }

    std::cout << "PASS: JSON escaping is correct for all special-character types." << std::endl;
}

#if defined(_WIN32)
inline int decode_exit_code(int sys_ret) {
    return sys_ret;
}
#else
#include <sys/wait.h>
inline int decode_exit_code(int sys_ret) {
    if (sys_ret == -1) return -1;
    if (WIFEXITED(sys_ret)) {
        return WEXITSTATUS(sys_ret);
    }
    return -1;
}
#endif

// ─── Test 9: Benchmark exit-code & CSV semantics ─────────────────────────────
// Under the quick-mode smoke contract:
// 1) Suite 1 baseline must pass 22/22.
// 2) CSV telemetry must be generated and non-empty.
// 3) No non-optimal row (ITERATION_LIMIT, TIME_LIMIT, FEASIBLE) may report PASSED.
// 4) indus_benchmark --quick must return exit code 0.
void test_benchmark_csv_semantics() {
    std::cout << "--- [Test 9] Benchmark Exit Code & CSV Verification Semantics ---" << std::endl;

#ifndef INDUS_SOURCE_DIR
#define INDUS_SOURCE_DIR "."
#endif
    namespace fs = std::filesystem;
    fs::path bench_bin;

#ifdef INDUS_BENCHMARK_BIN
    if (fs::exists(INDUS_BENCHMARK_BIN)) {
        bench_bin = INDUS_BENCHMARK_BIN;
    }
#endif

    if (bench_bin.empty()) {
        const char* env_bin = std::getenv("INDUS_BENCHMARK_BIN");
        if (env_bin && fs::exists(env_bin)) {
            bench_bin = env_bin;
        }
    }

    if (bench_bin.empty()) {
        const std::vector<fs::path> candidates = {
#ifdef INDUS_BINARY_DIR
            fs::path(INDUS_BINARY_DIR) / "indus_benchmark",
            fs::path(INDUS_BINARY_DIR) / "indus_benchmark.exe",
            fs::path(INDUS_BINARY_DIR) / "Release" / "indus_benchmark",
            fs::path(INDUS_BINARY_DIR) / "Release" / "indus_benchmark.exe",
            fs::path(INDUS_BINARY_DIR) / "Debug" / "indus_benchmark",
            fs::path(INDUS_BINARY_DIR) / "Debug" / "indus_benchmark.exe",
#endif
            fs::path(INDUS_SOURCE_DIR) / "build-cpu" / "indus_benchmark",
            fs::path(INDUS_SOURCE_DIR) / "build"     / "indus_benchmark",
            fs::path(INDUS_SOURCE_DIR) / "build-cpu" / "indus_benchmark.exe",
            fs::path("indus_benchmark"),
            fs::path("indus_benchmark.exe")
        };
        for (const auto& c : candidates) {
            if (!c.empty() && fs::exists(c)) { bench_bin = c; break; }
        }
    }

    // Do NOT silently skip if executable is missing: assert and fail!
    assert(!bench_bin.empty() && fs::exists(bench_bin) &&
           "indus_benchmark executable must exist in the current build tree");

    auto tmp_dir = make_unique_tmpdir("vajra_bench_csv");
    TmpDirGuard guard(tmp_dir);
    const std::string csv_path = (tmp_dir / "bench.csv").string();
    const std::string art_dir  = (tmp_dir / "artifacts").string();

    const std::string cmd = "\"" + bench_bin.string() + "\" --quick"
                            " --out-dir \"" + art_dir + "\""
                            " --csv \"" + csv_path + "\"";
    const int bench_ret = std::system(cmd.c_str());
    const int exit_code = decode_exit_code(bench_ret);

    std::cout << "  indus_benchmark --quick exit code: " << exit_code
              << " (raw system() return: " << bench_ret << ")" << std::endl;

    // Assert that the benchmark execution succeeded under the quick-mode smoke contract
    assert(exit_code == 0 &&
           "indus_benchmark --quick must exit 0 when Suite 1 passes and classification integrity holds");

    // Check CSV exists and is non-empty
    assert(fs::exists(csv_path) && "benchmark CSV must be produced");
    const std::string csv_txt = read_file(csv_path);
    assert(!csv_txt.empty() && "benchmark CSV must not be empty");

    // Verify CSV semantics: no non-optimal row can have VerificationStatus=PASSED
    {
        std::istringstream ss(csv_txt);
        std::string line;
        int line_no = 0;
        int suite1_rows = 0;
        int suite2_rows = 0;
        while (std::getline(ss, line)) {
            ++line_no;
            if (line_no == 1) {
                assert(line.find("ExecutionStatus") != std::string::npos &&
                       line.find("VerificationStatus") != std::string::npos &&
                       "CSV header must contain ExecutionStatus and VerificationStatus columns");
                continue;
            }
            if (line.find("NetlibBaseline") != std::string::npos) ++suite1_rows;
            if (line.find("EngineComparison") != std::string::npos) ++suite2_rows;

            if (line.find("ITERATION_LIMIT") != std::string::npos ||
                line.find("TIME_LIMIT") != std::string::npos ||
                line.find("FEASIBLE") != std::string::npos ||
                line.find("NUM_ERR") != std::string::npos) {
                // If it is not OPTIMAL, VerificationStatus must NOT be PASSED
                if (line.find(",PASSED") != std::string::npos) {
                    std::cerr << "FAIL: Non-optimal row reported PASSED: " << line << "\n";
                    std::abort();
                }
            }
        }
        if (suite1_rows != 22 || suite2_rows != 5) {
            std::cerr << "FAIL: Unexpected row counts in CSV: suite1=" << suite1_rows
                      << " (expected 22), suite2=" << suite2_rows << " (expected 5)\n";
            std::abort();
        }
    }

    std::cout << "PASS: Benchmark quick-mode exited 0, CSV produced 27 models, no false optimality claims." << std::endl;
}

} // namespace


int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "RUNNING PHASE 3 IO & INTEGRATION TESTS  " << std::endl;
    std::cout << "========================================" << std::endl;

    test_mps_parser_crude_blend();
    test_lp_parser_crude_blend();
    test_solve_crude_blend_and_export();
    test_netlib_afiro();
    test_netlib_blend_parse();
    test_unnamed_model_write();
    test_partial_names_model_write();
    test_json_escape_and_python_parse();
    test_benchmark_csv_semantics();

    std::cout << "\nALL PHASE 3 TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
