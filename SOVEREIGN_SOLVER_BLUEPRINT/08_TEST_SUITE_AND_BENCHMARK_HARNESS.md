# TEST SUITE & BENCHMARK HARNESS: INDUS-OPT

This document specifies the automated unit testing, rational arithmetic fuzzing, and benchmark execution harness for verifying **INDUS-OPT**.

---

## 1. Unit Testing Strategy (GoogleTest)

The test suite in `tests/unit/` covers every subsystem independently:

```cpp
// Example: tests/unit/test_simplex.cpp
#include <gtest/gtest.h>
#include "indus/model.hpp"
#include "indus/options.hpp"

using namespace indus;

TEST(SimplexTest, SolvesTextbookLP) {
    // Maximize 3 x1 + 2 x2
    // s.t. x1 + x2 <= 4
    //      x1 - x2 <= 2
    //      x1, x2 >= 0
    Model model;
    model.name = "textbook";
    model.sense = ObjSense::kMaximize;
    model.num_cols = 2;
    model.num_rows = 2;
    model.c = {3.0, 2.0};
    model.col_lower = {0.0, 0.0};
    model.col_upper = {kInfinity, kInfinity};
    model.row_lower = {-kInfinity, -kInfinity};
    model.row_upper = {4.0, 2.0};

    // Matrix entries: A = [[1, 1], [1, -1]]
    model.A.m = 2; model.A.n = 2;
    model.A.col_ptr = {0, 2, 4};
    model.A.row_idx = {0, 1, 0, 1};
    model.A.values  = {1.0, 1.0, 1.0, -1.0};

    Options opts;
    opts.algorithm = "dual_simplex";
    Solution sol = solve(model, opts);

    EXPECT_EQ(sol.status, SolveStatus::kOptimal);
    EXPECT_NEAR(sol.objective_value, 11.0, 1e-7); // x1=3, x2=1 -> obj = 3(3) + 2(1) = 11
    EXPECT_NEAR(sol.col_value[0], 3.0, 1e-7);
    EXPECT_NEAR(sol.col_value[1], 1.0, 1e-7);
}
```

---

## 2. Automated Netlib Benchmark Runner (`bench/runners/netlib_runner.py`)

A fully automated Python script executes the benchmark suite against reference published optima:

```python
#!/usr/bin/env python3
"""
bench/runners/netlib_runner.py
Runs INDUS-OPT against committed benchmark instances and verifies against reference optima.
"""

import os
import sys
import json
import subprocess
import time

SOLVER_BIN = "./build/apps/indus-cli/indus"
TEST_DIR = "./test_models"
REF_JSON = "./test_models/reference.json"

def run_benchmarks():
    if not os.path.exists(SOLVER_BIN):
        print(f"Error: Solver binary {SOLVER_BIN} not found. Please build the project first.")
        sys.exit(1)

    with open(REF_JSON, 'r') as f:
        references = json.load(f)

    passed = 0
    total = len(references)

    print(f"{'Instance':<15} | {'Status':<10} | {'Solver Obj':<18} | {'Reference Obj':<18} | {'Rel Gap':<10} | {'Time (s)':<8}")
    print("-" * 95)

    for instance, ref_obj in references.items():
        mps_file = os.path.join(TEST_DIR, f"{instance}.mps")
        if not os.path.exists(mps_file):
            continue

        sol_file = f"/tmp/{instance}.sol"
        t0 = time.time()
        res = subprocess.run([SOLVER_BIN, "solve", mps_file, "--write-sol", sol_file],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - t0

        # Parse solution
        calc_obj = None
        status = "FAIL"
        if os.path.exists(sol_file):
            with open(sol_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0] == "OBJECTIVE":
                        calc_obj = float(parts[1])
                    elif parts and parts[0] == "STATUS":
                        status = parts[1]

        rel_gap = abs(calc_obj - ref_obj) / (abs(ref_obj) + 1.0) if calc_obj is not None else 1.0
        match = (rel_gap <= 1e-6) and (status == "OPTIMAL")

        if match:
            passed += 1
            verdict = "PASS"
        else:
            verdict = "FAIL"

        calc_str = f"{calc_obj:.8e}" if calc_obj is not None else "N/A"
        ref_str = f"{ref_obj:.8e}"
        print(f"{instance:<15} | {verdict:<10} | {calc_str:<18} | {ref_str:<18} | {rel_gap:<10.2e} | {elapsed:<8.3f}")

    print("-" * 95)
    print(f"Final Benchmark Result: {passed} / {total} Passed ({passed/total*100:.1f}%)")

if __name__ == "__main__":
    run_benchmarks()
```
