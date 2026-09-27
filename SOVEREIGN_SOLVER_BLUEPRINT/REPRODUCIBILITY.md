# VAJRA-OPT (Siddhanta) Phase 9 Reproducibility Blueprint

This document details the exact, independent, and bitwise reproducible steps required to compile, validate, benchmark, and independently audit **VAJRA-OPT (Siddhanta)** for MRPL Problem Statement 26119.

---

## 1. System Environment

- **Operating System**: macOS (Darwin 24.3.0 / Apple Silicon ARM64) or Linux (Ubuntu 20.04/22.04 LTS x86_64)
- **Host Architecture**: `ARM64` (AArch64) / `x86_64`
- **Compiler**: Clang 15+ / AppleClang 15+ / GCC 11+ (C++20 standard, `-Wall -Wextra -pedantic`)
- **CMake Version**: 3.20 or newer
- **Python Version**: Python 3.9+ (Standard Library only for verifier; SciPy 1.10+ optional for external comparison)
- **CUDA Information**:
  - Host hardware: CPU-only macOS ARM64. Discrete NVIDIA GPU absent on current host.
  - CUDA Compilation: `INDUS_ENABLE_CUDA=OFF` (active by default for CPU evaluation).
  - CUDA Status: `NOT AVAILABLE` / Execution Backend: `CPU FALLBACK` / GPU Speedup: `NOT MEASURED`.
  - Phase 9B Physical Hardware Validation: Documented in [`CUDA_VALIDATION_WORKFLOW.md`](file:///Users/yadnyesh8250/Desktop/VAJRA-OPT/SOVEREIGN_SOLVER_BLUEPRINT/CUDA_VALIDATION_WORKFLOW.md).

---

## 2. Clean Out-of-Tree Build Instructions

```bash
# 1. Clean build directory configuration
cmake -S . -B /tmp/vajra-phase9-final \
  -DCMAKE_BUILD_TYPE=Release \
  -DINDUS_ENABLE_CUDA=OFF

# 2. Parallel compilation
cmake --build /tmp/vajra-phase9-final -j2
```

---

## 3. Automated Test Suite Execution (CTest)

Execute the 12 comprehensive unit and integration test suites:

```bash
ctest --test-dir /tmp/vajra-phase9-final --output-on-failure
```

### Registered Test Suites:
1. `test_linalg`: Sparse linear algebra (CSC/CSR, SpMV, Sparse LU, Sparse LDL, Ruiz scaling)
2. `test_simplex`: Primal and dual revised simplex algorithms, basis factorization, cycling avoidance
3. `test_io`: MPS reader, LP reader, solution export, JSON escaping, benchmark CSV semantics
4. `test_regression`: KKT residual precision, numerical stability, unbounded/infeasible certificates
5. `test_presolve`: Bound tightening, singleton row/col removal, doubleton equality reductions
6. `test_pdhg`: Restarted Primal-Dual Hybrid Gradient first-order solver, HALPn acceleration
7. `test_reliability`: Ill-conditioned bases, rank deficiency, numerical recovery
8. `test_milp`: Native branch-and-bound, integer feasibility, global bound proof, node limits
9. `test_qp`: Native primal active-set convex QP, augmented nullspace Hessian, KKT multipliers
10. `test_benchmark_harness`: Suite filtering, JSON validation, anti-fabrication assertions
11. `test_benchmark`: Quick smoke benchmark across all suites, telemetry generation
12. `test_python_verifier`: Zero-dependency Python independent mathematical solution audit

---

## 4. Benchmark Execution Commands

The unified benchmark harness `indus_benchmark` evaluates all engine capabilities and generates machine-readable CSV and JSON evidence.

### A. Quick Smoke Test (CI / Quick Evaluation)
```bash
/tmp/vajra-phase9-final/indus_benchmark \
  --quick \
  --suite all \
  --out-dir /tmp/vajra_artifacts \
  --csv /tmp/vajra_artifacts/benchmark_results.csv \
  --json /tmp/vajra_artifacts/benchmark_results.json
```

### B. Full Reference Verification Mode
```bash
/tmp/vajra-phase9-final/indus_benchmark \
  --full \
  --suite all \
  --out-dir /tmp/vajra_artifacts \
  --csv /tmp/vajra_artifacts/benchmark_full_results.csv \
  --json /tmp/vajra_artifacts/benchmark_full_results.json
```

### C. Individual Suite Execution
```bash
# Continuous LP Baseline (22 Netlib / MRPL instances)
/tmp/vajra-phase9-final/indus_benchmark --suite lp --quick --out-dir /tmp/vajra_artifacts

# Native MILP Branch-and-Bound
/tmp/vajra-phase9-final/indus_benchmark --suite milp --quick --out-dir /tmp/vajra_artifacts

# Native Convex Quadratic Programming (QP)
/tmp/vajra-phase9-final/indus_benchmark --suite qp --quick --out-dir /tmp/vajra_artifacts

# First-Order Operator-Splitting (PDHG)
/tmp/vajra-phase9-final/indus_benchmark --suite pdhg --quick --out-dir /tmp/vajra_artifacts

# GPU Hardware Acceleration (CPU Fallback on CPU-only hosts)
/tmp/vajra-phase9-final/indus_benchmark --suite cuda --quick --out-dir /tmp/vajra_artifacts
```

---

## 5. Input Models & Output Artifacts

| Model Name | Problem Class | Path | Description |
| :--- | :--- | :--- | :--- |
| `crude_blend` | LP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps` | MRPL 3-crude CDU distillation margin model |
| `afiro` | LP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps` | Standard Netlib baseline model |
| `share2b` | LP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/share2b.mps` | Netlib baseline model |
| `beaconfd` | LP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/beaconfd.mps` | Medium Netlib test model |
| `supply_chain` | LP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/supply_chain.mps` | Classical degenerate transportation network |
| `ill_conditioned` | LP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/ill_conditioned.mps` | Badly scaled matrix with 22 orders of magnitude |
| `blend_milp` | MILP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend_milp.mps` | Crude blending with discrete CDU mode switch |
| `lot_sizing` | MILP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/lot_sizing.mps` | Multi-period production lot sizing with setup binaries |
| `power_dispatch` | MILP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/power_dispatch.mps` | Cogeneration captive power unit commitment |
| `qp_blend` | QP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/qp_blend.mps` | MRPL convex quadratic blending pool |
| `crude_blend_qp`| QP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend_qp.mps` | Quadratic margin maximization |
| `miqp_blend` | MIQP | `SOVEREIGN_SOLVER_BLUEPRINT/test_models/miqp_blend.mps` | Explicit unsupported classification test |

### Output Artifacts:
- `build/benchmark_results.csv`: Machine-readable tabular telemetry with 51 columns.
- `build/benchmark_results.json`: JSON structure parseable with Python's standard `json` module.
- `build/benchmark_artifacts/*.sol`: Solvers' primal and dual solution vectors.
- `build/benchmark_artifacts/*.json`: Per-model solution certificates.

---

## 6. Verification Contracts & Status Semantics

| Status | Meaning |
| :--- | :--- |
| `OPTIMAL_VERIFIED` | Solver returned `kOptimal` and all mathematical feasibility, bound, stationarity, and reference criteria passed. |
| `FEASIBLE_UNVERIFIED` | Solver found a feasible solution vector, but optimality is unproven or iteration limit was reached. Never claimed as optimal. |
| `LIMIT_REACHED` | Search terminated due to `node_limit`, `time_limit`, or `iteration_limit`. Never claimed as optimal. |
| `INFEASIBLE_VERIFIED` | Model was proven infeasible by Phase I or Farkas certificate. |
| `UNSUPPORTED` | Feature is outside sovereign scope (e.g. Nonconvex QP, MIQP). Honest refusal. |
| `FAILED` | Solver claims did not match mathematical verification or reference objective. |
| `NOT_APPLICABLE` | Synthetic model without reference solution, or limit-reached instance where reference checking does not apply. |

---

## 7. External Comparison & Industrial Demo Scripts

```bash
# Run fair external solver comparison against HiGHS (via SciPy)
python3 scripts/compare_external_solvers.py

# Run MRPL industrial case study demonstration suite
./scripts/run_industrial_demos.sh
```
