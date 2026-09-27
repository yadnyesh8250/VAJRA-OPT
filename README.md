# VAJRA-OPT (INDUS-OPT): Indigenous Sovereign Optimization Engine

[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![CTest Coverage](https://img.shields.io/badge/CTest-9%2F9%20Passed%20(100%25)-success.svg)]()
[![Benchmark Verification](https://img.shields.io/badge/Benchmarks-22%2F22%20Verified%20(100%25)-success.svg)]()
[![Phase Status](https://img.shields.io/badge/Phase%206-Hardened%20%26%20Reproducible-blue.svg)]()
[![License](https://img.shields.io/badge/License-Proprietary%20%2F%20MRPL%20Sovereign-red.svg)]()

> **MRPL Problem Statement 26119:** Development of an indigenous, autonomous mathematical optimization solver for continuous and mixed-integer industrial operations, refinery scheduling, and resource allocation without foreign dependencies or telemetry.

---

## 1. Architectural Overview

VAJRA-OPT is an indigenously developed, self-contained optimization solver engineered in modern C++17 with optional CUDA GPU acceleration.

```
                           +----------------------------------------+
                           |           Public Interfaces            |
                           |  C++ API, CLI Tools, Python Auditor    |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |      Model & IO System (MPS / LP)      |
                           |     Dimension & Bound Input Guard      |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |  Presolve Engine (8 Reversible Passes) |
                           |  (Empty, Singleton, Forcing, Doubleton)|
                           +-------------------+--------------------+
                                               |
                                               v
                        +----------------------+----------------------+
                        |                                             |
                        v                                             v
        +-------------------------------+             +-------------------------------+
        |    Simplex Engine (CPU)       |             |   First-Order Engine (PDHG)   |
        | - Primal Simplex (Phase 1/2)  |             | - Adaptive Spectral Stepsize  |
        | - Dual Simplex (Devex)        |             | - Moreau Envelope Proximal    |
        | - Harris Two-Pass Ratio Test  |             | - CPU Reference & CUDA Kernel |
        | - Sparse Markowitz LU Factor  |             | - Graceful Fallback on Non-GPU|
        +---------------+---------------+             +---------------+---------------+
                        |                                             |
                        +----------------------+----------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |      Reversible Postsolve Stack        |
                           |  Exact Primal / Dual Reconstruction    |
                           +-------------------+--------------------+
                                               |
                                               v
                           +----------------------------------------+
                           |      Sovereign Verification Spine      |
                           | Primal/Dual KKT & Complementarity Guard|
                           +----------------------------------------+
```

### Core Subsystems:
1. **Sparse Linear Algebra Core (`src/linalg/`)**
   - High-performance Compressed Sparse Column (CSC) and Compressed Sparse Row (CSR) representations.
   - Threshold-pivoting Markowitz Sparse LU Factorization ($u=0.01$) with Product Form of Inverse (PFI) eta updates.
   - Sparse $LDL^T$ factorization with Approximate Minimum Degree (AMD) quotient graph reordering.
   - Matrix equilibration: Ruiz $\ell_\infty$ scaling and Pock-Chambolle diagonal preconditioning.
2. **Simplex Core (`src/solvers/simplex/`)**
   - Bounded Dual Simplex with Devex steepness pricing and bound-flipping ratio test.
   - Bounded Primal Simplex with composite Phase-1 objective and Harris two-pass ratio test.
   - Exact Farkas certificates of infeasibility and primal unbounded ray extraction.
3. **Presolve Engine & Postsolve Stack (`src/presolve/`)**
   - 8 reversible reduction passes: empty row/col, fixed variables, singleton rows, forcing constraints, redundant constraints, free variable singletons, and degree-$\le 2$ doubleton row substitutions.
   - Strict reverse-LIFO postsolve unwinding guaranteeing exact primal and dual feasibility on the original model.
4. **First-Order GPU / Large-Scale Core (`src/solvers/pdhg/`, `src/solvers/gpu/`)**
   - Restarted Primal-Dual Hybrid Gradient (PDHG / Chambolle-Pock) with closed-form Moreau proximal projections.
   - Adaptive step sizing via power-iteration spectral norm estimation and Halpern extrapolation.
   - Enforced zero-vector-transfer VRAM-resident CUDA iteration loop with device-side atomic residual reductions.
   - Seamless CPU fallback when CUDA hardware or compilation is unavailable.
5. **Independent Verification Spine (`src/engine/verifier.cpp`, `validator/independent_verifier.py`)**
   - Standalone C++ and pure-Python zero-dependency validators auditing primal feasibility, dual feasibility, bounds, complementarity, and objective matching without touching solver internals.

---

## 2. Supported Problem Classes & Algorithms

| Problem Class | Implemented Algorithms | Status |
|:---|:---|:---|
| **Continuous LP** | Bounded Dual Simplex (Devex), Bounded Primal Simplex (Harris), Restarted PDHG (CPU & CUDA) | **Fully Verified & Certified** |
| **Presolve / Scaling** | Ruiz Equilibration, Pock-Chambolle, 8-Pass Reversible Presolve | **Fully Verified & Certified** |
| **MILP / QP / MIQP** | Automatic Classification (`Model::classify()`), Status Guard Rejection | *Architecturally Classified; Solver Scope Unexpanded in Phase 6* |

---

## 3. Build Instructions

### Prerequisites
- Modern C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
- CMake 3.18 or higher
- Python 3.8+ (for independent verification auditor)
- *(Optional)* NVIDIA CUDA Toolkit 11.0+ and `nvcc` (for physical GPU execution)

### A. Clean CPU-Only Build (Default / Universal)
```bash
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DINDUS_ENABLE_CUDA=OFF

cmake --build build-cpu -j4
```

### B. CUDA Build (with Graceful CPU Fallback)
```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DINDUS_ENABLE_CUDA=ON

cmake --build build-cuda -j4
```
*Note:* If `nvcc` is not installed or detected on your system, CMake will output a clear notice and safely configure in CPU-only mode.

---

## 4. Test Suite Execution

Run the complete CTest test suite:
```bash
ctest --test-dir build-cpu --output-on-failure
```

### Registered Test Suites:
1. `test_linalg` — Sparse CSC/CSR arithmetic, Markowitz LU, $LDL^T$, and Ruiz scaling.
2. `test_simplex` — Primal/Dual simplex, textbook LPs, cycling, and ray certificates.
3. `test_io` — MPS and LP format parsing, writing, and solution serialization.
4. `test_regression` — 9 critical regression models (free vars, ranged rows, Farkas, Beale).
5. `test_presolve` — 16 presolve reduction tests with full dual postsolve validation.
6. `test_pdhg` — 11 PDHG tests, including all 12 CUDA kernel specs verified mathematically.
7. `test_reliability` — 10 hardening and boundary tests (empty models, 0-var, 0-row, malformed bounds, limits, repeated use, fallback).
8. `test_benchmark` — Multi-engine benchmark harness in quick mode.
9. `test_python_verifier` — Independent pure-Python external audit over benchmark instances.

---

## 5. Solver CLI (`indus_solve`) & Benchmark Execution

### A. General-Purpose Solver CLI (`indus_solve`)
VAJRA-OPT provides a sovereign, high-performance CLI to solve arbitrary MPS and LP optimization models:

```bash
# Solve an MPS model with dual simplex and export JSON telemetry:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps --output afiro_sol.json

# Solve an LP model with primal simplex:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp --algorithm primal_simplex

# Solve with CPU PDHG, time limit, and export .sol text format:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps --algorithm pdhg_cpu --time-limit 60 --sol afiro.sol

# Solve requesting CUDA PDHG (cleanly reports CPU fallback if no discrete GPU):
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps --algorithm pdhg_cuda
```

CLI Features:
- Format auto-detection: supports both `.mps` (fixed/free format) and `.lp` algebraic files.
- Algorithm selection: `dual_simplex` (default auto), `primal_simplex`, `simplex`, `pdhg_cpu`, `pdhg_cuda`.
- Parameter controls: `--time-limit`, `--iter-limit`, `--tol`, `--presolve`/`--no-presolve`, `--scaling`/`--no-scaling`.
- Dual export: supports standard `.sol` and structured `.json` solutions.
- Built-in verification: automatically runs KKT and complementarity audit on solution vectors.
- Exit code semantics: returns 0 on verified optimal solution; nonzero on infeasibility, unboundedness, limits, or numerical error.

### B. Benchmark Execution Modes (`indus_benchmark`)

The multi-engine benchmark harness supports explicit execution contracts:

1. **Quick Smoke Mode (`--quick`):**
   - Validates Suite 1 baseline (22/22 Netlib & MRPL LP models) against independent verifiers.
   - Runs fast telemetry on Suite 2 (multi-engine PDHG) with throttled iterations.
   - Guarantees honest status reporting: non-optimal rows (e.g. `ITERATION_LIMIT` or `FEASIBLE`) are never falsely marked as `PASSED`.
   - Exits with code 0 on clean smoke run; used by automated CI/CTest gates.
   ```bash
   ./build-cpu/indus_benchmark --quick --csv build-cpu/benchmark_results.csv
   ```

2. **Full Reference Verification Mode (`--full` / default):**
   - Runs full 50,000-iteration budgets on reference models.
   - Enforces independent mathematical verification and published-objective matching for Suite 2 reference models.
   - Exits nonzero if any reference model fails verification or convergence limits.
   ```bash
   ./build-cpu/indus_benchmark --full --csv build-cpu/benchmark_results.csv
   ```

### C. Independent Solution Verification Tool
You can independently verify any solution file (`.sol`) against its model (`.mps` or `.lp`):

**Using C++ Audit CLI:**
```bash
./build-cpu/indus_verifier SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp artifacts/solutions/crude_blend.sol --tol 1e-4
```

**Using Pure Python Auditor (Zero Dependencies):**
```bash
python3 validator/independent_verifier.py SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp artifacts/solutions/crude_blend.sol 1e-4
```

---

## 6. CUDA Hardware Validation Status

- **CUDA Source Code:** Fully implemented in `src/solvers/gpu/pdhg_cuda.cu` and `spmv_kernels.cuh`.
- **Warp SpMV & Device Reductions:** Complete and active.
- **Mathematical Specification:** 100% verified via CPU mathematical equivalence tests in `test_pdhg.cpp`.
- **Host-Device Transfer Pipeline:** Enforced zero-transfer loop (transfers only 48-byte diagnostics struct per check interval).
- **Graceful Fallback:** Verified on CPU-only machines. If CUDA is disabled or no NVIDIA device exists, the solver falls back smoothly to CPU Restarted PDHG.
- **Physical GPU Execution:** **NOT HARDWARE VALIDATED / PENDING NVIDIA TESTBED.** The current build host is an Apple Silicon Mac without an NVIDIA GPU or `nvcc`. No physical speedup or hardware execution is claimed without discrete NVIDIA hardware validation.

---

## 7. Known Platform Differences & Diagnostics

1. **Windows Pathing and Encodings:**
   - Python on Windows uses system-dependent default encodings (`cp1252`), which caused UTF-8 character crashes. `validator/independent_verifier.py` explicitly enforces `encoding="utf-8", errors="replace"`.
2. **CTest Execution Timeouts:**
   - Category 5 in the benchmark suite solves a 10,000 &times; 20,000 synthetic LP for 2,001 iterations. On throttled or power-saving CPU configurations (e.g. Windows laptops on battery), this takes >90s. CTest target `test_benchmark` has been configured with `--quick` mode and an explicit 600-second timeout to prevent false timeout failures.
3. **Empty Variable / Row Names:**
   - When models are constructed programmatically without variable or row names, name lookup helpers safely default to `"c<index>"` and `"r<index>"`, preventing memory faults.

---

## 8. Sovereign Engineering Standards
- **Zero Foreign Solvers:** 100% self-contained codebase; zero calls to HiGHS, GLPK, CBC, Gurobi, or CPLEX.
- **No Third-Party Bloat:** Standard C++ STL and zero required external libraries.
- **Audit Rigor:** No model is marked `OPTIMAL` unless verified by the independent verification spine against KKT conditions.
