# VAJRA-OPT (INDUS-OPT): Indigenous Sovereign Optimization Engine

[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![CTest Coverage](https://img.shields.io/badge/CTest-10%2F10%20Passed%20(100%25)-success.svg)]()
[![Benchmark Verification](https://img.shields.io/badge/Benchmarks-22%2F22%20Verified%20(100%25)-success.svg)]()
[![Phase Status](https://img.shields.io/badge/Phase%207-Native%20MILP%20Complete-blue.svg)]()
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
| **MILP (Mixed-Integer LP)** | Native Branch-and-Bound (`milp`), Deterministic Best-Bound Search, Cold Unpresolved Dual Simplex Relaxations, Most-Fractional Branching, Global Bound Proof | **Fully Verified & Certified (Phase 7)** |
| **Convex QP (Continuous QP)** | Native Primal Active-Set (`qp`), Augmented Nullspace Hessian ($Q + \gamma A_W^T A_W$), Sparse $LDL^T$ Factorization, Exact KKT Stationarity, Ray Detection | **Fully Verified & Certified (Phase 8)** |
| **Presolve / Scaling** | Ruiz Equilibration, Pock-Chambolle, 8-Pass Reversible Presolve (disabled on discrete nodes) | **Fully Verified & Certified** |
| **MIQP / Nonconvex QP** | Automatic Classification (`Model::classify()`), Strict Integrity Guards (`UNSUPPORTED` / `MODEL_ERROR`) | *Explicitly Unsupported in Phase 8* |

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
8. `test_milp` — Native Branch-and-Bound MILP suite (13/13 tests: 0-1 knapsack, multidimensional knapsack, production planning, MIP minimization, integer infeasibility proof, verifier integrality violation rejection, root node optimality, LP iteration-limit handling, node-limit without incumbent, time-limit handling, invalid model metadata validation, unnamed model export and re-verification, and multi-node branching trees).
9. `test_qp` — Native Convex QP suite (21/21 tests: unconstrained analytical QP, 1-var bounded QP, equality constraints, inequality constraints, ranged constraints, sparse multi-variable QP, semidefinite QP with flat direction, objective offset, maximization sign test, asymmetric Q rejection, nonconvex Q rejection, MIQP rejection, infeasible QP, unbounded QP with ray certificate, iteration limit, time limit, NaN/Inf rejection, verifier rejection of corrupt solutions, repeated solve determinism, LP/MILP regression protection, and unnamed QP model serialization).
10. `test_benchmark` — Multi-engine benchmark harness in quick mode (Suite 1 LP baseline, Suite 2 multi-engine telemetry, Suite 3 convex QP benchmark).
11. `test_python_verifier` — Independent pure-Python external audit over benchmark instances with full QUADOBJ support.

---

## 5. Solver CLI (`indus_solve`) & Benchmark Execution

### A. General-Purpose Solver CLI (`indus_solve`)
VAJRA-OPT provides a sovereign, high-performance CLI to solve arbitrary MPS and LP optimization models across Continuous LP, MILP, and Continuous Convex QP:

```bash
# Solve an MPS model with dual simplex and export JSON telemetry:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps --output afiro_sol.json

# Solve a Mixed-Integer Linear Program (MILP) with Branch-and-Bound:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend_milp.mps --algorithm milp --node-limit 50000 --mip-gap 1e-4

# Solve a Continuous Convex Quadratic Program (QP) with Native Active-Set:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/qp_blend.mps --algorithm qp --tol 1e-6 --output qp_blend.json

# Solve an LP model with primal simplex:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.lp --algorithm primal_simplex

# Solve with CPU PDHG, time limit, and export .sol text format:
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/afiro.mps --algorithm pdhg_cpu --time-limit 60 --sol afiro.sol

# Solve requesting CUDA PDHG (cleanly reports CPU fallback if no discrete GPU):
./build-cpu/indus_solve --input SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps --algorithm pdhg_cuda
```

CLI Features:
- Format auto-detection: supports both `.mps` (fixed/free format with QUADOBJ/QMATRIX) and `.lp` algebraic files with integer markers/types.
- Algorithm selection: `dual_simplex` (default auto), `primal_simplex`, `simplex`, `pdhg_cpu`, `pdhg_cuda`, `milp`, `qp`.
- QP parameter controls & conventions:
  - Form: $\min / \max: \frac{1}{2} x^T Q x + c^T x + \text{offset}$
  - Gradient: $\nabla f(x) = Q x + c$
  - Convexity validation via LDLᵀ inertia analysis: $Q \succeq 0$ (minimization), $-Q \succeq 0$ (maximization).
  - Explicit integrity guards: MIQP returns `UNSUPPORTED` (exit code 1); Nonconvex QP returns `MODEL_ERROR` (exit code 1).
- MILP parameter controls: `--node-limit`, `--mip-gap`, `--abs-gap`, `--integer-tol`.
- General parameter controls: `--time-limit`, `--iter-limit`, `--tol`, `--presolve`/`--no-presolve`, `--scaling`/`--no-scaling`.
- Dual export: supports standard `.sol` and structured `.json` solutions with full QP metadata.
- Built-in verification: automatically runs KKT audit on LP models, integrality & global bound proof on MILP models, and stationarity/complementarity/convexity verification on QP models.
- Exit code semantics: returns 0 on verified optimal solution; nonzero on infeasibility, unboundedness, limits, numerical error, or unsupported model class.

### B. Benchmark Execution Modes (`indus_benchmark`)

The multi-engine benchmark harness supports explicit execution contracts:

1. **Quick Smoke Mode (`--quick`):**
   - Validates Suite 1 baseline (22/22 Netlib & MRPL LP models) against independent verifiers.
   - Runs fast telemetry on Suite 2 (multi-engine PDHG) with throttled iterations.
   - Validates Suite 3 (Convex QP Benchmark: `qp_blend`, `crude_blend_qp`, `portfolio_qp_100`) with independent KKT stationarity audit.
   - Guarantees honest status reporting: non-optimal rows (e.g. `ITERATION_LIMIT` or `FEASIBLE`) are never falsely marked as `PASSED`.
   - Exits with code 0 on clean smoke run; used by automated CI/CTest gates.
   ```bash
   ./build-cpu/indus_benchmark --quick --csv build-cpu/benchmark_results.csv
   ```

2. **Full Reference Verification Mode (`--full` / default):**
   - Runs full 50,000-iteration budgets on reference models.
   - Enforces independent mathematical verification and published-objective matching for Suite 2 reference models.
   - Verifies 100% of Suite 3 QP instances.
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
