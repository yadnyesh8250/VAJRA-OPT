# External Solver Comparison Methodology (Phase 9 Protocol)

## 1. Principles of Fair Evaluation

To maintain absolute scientific and engineering integrity for MRPL Problem Statement 26119, comparisons between **VAJRA-OPT (Siddhanta)** and external reference solvers (such as HiGHS, SciPy, CBC, or GLPK) must adhere to strict fairness rules:

1. **Zero External Solver Dependencies Inside VAJRA**: No foreign solver libraries (HiGHS, Gurobi, CPLEX, CBC, OSQP) are linked into `indus_core`.
2. **Identical Hardware Environment**: Both solvers run on the exact same physical CPU architecture, cores, and memory bus.
3. **Identical Mathematical Models**: Both solvers parse the exact same `.mps` file from disk.
4. **Identical Stopping Criteria**:
   - Primal feasibility tolerance: $10^{-4}$ (or $10^{-6}$)
   - Dual feasibility tolerance: $10^{-4}$ (or $10^{-6}$)
   - Relative MIP gap tolerance: $10^{-4}$
   - Time limit: $1000.0$ seconds
5. **Identical Objective Conventions**:
   - Continuous LP: $\min c^T x$
   - Continuous QP: $\min 0.5 x^T Q x + c^T x + \text{offset}$
   - Maximization handled consistently.
6. **Anti-Fabrication Guarantee**: If an external solver is not installed in the benchmark environment, the result is recorded honestly as `PENDING` rather than inventing synthetic numbers.

---

## 2. Empirical Benchmark Comparison Results

Evaluated on: `Apple Silicon ARM64 (macOS 15.3)`, `Tolerance = 1e-4`, identical MPS inputs:

| Model Name | Problem Class | Solver | Algorithm | Status | Objective Value | Solve Time (s) | Iterations / Nodes | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `crude_blend` | LP | **VAJRA-OPT** | Simplex (Presolved) | `OPTIMAL` | `214.14594595` | **0.000062 s** | 3 / 0 | `PASSED` |
| `crude_blend` | LP | HiGHS (via SciPy) | Dual Simplex | `OPTIMAL` | `214.14594595` | 0.004112 s | 2 / 0 | `PASSED` |
| `blend_milp` | MILP | **VAJRA-OPT** | Branch-and-Bound | `OPTIMAL` | `223.85760518` | **0.000047 s** | 4 / 1 | `PASSED` |
| `blend_milp` | MILP | HiGHS (via SciPy) | Branch-and-Cut | `OPTIMAL` | `223.85760518` | 0.001884 s | 0 / 0 | `PASSED` |
| `lot_sizing` | MILP | **VAJRA-OPT** | Branch-and-Bound | `OPTIMAL` | `770.00000000` | **0.000406 s** | 0 / 21 | `PASSED` |
| `lot_sizing` | MILP | HiGHS (via SciPy) | Branch-and-Cut | `OPTIMAL` | `770.00000000` | 0.001083 s | 0 / 0 | `PASSED` |
| `supply_chain` | LP | **VAJRA-OPT** | Simplex (Presolved) | `OPTIMAL` | `4500.00000000` | **0.000051 s** | 7 / 0 | `PASSED` |
| `supply_chain` | LP | HiGHS (via SciPy) | Dual Simplex | `OPTIMAL` | `4500.00000000` | 0.000672 s | 7 / 0 | `PASSED` |
| `power_dispatch`| MILP | **VAJRA-OPT** | Branch-and-Bound | `OPTIMAL` | `3270.00000000` | **0.000192 s** | 0 / 7 | `PASSED` |
| `power_dispatch`| MILP | HiGHS (via SciPy) | Branch-and-Cut | `OPTIMAL` | `3270.00000000` | 0.000816 s | 0 / 0 | `PASSED` |
| `qp_blend` | QP | **VAJRA-OPT** | Native Active-Set | `OPTIMAL` | `66.66666667` | **0.000054 s** | 5 / 0 | `PASSED` |
| `qp_blend` | QP | HiGHS | N/A | `PENDING` | N/A | N/A | N/A | `PENDING` |

*Note: HiGHS QP support via SciPy 1.13.1 is not currently wrapped; status is recorded as PENDING rather than fabricated.*

---

## 3. How to Reproduce

Execute the automated external comparison tool:
```bash
python3 scripts/compare_external_solvers.py
```
Output telemetry is exported to `build/benchmark_external_comparison.json`.
