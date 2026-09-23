# PHASE IMPLEMENTATION PROGRESS TRACKER: INDUS-OPT

> **INSTRUCTIONS FOR AI AGENT:**  
> Update this checklist as you complete each phase. Do not mark a phase as completed until its gate check has been verified by compilation or test execution.

---

## Overall Implementation Status

- [X] **PHASE 1: Linear Algebra Core & Factorization**
  - [X] `include/indus/types.hpp`, `tolerances.hpp`, `sparse.hpp`
  - [X] `src/linalg/sparse.cpp` (CSC/CSR matrix operations & transpose)
  - [X] `src/linalg/lu.cpp` (Sparse Markowitz LU with threshold stability $u=0.01$, PFI eta file)
  - [X] `src/linalg/ldl.cpp` (Sparse LDLᵀ with AMD quotient graph ordering)
  - [X] `src/linalg/scaling.cpp` (Ruiz $\ell_\infty$ equilibration & Pock-Chambolle preconditioning)
  - *Gate Check:* Tested sparse LU factorization & FTRAN/BTRAN back-substitution [PASSED: 100% test coverage, residual < 1e-12].

- [X] **PHASE 2: Bounded Simplex Engine (Continuous LP)**
  - [X] `src/solvers/simplex/simplex_core.hpp` (Basis state, factors, pricing weights)
  - [X] `src/solvers/simplex/dual_simplex.cpp` (Dual simplex, Devex pricing, bound-flipping ratio test)
  - [X] `src/solvers/simplex/primal_simplex.cpp` (Primal simplex, composite Phase-1, Harris ratio test)
  - *Gate Check:* Solves textbook $2 \times 2$ and $3 \times 3$ LPs with verified optimality [PASSED: 100% test coverage, anti-cycling Harris verified, Farkas/Ray certificates verified].

- [ ] **PHASE 3: High-Performance IO & Model Classes**
  - [ ] `include/indus/model.hpp`, `options.hpp`
  - [ ] `src/engine/model.cpp` (Model building, classification, and validation)
  - [ ] `src/io/mps_reader.cpp` (Fixed/Free MPS parser with RANGES & QUADOBJ)
  - [ ] `src/io/lp_reader.cpp` (CPLEX algebraic LP format parser)
  - [ ] `src/io/writer.cpp` (`.sol` solution and `.json` stats exporter)
  - *Gate Check:* Parses `test_models/crude_blend.mps` without errors.

- [ ] **PHASE 4: Presolve Engine & Reversible Postsolve Stack**
  - [ ] `src/presolve/presolve.hpp` (Reduction stack)
  - [ ] `src/presolve/presolve.cpp` (8 reductions: empty/fixed/singleton/forcing/redundant/free/doubleton)
  - [ ] Dual fixed-point postsolve reconstruction
  - *Gate Check:* Model dimensions reduced; postsolve yields dual feasible point on original model.

- [ ] **PHASE 5: GPU CUDA SpMV Acceleration & Restarted PDHG**
  - [ ] `src/solvers/pdhg/pdhg.cpp` (CPU Restarted PDHG with adaptive restarts)
  - [ ] `src/solvers/gpu/spmv_kernels.cuh` (Warp-aggregated CUDA SpMV & Transposed SpMV)
  - [ ] `src/solvers/gpu/pdhg_cuda.cu` (Zero-host-transfer GPU iteration pipeline)
  - [ ] Smooth CPU fallback when CUDA device is absent
  - *Gate Check:* Benchmarked on large model with verified speedup and matching objective.

- [ ] **PHASE 6: Interior Point, Convex QP & Branch-and-Bound**
  - [ ] `src/solvers/interior_point/ipm.cpp` (Mehrotra predictor-corrector over normal equations)
  - [ ] `src/solvers/quadratic/convexity.cpp`, `qp_condat_vu.cpp` (Condat-Vũ QP + LDLᵀ convexity check)
  - [ ] `src/solvers/branch_bound/branch_and_bound.cpp` (Reliability branching, strong branching, root dive)
  - [ ] `src/solvers/branch_bound/cuts.cpp` (Gomory mixed-integer & knapsack cover cuts)
  - *Gate Check:* Solves `crude_blend_qp.mps` (QP) and `blend_milp.mps` (MILP) to certified optima.

- [ ] **PHASE 7: CLI, Pyomo Plugin & Netlib Benchmarks**
  - [ ] `apps/indus-cli/main.cpp` (Full CLI interface)
  - [ ] `src/interfaces/capi/c_api.cpp` (Pure C ABI wrapper)
  - [ ] `src/interfaces/python/indus_opt/pyomo_adapter.py` (Pyomo solver plugin)
  - [ ] Root `CMakeLists.txt`
  - [ ] Execute `bench/runners/netlib_runner.py` on `test_models/`
  - *Gate Check:* 100% of benchmark instances pass with relative gap $\le 10^{-6}$ and independent KKT verification!
