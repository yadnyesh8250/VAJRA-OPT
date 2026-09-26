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

- [X] **PHASE 3: High-Performance IO, Model Classes & Sovereign Verification Spine**
  - [X] `include/indus/model.hpp`, `options.hpp`
  - [X] `src/engine/model.cpp` (Model building, classification, validation, Status Guard with auto-downgrade, dispatcher seam)
  - [X] `src/io/mps_reader.cpp` (Fixed/Free MPS parser with RANGES, OBJSENSE, BOUNDS & QUADOBJ)
  - [X] `src/io/lp_reader.cpp` (CPLEX algebraic LP format parser with range and bounds support)
  - [X] `src/io/writer.cpp` (Standardized 17-digit precision `.sol` output and enriched JSON telemetry with git commit)
  - [X] `include/indus/verifier.hpp`, `src/engine/verifier.cpp` (Strict disk-based solution verifier checking primal feasibility, dual feasibility, bounds, complementarity, and objective without solver internals; rejects unknown variables, duplicates, missing entries, malformed floats)
  - [X] `validator/independent_verifier.py` (Zero-dependency pure-Python independent sovereign auditor with its OWN MPS and .sol parsers and KKT certificate checks)
  - [X] `include/indus/presolve_lite.hpp`, `src/presolve/presolve_lite.cpp` (Empty rows, fixed cols, singleton rows with dual postsolve reconstruction, crossed bound detection)
  - [X] `tests/test_regression.cpp` (9/9 regression tests: free vars, ranged rows, equality rows, negative bounds, Farkas infeasibility, ray unboundedness, Beale cycling, ill-conditioned matrix scaling, strict solution parsing)
  - [X] `apps/benchmark_runner.cpp` (Expanded 22-model benchmark harness emitting `build/benchmark_results.csv` and artifacts)
  - *Gate Check:* 100% of benchmark instances pass independent verification (22/22 Netlib and MRPL instances pass Netlib LP optimality criteria rel err $\le 10^{-6}$; 21/22 models achieve rel err $< 10^{-11}$; `scagr7` achieves $2.44 \times 10^{-7}$); 6/6 CTest test suites pass cleanly from normal build directory; 0 foreign solver symbols [PASSED].

- [X] **PHASE 4: Presolve Engine & Reversible Postsolve Stack**
  - [X] `include/indus/presolve.hpp` (Reversible LIFO reduction stack `PresolveStack`, `ReductionRecord`, `ReductionType`, index mappings `reduced_to_orig_col`, `reduced_to_orig_row`)
  - [X] `src/presolve/presolve.cpp` (All 8 reversible LP reductions fully active and reachable: empty rows, empty columns, fixed columns, singleton rows, forcing rows with active bound extraction, redundant rows with strict inactivity checks, free-variable column singletons, and mathematically safe doubleton row substitutions $a_1 x_1 + a_2 x_2 = b$)
  - [X] Safe doubleton equality substitution: verifies finite non-zero coefficients ($|a| \ge 10^{-6}$, conditioning ratio $\le 10^4$), requires clean uncoupled column structure (`orig.A.col_rows(col).size() == col_mat[col].size()` with all rows active), restricts substitution degree to $\le 2$ to prevent fill-in cascade, tracks `row_modified` to prevent circular chaining, updates objective offset and row bounds, and propagates implied bounds with crossed-bound infeasibility detection.
  - [X] Exact reversible dual postsolve: unwinds in strict reverse LIFO order; reconstructs primal values $x_1 = (b - a_2 x_2)/a_1$; restores row activities; reconstructs row duals using tracked `rec.obj_cost` and sense factor; performs dual pivoting ($y_i = (c_2 - a_2^T y_{\ne i}) / a_2$) when the retained variable was interior in the original model and hit an implied bound, guaranteeing exact dual feasibility and complementarity.
  - [X] Integration into `indus::solve()` in `src/engine/model.cpp`: presolved reduced solve followed by complete postsolve unwinding and strict verification against the original model; never reports `kOptimal` unless original model passes primal, dual, objective, and complementarity checks.
  - [X] `tests/test_presolve.cpp` (16/16 regression tests with explicit assertions active in Release builds covering: empty redundant/infeasible rows, fixed variable substitution, singleton row bound tightening, forcing constraints, redundant rows, free variable elimination, multi-pass fixed-point cascading, ill-conditioned numerical safety, basic doubleton equality substitution, doubleton substitution with eliminated variable in another row, objective preservation before and after presolve, bound propagation through doubleton equality, infeasible implied bounds, chained multi-reduction cascade, and dual feasibility/complementarity validation).
  - [X] Observability: Exposed doubleton reduction count, original vs presolved dimensions, primal/dual violations, and objective discrepancy in `indus_benchmark` stdout table, `benchmark_results.csv`, and JSON telemetry.
  - [X] Numerical Limitations Honestly Documented: Doubleton reductions on coupled columns (degree $> 2$ in active matrix, or where columns were modified by earlier substitutions in the same pass) and ill-conditioned pairs (ratio $> 10^4$ or pivot $< 10^{-4}$) are conservatively preserved rather than heuristically eliminated, guaranteeing zero dual drift across all models.
  - *Gate Check:* Model dimensions demonstrably reduced (e.g. `bandm` 305x472 -> 222x377 with 25 doubletons, `scagr7` 129x140 -> 90x134 with 5 doubletons, `lotfi` 153x308 -> 128x294 with 6 doubletons, `beaconfd` 173x262 -> 109x170 with 4 doubletons, `sc50a` 50x48 -> 47x46 with 2 doubletons); postsolve reconstructs exact primal and dual solutions; 100% of benchmark instances pass independent verification (22/22 Netlib and MRPL models pass; 21/22 models achieve rel err $< 10^{-11}$; `scagr7` achieves $2.44 \times 10^{-7}$); 7/7 CTest test suites pass cleanly; independent pure-Python auditor passes 22/22; zero foreign solver symbols [PASSED].

- [-] **PHASE 5: GPU CUDA SpMV Acceleration & Restarted PDHG (IN PROGRESS — CPU Verified, Real GPU Hardware Pending)**
  - [X] `include/indus/pdhg.hpp`, `src/solvers/pdhg/pdhg.cpp` (CPU Restarted PDHG solver with closed-form Moreau proximal projection, adaptive step sizing via spectral norm power iteration, Halpern extrapolation, and residual-based adaptive restarts)
  - [X] `include/indus/gpu.hpp`, `src/solvers/gpu/gpu_context.cpp` (Hardware device capability prober and clean CPU fallback with zero runtime crash)
  - [X] `src/solvers/gpu/spmv_kernels.cuh` (Warp-aggregated CUDA SpMV, transpose SpMV, branchless dual Moreau update, primal Halpern extrapolation, and device-side atomic residual reduction kernel)
  - [X] `src/solvers/gpu/pdhg_cuda.cu` (Zero-host-transfer VRAM-resident GPU iteration pipeline with RAII `CudaBuffer` memory management and 16-byte scalar residual checks with zero intermediate vector transfers)
  - [X] Optional CMake compilation controlled by `INDUS_ENABLE_CUDA` (detects CUDA compiler; falls back safely to CPU reference if absent)
  - [X] Integration into `indus::solve()` in `src/engine/model.cpp`: dispatch support for `pdhg`, `pdhg_cpu`, `pdhg_cuda`, model class guards rejecting QP/MIP, and original-model Status Guard verification
  - [X] `tests/test_pdhg.cpp` (10/10 tests passing: 2-var known LP, equality constraints, mixed <=/>=/ranged constraints, free & box variables, min/max symmetry, presolve+PDHG+postsolve, infeasible detection, iteration limit, Netlib objective matching with simplex, and hardware probe / CPU fallback)
  - [X] Multi-engine benchmark suite in `apps/benchmark_runner.cpp` with synthetic large sparse LP (94,889 nonzeros) and honest reporting of hardware context (reported: "CUDA implementation compiled but hardware speedup not measured" when running on non-NVIDIA host)
  - [ ] Physical NVIDIA Hardware Gate: Compilation with `nvcc` and execution on genuine NVIDIA GPU hardware (e.g., RTX 4090 / A100 / T4) to benchmark real physical warp performance, VRAM occupancy, and measure $15\times-50\times$ speedup.
  - *Current Status:* **IN PROGRESS (CPU Reference Engine 100% verified, CUDA pipeline structured with warp SpMV & device-side residuals; physical hardware execution pending deployment to NVIDIA CUDA testbed).**

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
