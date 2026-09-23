# AI AGENT EXECUTION PROTOCOL & OPERATING RULES: PROJECT INDUS-OPT

> **MANDATORY INSTRUCTION FOR THE AI AGENT / DEVELOPER:**  
> You are tasked with engineering **INDUS-OPT**, an enterprise-grade, sovereign mathematical optimization solver for **Problem Statement SIH26119 (Mangalore Refinery and Petrochemicals Limited - MRPL)**.  
> **You MUST read and strictly adhere to this document before writing any code.**

---

## 1. The 5 Anti-Shortcut / Anti-Lazy Rules (Strictly Enforced)

1. **NO MOCK / PLACEHOLDER IMPLEMENTATIONS:**
   * Never write `// TODO: implement later`, `throw std::runtime_error("Not implemented")`, or stub out mathematical routines with dummy returns.
   * Every algorithm (Sparse LU, Devex, Bound-flipping, Mehrotra IPM, PDHG, Branch & Bound) must be fully fleshed out with complete mathematical arithmetic.
2. **ZERO EXTERNAL SOLVER CODE (The Sovereign Red Line):**
   * Do NOT link, include, or copy from CPLEX, Gurobi, Xpress, HiGHS, SCIP, GLPK, Clp, Cbc, or OR-Tools. The binary will be audited with `nm` and `ldd`.
3. **DO NOT ATTEMPT TO GENERATE THE ENTIRE SOLVER IN ONE TURN:**
   * A full solver is ~15,000–20,000 lines of C++ across 40 files. Attempting to write everything at once causes token exhaustion, truncated files, and broken builds.
   * **You MUST execute Phase-by-Phase (Phases 1 through 7).** Only proceed to the next phase after compiling and verifying the current phase.
4. **NO MAGIC NUMBERS IN NUMERICAL CODE:**
   * Every pivot tolerance, epsilon drop, integrality bound, and gap must be imported from `include/indus/tolerances.hpp`.
5. **UPDATE THE PROGRESS TRACKER:**
   * After completing each phase, you must mark it as `[X] DONE` in `PHASE_PROGRESS_TRACKER.md`.

---

## 2. The 7-Phase Step-by-Step Implementation Sequence

```
                               PHASE 1: Linear Algebra Core
                               (Sparse CSC/CSR, Markowitz LU, LDLᵀ)
                                                │
                                                ▼
                               PHASE 2: Bounded Simplex Engine
                               (Dual Simplex, Devex, Primal Simplex)
                                                │
                                                ▼
                               PHASE 3: High-Performance IO
                               (MPS Fixed/Free, CPLEX LP, .sol Writer)
                                                │
                                                ▼
                               PHASE 4: Presolve & Postsolve Stack
                               (8 Reductions, Dual Fixed-Point Recovery)
                                                │
                                                ▼
                               PHASE 5: GPU CUDA SpMV Acceleration
                               (Warp SpMV Kernels, VRAM Iteration Loop)
                                                │
                                                ▼
                               PHASE 6: IPM, Convex QP & Branch-and-Bound
                               (Mehrotra, Condat-Vũ, Reliability Branching)
                                                │
                                                ▼
                               PHASE 7: CLI, Pyomo Plugin & Verification
                               (CLI Executable, Pyomo Adapter, Netlib Run)
```

---

### Detailed Phase Specifications

#### PHASE 1: Linear Algebra Core
* **Headers:** `include/indus/types.hpp`, `tolerances.hpp`, `sparse.hpp`, `version.hpp`.
* **Sources:** `src/linalg/sparse.cpp`, `src/linalg/lu.cpp` (Markowitz threshold $u=0.01$, PFI eta file, hyper-sparse FTRAN/BTRAN), `src/linalg/ldl.cpp` (AMD quotient graph ordering), `src/linalg/scaling.cpp` (Ruiz & Pock-Chambolle scaling).
* **Gate Check:** Compile with `g++ -std=c++20` or CMake. Verify that a $5 \times 5$ singular matrix triggers rank repair and that FTRAN solves $B x = b$ with residual $< 10^{-12}$.

#### PHASE 2: Bounded Revised Simplex Engine
* **Headers:** `src/solvers/simplex/simplex_core.hpp`.
* **Sources:** 
  * `src/solvers/simplex/dual_simplex.cpp`: Bounded dual simplex, dual Devex pricing weights, bound-flipping ratio test, degenerate cost perturbation.
  * `src/solvers/simplex/primal_simplex.cpp`: Composite Phase-1 (no big-M), Harris two-pass ratio test.
* **Gate Check:** Successfully solve textbook $2 \times 2$ LP to optimal objective.

#### PHASE 3: High-Performance IO & Model Classes
* **Headers:** `include/indus/model.hpp`, `options.hpp`.
* **Sources:** `src/engine/model.cpp`, `src/engine/options.cpp`, `src/io/mps_reader.cpp`, `src/io/lp_reader.cpp`, `src/io/writer.cpp`.
* **Gate Check:** Successfully read `test_models/crude_blend.mps` and verify row and column count matches (12 rows, 18 columns).

#### PHASE 4: Presolve Engine & Reversible Postsolve
* **Headers:** `src/presolve/presolve.hpp`.
* **Sources:** `src/presolve/presolve.cpp`.
* **Tasks:** Implement all 8 reductions (empty rows/cols, fixed cols, singleton rows, forcing rows, redundant rows, free column singletons, doubletons) and the dual fixed-point reconstruction loop.
* **Gate Check:** Postsolve reconstructed dual vector satisfies $c - A^T y \ge -10^{-7}$ on reduced model.

#### PHASE 5: GPU CUDA SpMV Acceleration Layer
* **Headers:** `include/indus/gpu.hpp`, `src/solvers/gpu/spmv_kernels.cuh`.
* **Sources:** `src/solvers/pdhg/pdhg.cpp` (CPU reference engine), `src/solvers/gpu/pdhg_cuda.cu` (NVIDIA CUDA VRAM iteration loop).
* **Tasks:** Allocate matrix in device memory, run 10,000 iterations without CPU-GPU synchronization, verify $15\times-50\times$ speedup on large models. Include CPU fallback when CUDA is absent.

#### PHASE 6: Interior Point, Convex QP & Branch-and-Bound
* **Sources:**
  * `src/solvers/interior_point/ipm.cpp`: Mehrotra predictor-corrector on normal equations $A \Theta A^T$.
  * `src/solvers/quadratic/convexity.cpp`, `qp_condat_vu.cpp`: Condat-Vũ QP with LDLᵀ positive semi-definiteness check (reject non-convex with negative pivot certificate).
  * `src/solvers/branch_bound/branch_and_bound.cpp`: Reliability branching (pseudocosts + strong branching), root diving heuristic, warm-started dual node solves.
* **Gate Check:** Solve `test_models/crude_blend_qp.mps` and `test_models/blend_milp.mps` to verified optimality.

#### PHASE 7: CLI, Pyomo Plugin & Netlib Benchmarking
* **Sources:**
  * `apps/indus-cli/main.cpp`: Complete CLI tool supporting `solve`, `info`, `options`, `--write-sol`, `--gpu`.
  * `src/interfaces/python/indus_opt/pyomo_adapter.py`: Drop-in Pyomo/PuLP solver class.
  * Root `CMakeLists.txt`.
* **Gate Check:** Run `python3 bench/runners/netlib_runner.py` against all models in `test_models/`. All models must pass with relative gap $\le 10^{-6}$ and independent KKT verification!

---

## 3. How to Prompt the Agent for Each Phase

Copy and paste these exact prompts to guide the agent phase by phase:

* **Prompt for Phase 1:**  
  `"Agent: Proceed with PHASE 1 of AGENT_EXECUTION_PROTOCOL.md. Implement include/indus/types.hpp, tolerances.hpp, sparse.hpp, version.hpp, and src/linalg/ (sparse.cpp, lu.cpp, ldl.cpp, scaling.cpp). Follow the exact algorithms in 02_SPARSE_LINEAR_ALGEBRA_AND_PRESOLVE.md. Do not stub or skip any math. Report when complete."`
* **Prompt for Phase 2:**  
  `"Agent: Proceed with PHASE 2. Implement the Revised Bounded Simplex Engine in src/solvers/simplex/ (dual_simplex.cpp, primal_simplex.cpp, simplex_core.hpp). Follow 01_MATHEMATICAL_ENGINES_AND_ALGORITHMS.md for Devex pricing, bound-flipping, and Harris ratio test."`
* **Prompt for Phase 3:**  
  `"Agent: Proceed with PHASE 3. Implement the Model, Options, and MPS/LP readers in src/engine/ and src/io/ according to 06_CORE_CPP_HEADER_INTERFACES.md and 07_MPS_LP_PARSER_SPECIFICATION.md."`
* **Prompt for Phase 4:**  
  `"Agent: Proceed with PHASE 4. Implement the Presolve Engine and Dual Fixed-Point Postsolve in src/presolve/ according to 02_SPARSE_LINEAR_ALGEBRA_AND_PRESOLVE.md."`
* **Prompt for Phase 5:**  
  `"Agent: Proceed with PHASE 5. Implement the Restarted PDHG and the GPU CUDA acceleration layer in src/solvers/pdhg/ and src/solvers/gpu/pdhg_cuda.cu according to 03_GPU_CUDA_ACCELERATION_CORE.md."`
* **Prompt for Phase 6:**  
  `"Agent: Proceed with PHASE 6. Implement Mehrotra IPM, Condat-Vu Convex QP, and Branch & Bound in src/solvers/ according to 01_MATHEMATICAL_ENGINES_AND_ALGORITHMS.md."`
* **Prompt for Phase 7:**  
  `"Agent: Proceed with PHASE 7. Implement apps/indus-cli/main.cpp, pyomo_adapter.py, and root CMakeLists.txt. Compile the project and execute netlib_runner.py on test_models/."`
