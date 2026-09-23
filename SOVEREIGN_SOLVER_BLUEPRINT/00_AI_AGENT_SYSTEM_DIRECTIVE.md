# SOVEREIGN OPTIMIZATION SOLVER (INDUS-OPT): MASTER AI SYSTEM DIRECTIVE
**Problem Statement SIH26119:** Indigenous GPU-Accelerated Optimization Solver (Sovereign Alternative to CPLEX / Xpress)  
**Issuing Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Target Identity:** `INDUS-OPT` (C++ namespace: `indus`, library: `libindus`, CLI: `indus`)

---

## 1. Primary Mission & The Sovereign Red-Line Rule

You are tasked with engineering a production-grade, numerically robust mathematical optimization solver from mathematical first principles in **modern C++20**, accompanied by **CUDA GPU acceleration** and a **Python Pyomo/PuLP interface**.

### The Non-Negotiable Red-Line Constraint
```
[CRITICAL RULE: STRICT PROVENANCE & ZERO SOLVER INCLUSION]
No source code, header, algorithm implementation, or binary from ANY existing mathematical 
solver may be linked, vendored, copied, wrapped, or derived.
Banned solvers include:
- Commercial: IBM ILOG CPLEX, Gurobi, FICO Xpress, Mosek.
- Open Source: COIN-OR (Clp, Cbc, Ipopt), HiGHS, SCIP, SoPlex, GLPK, lp_solve, OSQP,
  PDLP, cuPDLP, Google OR-Tools (GLOP, CP-SAT).
Not their simplex, not their cutting planes, not even their MPS file reader.
```

### Explicitly Permitted Foundations
1. **Textbooks & Academic Literature:** Chvátal, Bertsimas & Tsitsiklis, Nocedal & Wright, Bixby, Maros, Koberstein, Achterberg.
2. **Standard Non-Solver Utility Libraries:**
   * Modern formatting: `fmt` (10.2.1)
   * Command line parsing: `CLI11` (v2.4.2)
   * JSON serialization: `nlohmann/json` (v3.11.3)
   * Compression: `zlib` (for `.mps.gz`)
   * Unit Testing: `GoogleTest`
   * Python Bindings: `pybind11` or `ctypes` over the C API
   * Vendor GPU BLAS/Sparse: NVIDIA `cuSPARSE` and standard CUDA runtime (these are hardware BLAS libraries, not solvers).
3. **Reference Benchmarks:** Netlib LP, MIPLIB 2017, Mittelmann, QPLIB.

---

## 2. Target Directory & Architectural Layout

The codebase must follow a clean, modern, enterprise-grade solver structure:

```
indus_opt/
├── CMakeLists.txt                      # Root build file (C++20, warnings as errors)
├── include/indus/                      # Public C++ API headers
│   ├── indus.h                         # Pure C FFI API (ABI-stable)
│   ├── model.hpp                       # Core Model, Solution, BasisStatus definitions
│   ├── solver.hpp                      # Abstract ISolver interface and factory
│   ├── options.hpp                     # Strongly-typed runtime option registry
│   ├── tolerances.hpp                  # Unified numerical tolerances (zero magic numbers)
│   ├── sparse.hpp                      # Sparse CSC, CSR, and Coordinate views
│   ├── certificate.hpp                 # Farkas & Ray proof certificates
│   ├── mip.hpp                         # Branch & bound settings and statistics
│   ├── qp.hpp                          # Quadratic objective parameters
│   ├── pdhg.hpp                        # First-order restarted PDHG parameters
│   ├── gpu.hpp                         # GPU hardware context & kernel declarations
│   ├── logging.hpp                     # Thread-safe hierarchical logger
│   ├── timer.hpp                       # High-resolution wall-clock & CPU timer
│   └── version.hpp                     # Git SHA, build type, and CUDA status
├── src/
│   ├── engine/                         # Model representation, Solution quality, dispatcher
│   │   ├── model.cpp                   # Model builder and validation
│   │   ├── solve.cpp                   # Master class-based solver dispatcher
│   │   ├── certificate.cpp             # Farkas/Ray certificate generators
│   │   ├── status_guard.hpp            # Post-solve KKT residual audit & status clamp
│   │   ├── options.cpp                 # Option registry implementation
│   │   └── logging.cpp                 # Formatted console/file logger
│   ├── linalg/                         # Core sparse numerical linear algebra
│   │   ├── sparse.cpp                  # CSC/CSR matrix operations & transpose
│   │   ├── lu.hpp / lu.cpp             # Sparse Markowitz LU with threshold pivoting & PFI
│   │   ├── ldl.hpp / ldl.cpp           # Sparse LDLᵀ with AMD quotient graph ordering
│   │   └── scaling.hpp / scaling.cpp   # Ruiz & Pock-Chambolle matrix equilibration
│   ├── io/                             # High-performance format parsers & writers
│   │   ├── mps_reader.cpp              # Fixed/Free format MPS with RANGES & QUADOBJ
│   │   ├── lp_reader.cpp               # CPLEX-style .lp file reader
│   │   ├── writer.cpp                  # .sol solution and .json stats exporter
│   │   └── line_reader.cpp             # Fast buffered line reader with gz support
│   ├── presolve/                       # Dual-reconstruction presolve & postsolve
│   │   ├── presolve.hpp                # Reduction stack interface
│   │   └── presolve.cpp                # 8 Reductions & exact dual fixed-point postsolve
│   ├── solvers/                        # Optimization solver implementations
│   │   ├── simplex/                    # Revised Simplex family
│   │   │   ├── simplex_core.hpp        # Shared basis, factorizations, pricing weights
│   │   │   ├── dual_simplex.cpp        # Bounded Dual Simplex (Devex, bound-flipping)
│   │   │   ├── primal_simplex.cpp      # Bounded Primal Simplex (composite Phase-1)
│   │   │   └── dense_lu.cpp            # Test reference oracle
│   │   ├── pdhg/                       # Restarted PDHG (First-Order CPU Engine)
│   │   │   └── pdhg.cpp                # Adaptive step-size PDLP method with IPM polish
│   │   ├── interior_point/             # Mehrotra Predictor-Corrector IPM
│   │   │   └── ipm.cpp                 # Normal equations A D Aᵀ via sparse LDLᵀ
│   │   ├── quadratic/                  # Convex Quadratic Programming
│   │   │   ├── convexity.cpp           # Positive semi-definiteness certifier
│   │   │   └── qp_condat_vu.cpp        # Condat-Vũ primal-dual algorithm
│   │   ├── branch_bound/               # Mixed-Integer Programming (MILP / MIQP)
│   │   │   ├── branch_and_bound.cpp    # Tree search, reliability branching, diving
│   │   │   └── cuts.cpp                # Gomory Mixed-Integer & lifted knapsack cuts
│   │   └── gpu/                        # CUDA Acceleration Layer
│   │       ├── cuda_context.hpp        # Device detection, streams, VRAM management
│   │       ├── spmv_kernels.cuh        # Fast CUDA SpMV and Transposed SpMV
│   │       └── pdhg_cuda.cu            # VRAM-resident GPU PDHG loop
│   └── interfaces/
│       ├── capi/c_api.cpp              # C ABI wrapper functions
│       └── python/indus_opt/           # Python ctypes & Pyomo connector
│           ├── __init__.py
│           ├── solver.py               # Pythonic wrapper class
│           └── pyomo_adapter.py        # Drop-in Pyomo/PuLP Solver plugin
├── apps/indus-cli/
│   └── main.cpp                        # CLI tool: `indus solve|info|options|bench`
├── validator/
│   └── independent_verifier.py         # Standalone Python KKT verifier (0 C++ code)
└── demo/
    ├── crude_blend.mps                 # MRPL CDU & diesel sulfur blending model
    ├── crude_blend_qp.mps              # MRPL nonlinear price impact model
    └── run_mrpl_demo.sh                # End-to-end benchmark & demo script
```

---

## 3. Strict Numerical Standards (include/indus/tolerances.hpp)

No arbitrary magic numbers are permitted in the numerical code. All tolerances must adhere to these defaults:

| Parameter | Value | Mathematical Purpose |
|---|---|---|
| `kPrimalFeasibility` | `1e-7` | Maximum allowed row/column bound violation: $\|Ax - b\|_\infty \le 10^{-7}$ |
| `kDualFeasibility` | `1e-7` | Maximum violation of dual sign conditions / reduced costs |
| `kIntegrality` | `1e-6` | Distance from integer: $|x_j - \text{round}(x_j)| \le 10^{-6}$ |
| `kMipRelativeGap` | `1e-4` | Relative optimality gap: $\frac{\|z_{\text{incumbent}} - z_{\text{bound}}\|}{\|z_{\text{incumbent}}\| + 10^{-10}} \le 10^{-4}$ |
| `kMipAbsoluteGap` | `1e-6` | Absolute optimality gap: $\|z_{\text{incumbent}} - z_{\text{bound}}\| \le 10^{-6}$ |
| `kMarkowitzThreshold` | `0.01` | Markowitz pivoting threshold $u \in [0.01, 0.1]$ to control element growth in LU |
| `kZeroDrop` | `1e-11` | Elements smaller than $10^{-11}$ are dropped from sparse factorization |
| `kPseudocostReliability` | `8` | Minimum observations before trusting variable branching pseudocosts |
| `kStrongBranchingCandidates` | `10` | Maximum candidate variables evaluated by strong branching per node |
| `kStrongBranchingIterations` | `50` | Maximum dual simplex iterations allowed per strong branching child probe |

---

## 4. Phased Implementation Sequence for the AI Agent

When implementing the solver, follow this exact dependency-ordered sequence:

* **Phase 1: Linear Algebra Core:** Sparse CSC/CSR, Markowitz LU with threshold stability, Product Form of Inverse (PFI) with eta file, sparse LDLᵀ with AMD.
* **Phase 2: Bounded Simplex Engine:** Dual simplex (primary node solver) with Devex pricing and bound-flipping ratio test; Primal simplex with composite Phase-1.
* **Phase 3: Format Readers & C API:** MPS reader (supporting fixed and free formats, OBJSENSE, RANGES, BOUNDS, and QUADOBJ), `.sol` solution writer, and C API.
* **Phase 4: Presolve & Postsolve Stack:** 8 row/column reductions with reversible dual reconstruction to fixed point.
* **Phase 5: Restarted PDHG & CUDA GPU Acceleration:** First-order matrix-free PDHG on CPU, followed by `cuSPARSE` / CUDA SpMV kernels to run 50,000 iterations entirely in VRAM.
* **Phase 6: Interior Point & Convex QP:** Mehrotra predictor-corrector on normal equations $ADA^T$; Condat-Vũ primal-dual for convex QP with LDLᵀ convexity verification.
* **Phase 7: Branch & Bound (MILP / MIQP):** Domain change stack, reliability branching, root diving heuristic, warm-started dual simplex node solves, Gomory root cuts.
* **Phase 8: Python Pyomo Integration & MRPL Demo:** Pyomo solver adapter, independent Python KKT verifier (`validator/independent_verifier.py`), and live crude blending demo.
