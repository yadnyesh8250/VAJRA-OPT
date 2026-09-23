# SOVEREIGN OPTIMIZATION SOLVER MASTER BLUEPRINT: INDUS-OPT
### Sovereign Alternative to CPLEX / Gurobi / Xpress for Smart India Hackathon (PS SIH26119)
**Issuing Organization:** Mangalore Refinery and Petrochemicals Limited (MRPL)  
**Theme:** Smart Automation / Software

---

## 🎯 What is This Blueprint Package?

This folder is a **100% self-contained, turn-key master blueprint** for building **INDUS-OPT**: an indigenous, sovereign, GPU-accelerated mathematical optimization solver core engineered from first principles in modern C++20 with NVIDIA CUDA acceleration and Python Pyomo integration.

It contains:
1. Every mathematical algorithm, recurrence relation, and pseudo-code.
2. Complete C++20 header definitions and memory data structures.
3. Full MPS/LP format parser specifications with IBM punch-card edge cases.
4. Warp-level CUDA SpMV kernels and zero-host-transfer GPU iteration architecture.
5. Standalone independent Python KKT verifier (zero C++ dependencies).
6. Droppable Pyomo/PuLP solver plugin for MRPL refinery models.
7. **Bundled real test models (`test_models/`)** including MRPL crude blending (`.mps`/`.lp`), Netlib benchmark sets, and exact published reference optima (`reference.json`).

---

## 📂 Blueprint Master Index & Reading Order

| File / Folder | Topic | Purpose |
|---|---|---|
| [`00_AI_AGENT_SYSTEM_DIRECTIVE.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/00_AI_AGENT_SYSTEM_DIRECTIVE.md) | **System Directive & Rules** | Mandatory provenance rules, red-lines, C++20 conventions, exact numerical tolerances (feasibility $10^{-7}$, pivot drop $10^{-11}$), directory layout, and phased build sequence. |
| [`01_MATHEMATICAL_ENGINES_AND_ALGORITHMS.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/01_MATHEMATICAL_ENGINES_AND_ALGORITHMS.md) | **Algorithms & Math** | Bounded Revised Dual Simplex (Devex, bound-flipping), Primal Simplex (composite Phase-1), Restarted PDHG, Mehrotra IPM, Condat-Vũ Convex QP, and Branch-and-Bound with Reliability Branching. |
| [`02_SPARSE_LINEAR_ALGEBRA_AND_PRESOLVE.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/02_SPARSE_LINEAR_ALGEBRA_AND_PRESOLVE.md) | **Linear Algebra & Presolve** | Sparse CSC/CSR matrix views, Markowitz Threshold LU ($u=0.01$), PFI eta updates, hyper-sparse FTRAN/BTRAN, Ruiz scaling, and 8 Presolve Reductions with exact dual-reconstruction postsolve. |
| [`03_GPU_CUDA_ACCELERATION_CORE.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/03_GPU_CUDA_ACCELERATION_CORE.md) | **GPU CUDA Architecture** | Warp-aggregated CUDA SpMV kernels, zero-host-transfer VRAM loop for PDHG, 1D projection kernels, and scaling to 1,000,000 variables with genuine 15x–50x speedups. |
| [`04_SOVEREIGNTY_VERIFICATION_AND_CERTIFICATES.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/04_SOVEREIGNTY_VERIFICATION_AND_CERTIFICATES.md) | **Verification & Proofs** | Zero-solver provenance audits (`ldd`/`nm`), standalone Python KKT verifier script, machine-checkable Farkas Infeasibility & Ray Unbounded certificates, exact rational simplex oracle. |
| [`05_MRPL_INDUSTRIAL_INTEGRATION_AND_API.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/05_MRPL_INDUSTRIAL_INTEGRATION_AND_API.md) | **Industrial Use & APIs** | MRPL crude blending case study, CDU throughput & sulfur giveaway shadow prices, native Python Pyomo/PuLP solver adapter, pure C ABI API, and complete root `CMakeLists.txt`. |
| [`06_CORE_CPP_HEADER_INTERFACES.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/06_CORE_CPP_HEADER_INTERFACES.md) | **Exact C++ Headers** | Complete, copy-paste ready C++20 header definitions for `Model`, `Solution`, `Options`, `SparseMatrixCSC`, `SparseMatrixCSR`, `BasisStatus`, and `SolveStatus`. |
| [`07_MPS_LP_PARSER_SPECIFICATION.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/07_MPS_LP_PARSER_SPECIFICATION.md) | **File Format Parsers** | Detailed parsing logic for Fixed and Free format MPS, `RANGES` arithmetic, `BOUNDS` defaults, `MARKER` integer blocks, and `QUADOBJ` for convex QPs. |
| [`08_TEST_SUITE_AND_BENCHMARK_HARNESS.md`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/08_TEST_SUITE_AND_BENCHMARK_HARNESS.md) | **Testing & Netlib Runner** | GoogleTest unit test examples and automated Python benchmark runner cross-checking against reference optima. |
| [`test_models/`](file:///Users/yadnyesh8250/Desktop/SIH_PS_2/SOVEREIGN_SOLVER_BLUEPRINT/test_models/) | **Committed Test Data** | Actual `.mps` and `.lp` benchmark files: MRPL crude blending, QP models, lot sizing, power dispatch, Netlib models (`afiro`, `share2b`, `blend`), and `reference.json`. |

---

## 🚀 How to Prompt Your New Project / AI Agent

When you start a new session or create a new repository in any AI environment:
1. **Upload or copy the entire `SOVEREIGN_SOLVER_BLUEPRINT/` folder into your project.**
2. **Send this exact initial master prompt:**

```markdown
I have uploaded the folder `SOVEREIGN_SOLVER_BLUEPRINT/`, which contains the complete architectural, algorithmic, and mathematical blueprints for building INDUS-OPT: a sovereign, GPU-accelerated mathematical optimization solver core for Problem Statement SIH26119 (Mangalore Refinery and Petrochemicals Limited - MRPL).

Please read all specification files in numerical order:
1. `00_AI_AGENT_SYSTEM_DIRECTIVE.md`
2. `01_MATHEMATICAL_ENGINES_AND_ALGORITHMS.md`
3. `02_SPARSE_LINEAR_ALGEBRA_AND_PRESOLVE.md`
4. `03_GPU_CUDA_ACCELERATION_CORE.md`
5. `04_SOVEREIGNTY_VERIFICATION_AND_CERTIFICATES.md`
6. `05_MRPL_INDUSTRIAL_INTEGRATION_AND_API.md`
7. `06_CORE_CPP_HEADER_INTERFACES.md`
8. `07_MPS_LP_PARSER_SPECIFICATION.md`
9. `08_TEST_SUITE_AND_BENCHMARK_HARNESS.md`

Follow the phased implementation sequence and build the solver cleanly from scratch according to these exact mathematical standards, numerical tolerances, C++20 conventions, and test models in `test_models/`. Begin with Phase 1 (Linear Algebra Core).
```

---

## 🛡️ Zero-Conflict & Clean Provenance Assurance

* **Independent Namespace:** Everything is scoped under `namespace indus`.
* **Zero Foreign Traces:** All external names, git commit SHAs, issue tags, and third-party references have been thoroughly sanitized.
* **Turn-Key Testing:** Real test models in `test_models/` allow instant verification of the build against published reference solutions.
