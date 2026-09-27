# Industrial Demonstration Models for MRPL Problem Statement 26119

This document details the formulation, operational context, categorization, and benchmark results for the refinery and industrial optimization models included in VAJRA-OPT (Siddhanta).

---

## 1. Model Catalog & Classification

| Model File | Formulation | Category | Problem Class | Rows | Cols | NNZ | Published / Verified Objective |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `crude_blend.mps` | CDU crude blending pool | **Industrial-style demonstration** | LP | 3 | 3 | 9 | `214.14594595` |
| `crude_blend.lp` | Human-readable blending LP | **Industrial-style demonstration** | LP | 3 | 3 | 9 | `214.14594595` |
| `blend_milp.mps` | Crude blend with mode switch | **Industrial-style demonstration** | MILP | 3 | 4 | 11 | `223.85760518` |
| `lot_sizing.mps` | Production planning lot sizing | **Industrial-style demonstration** | MILP | 8 | 11 | 18 | `770.00000000` |
| `supply_chain.mps` | Depot distribution network | **Industrial-style demonstration** | LP | 7 | 12 | 24 | `4500.00000000` |
| `power_dispatch.mps`| Cogeneration unit commitment | **Industrial-style demonstration** | MILP | 10 | 8 | 24 | `3270.00000000` |
| `crude_blend_qp.mps`| Quadratic quality giveaway QP | **Industrial-style demonstration** | QP | 3 | 3 | 9 | `179.91776118` |
| `mrpl_structured_refinery`| Multi-train 16-period CDU/FCC | **Synthetic industrial stress model** | LP | 1,024 | 1,088 | 3,328 | Solved via PDHG |
| `ill_conditioned.mps`| Ill-conditioned scaled network | **Synthetic stress model** | LP | 7 | 12 | 24 | `4500.00000000` |

---

## 2. Technical Formulation of Each Model

### 1. `crude_blend.mps` (MRPL Crude Blending LP)
- **Category**: Industrial-style demonstration
- **Context**: 3-crude charge to Atmospheric Crude Distillation Unit (CDU):
  - Arab Light (AL): 74 $/bbl, diesel yield 0.30, sulfur 1.80 %wt
  - Bonny Light (BN): 79 $/bbl, diesel yield 0.45, sulfur 0.14 %wt
  - Murban (MU): 77 $/bbl, diesel yield 0.38, sulfur 0.78 %wt
- **Constraints**:
  - `THRUPUT`: Total CDU throughput 90 to 120 kbbl/day (ranged constraint)
  - `DIESEL`: Target diesel production exactly 40 kbbl/day (equality constraint)
  - `SULPHUR`: Pool sulfur $\le 1.00$ %wt
- **Optimal Solution**: AL = 70.27, BN = 42.07, MU = 0.0, Margin = $214.15 k$/day.

### 2. `blend_milp.mps` (Crude Blending with Discrete CDU Mode Switching)
- **Category**: Industrial-style demonstration
- **Context**: Extends `crude_blend` with binary decision variable `MODE`:
  - `MODE = 0`: Standard CDU operating mode.
  - `MODE = 1`: Max-diesel mode (incurs 12 k$/day operating cost, relaxes sulfur limit by 20 units via big-M linking).
- **Optimal Solution**: `MODE = 1` chosen, Optimal Objective = `223.85760518`.

### 3. `lot_sizing.mps` (Multi-Period Production Planning with Setup Binaries)
- **Category**: Industrial-style demonstration
- **Context**: Textbook weak LP relaxation test bed. Fixed setup charge paid whenever unit produces:
  $$x_t \le M y_t, \quad y_t \in \{0, 1\}$$
- **Horizon**: 4 planning periods, demand $[40, 60, 30, 50]$, setup cost $150$ per period.
- **Optimal Solution**: Objective = `770.0`, explored 21 branch-and-bound nodes.

### 4. `supply_chain.mps` (Refinery to Depot Transportation Network)
- **Category**: Industrial-style demonstration
- **Context**: Classical balanced transportation model connecting 3 refineries to 4 distribution depots.
- **Mathematical Property**: Redundant constraint matrix ($\text{rank} = 6$ with 7 rows). Highly degenerate vertices designed to induce ratio-test ties and test basis cycling avoidance.
- **Optimal Solution**: Solved in 7 iterations without cycling, Objective = `4500.0`.

### 5. `power_dispatch.mps` (Captive Cogeneration Power Plant Unit Commitment)
- **Category**: Industrial-style demonstration
- **Context**: Single-period unit commitment with binary on/off commitment variables ($u_g$) and continuous generation levels ($p_g$) bounded by minimum stable generation ($P_{\min}$) and maximum capacity ($P_{\max}$).
- **Optimal Solution**: Objective = `3270.0`, explored 7 nodes.

### 6. `crude_blend_qp.mps` (Convex Quadratic Blending Quality Giveaway)
- **Category**: Industrial-style demonstration
- **Context**: Blending with non-linear penalties for octane or cetane giveaway.
- **Objective**: $\max 0.5 x^T Q x + c^T x$ where $Q$ is negative semidefinite (concave maximization).
- **Optimal Solution**: Objective = `179.91776118`, stationarity residual $< 2 \times 10^{-12}$.

### 7. `ill_conditioned.mps` (Extreme Scaling Ill-Conditioned Constraint Matrix)
- **Category**: Synthetic stress model
- **Context**: Re-parameterization of `supply_chain.mps` where row and column scales span 22 orders of magnitude ($10^{-10}$ to $10^{+12}$). Matrix 2-norm condition number $\approx 2 \times 10^{28}$.
- **Result**: Ruiz equilibration and LU partial pivoting successfully solve to exact objective `4500.0`.

---

## 3. Execution & Automated Demonstration

Run the automated demonstration suite:
```bash
./scripts/run_industrial_demos.sh
```
All outputs are saved to `build/industrial_demos/` as `.sol` and `.json` artifacts and verified by the sovereign audit engine.
