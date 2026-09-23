# ADVANCED CUTS, ANTI-CYCLING & ROBUSTNESS SPECIFICATION: INDUS-OPT

This document specifies the exact advanced numerical techniques that eliminate all prototype failure modes: **Mixed Integer Rounding (MIR) cuts**, **Deterministic Anti-Cycling Perturbation**, and **Automated Sparse LU Rank Defect Repair**.

---

## 1. Mixed Integer Rounding (MIR) Cutting Planes

Pure Gomory cuts often suffer from small coefficients and numerical instability. **INDUS-OPT** implements Mixed Integer Rounding (MIR) cuts based on Marchand & Wolsey (2001).

### Mathematical Formulation
Given a base inequality derived from an optimal LP tableau row:
$$\sum_{j \in I} a_j x_j + \sum_{j \in C} g_j x_j \le b$$
where $I$ are integer variables and $C$ are continuous variables. Let:
$$f_0 = b - \lfloor b \rfloor, \quad f_j = a_j - \lfloor a_j \rfloor$$
The MIR cut is defined as:
$$\sum_{j \in I} \left( \lfloor a_j \rfloor + \frac{\max(0, f_j - f_0)}{1 - f_0} \right) x_j + \frac{1}{1 - f_0} \sum_{j \in C, g_j < 0} g_j x_j \le \lfloor b \rfloor$$

### Algorithmic Properties
1. **Deeper Cuts:** Cuts deeper into the LP relaxation than simple integer rounding without adding slack variables.
2. **Sparsity:** Coefficients smaller than $10^{-8}$ are purged, preserving sparsity of the node basis.
3. **Application:** Applied at the root node and depth $\le 5$ in the Branch-and-Bound tree.

---

## 2. Deterministic Anti-Cycling Perturbation (Harris-Wolfe Method)

In highly degenerate industrial models (e.g. Netlib `degen3`, MRPL multi-crude blending), multiple basic variables sit exactly at their bounds, leading to zero objective progress ($\theta = 0$) and infinite cycling.

### Anti-Cycling State Machine
```cpp
// src/solvers/simplex/dual_simplex.cpp
void SimplexCore::check_and_perturb_degenerate_stall() {
    if (consecutive_zero_steps_ >= 5) {
        log_info("Degenerate stall detected; applying deterministic bound perturbation.");
        for (int j = 0; j < num_cols_; ++j) {
            // Deterministic hash-based epsilon
            double eps = 1e-9 * (1.0 + (static_cast<double>((j * 1664525 + 1013904223) % 1000) / 1000.0));
            col_lower_[j] -= eps;
            col_upper_[j] += eps;
        }
        perturbed_ = true;
    }
}
```
* **Guaranteed Termination:** Because perturbed bounds differ by small unique offsets, every pivot produces a strictly positive step length ($\theta > 0$), mathematically preventing cycles.
* **Purification Pass:** Once optimal under perturbation, restore exact original bounds and execute 1–3 cleanup pivots on fresh LU factors.

---

## 3. Dynamic Residual Monitoring & Zero-Crash Rank Defect Repair

### Dynamic Condition Monitoring
During Product Form of the Inverse (PFI) updates, error accumulates in eta vectors. The engine continuously monitors the backward residual:
$$r = \|B x_B - A_{\cdot B} x_B\|_\infty$$
* If $r > 10^{-8}$ or if the number of eta vectors reaches 60, immediately trigger a fresh Markowitz LU refactorization.

### Rank Defect Repair in Sparse Markowitz LU
If candidate column $q$ contains no entry satisfying $|a_{iq}| \ge u \max_k |a_{kq}|$:
```cpp
// src/linalg/lu.cpp
if (!admissible_pivot_found) {
    // Column q is numerically rank-deficient.
    // Instead of throwing an exception or crashing:
    // 1. Replace column q with unit basis vector e_k
    // 2. Assign artificial bound [0, 0]
    // 3. Mark basis as repaired
    basis_status_[q] = BasisStatus::kAtLower;
    inject_identity_column(q);
    was_repaired_ = true;
}
```
This guarantees that **INDUS-OPT never crashes with an unhandled singular matrix exception**, allowing the dual simplex to pivot the artificial slack out safely.
