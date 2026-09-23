# SPARSE LINEAR ALGEBRA & PRESOLVE SPECIFICATION: INDUS-OPT

This document specifies the exact sparse matrix representations, factorization algorithms, hyper-sparse updates, matrix equilibration, and the reversible presolve engine for **INDUS-OPT**.

---

## 1. Sparse Matrix Primitives (Compressed Sparse Column & Row)

All constraint matrices are stored in dual formats to enable $O(\text{nnz})$ column and row traversals.

```cpp
namespace indus::la {

struct SparseMatrixCSC {
    int m = 0;                          // Number of rows
    int n = 0;                          // Number of columns
    std::vector<int64_t> col_ptr;       // Size n + 1: column start offsets
    std::vector<int> row_idx;           // Size nnz: row indices
    std::vector<double> values;         // Size nnz: non-zero coefficients

    [[nodiscard]] int64_t nnz() const noexcept { return values.size(); }
    [[nodiscard]] double coeff(int r, int c) const;
    void transpose_into(SparseMatrixCSR& out) const;
};

struct SparseMatrixCSR {
    int m = 0;                          // Number of rows
    int n = 0;                          // Number of columns
    std::vector<int64_t> row_ptr;       // Size m + 1: row start offsets
    std::vector<int> col_idx;           // Size nnz: column indices
    std::vector<double> values;         // Size nnz: non-zero coefficients
};

} // namespace indus::la
```

---

## 2. Sparse Markowitz LU Factorization with Threshold Stability

Given basis matrix $B$ of dimension $m \times m$:
$$P B Q = L U$$
where $P$ and $Q$ are permutation matrices, $L$ is unit lower triangular, and $U$ is upper triangular.

### 2.1 The Markowitz Minimum Fill-in Heuristic
At each elimination step $k \in [0, m-1]$:
1. Count non-zeros in each uneliminated row $i$ ($r_i$) and column $j$ ($c_j$).
2. The Markowitz product for candidate pivot $(i, j)$ is:
   $$M_{ij} = (r_i - 1)(c_j - 1)$$
3. **Numerical Threshold Criterion:**
   To guarantee stability against catastrophic cancellation, entry $a_{ij}$ must satisfy:
   $$|a_{ij}| \ge u \cdot \max_{k} |a_{kj}|$$
   where threshold $u = 0.01$ (tightened to $0.1$ if conditioning degrades).
4. Select pivot $(p, q)$ that minimizes $M_{pq}$ among all entries meeting the threshold criterion.
5. Eliminate row $p$ and update remaining matrix using sparse outer-product rank-1 updates.

### 2.2 Singularity Handling & Rank Defect Repair
If no entry in column $q$ satisfies the threshold criterion, $B$ is structurally or numerically singular:
* Mark column $q$ as rank-deficient.
* Inject an artificial slack vector $e_q$ (identity unit vector) with an artificial bound $[0, 0]$ into the basis.
* Mark basis as repaired; this allows the Simplex loop to continue without crashing, eventually pivoting the artificial variable out.

---

## 3. Basis Updates & Hyper-Sparse FTRAN / BTRAN

Between refactorizations, as entering column $q$ replaces leaving column $p$, the basis changes by a rank-1 column swap:
$$B_{\text{new}} = B + (a_q - B e_p) e_p^T = B (I + (\alpha_q - e_p) e_p^T) = B E$$
where $\alpha_q = B^{-1} a_q$ is the FTRAN solution and $E$ is an elementary **Eta Matrix**.

### 3.1 Product Form of the Inverse (PFI)
$$B_k^{-1} = E_k E_{k-1} \dots E_1 (L U)^{-1}$$
Store eta vectors $\eta_k$ compactly: pivot position $p_k$ and sparse column vector $v_k$.

### 3.2 Hyper-Sparse FTRAN and BTRAN
When solving $B x = b$ (FTRAN) or $B^T y = c$ (BTRAN) where $b$ or $c$ has only a few non-zeros (e.g. unit vector $e_p$ during pricing):
1. **Topological Graph Search:**
   Run a Depth-First Search (DFS) on the directed acyclic graph (DAG) defined by the non-zero patterns of $L$ and $U$, starting strictly from the non-zero indices of the RHS vector.
2. **Elimination of Zero-Sweeps:**
   Only traverse the reachable subset of rows and columns (often $< 5\%$ of $m$).
3. **Execution Time:**
   Reduces solve time from $O(m)$ to $O(\text{nnz}(x))$, speeding up large-scale dual simplex pivots by up to $10\times$.

---

## 4. Matrix Equilibration & Scaling

Ill-conditioned matrices (e.g. Netlib models with $10^{20}$ coefficient ranges) fail without scaling.

### 4.1 Ruiz Two-Sided $\ell_\infty$ Scaling
Compute diagonal row scaling $R = \text{diag}(r_1, \dots, r_m)$ and column scaling $C = \text{diag}(c_1, \dots, c_n)$:
1. Initialize $R^{(0)} = I$, $C^{(0)} = I$, $A^{(0)} = A$.
2. For iteration $k = 1, \dots, 10$:
   $$r_i^{(k)} = \frac{1}{\sqrt{\|A_{i, :}^{(k-1)}\|_\infty}}, \quad c_j^{(k)} = \frac{1}{\sqrt{\|A_{:, j}^{(k-1)}\|_\infty}}$$
   $$A^{(k)} = R^{(k)} A^{(k-1)} C^{(k)}$$
3. Stop when $\max_i |1 - \|A_{i, :}^{(k)}\|_\infty| \le 0.05$ and $\max_j |1 - \|A_{:, j}^{(k)}\|_\infty| \le 0.05$.

### 4.2 Pock-Chambolle Diagonal Preconditioning (for PDHG)
For first-order methods, compute:
$$D_R = \text{diag}\left(\frac{1}{\sum_{j} |A_{ij}|^{2 - \alpha}}\right), \quad D_C = \text{diag}\left(\frac{1}{\sum_{i} |A_{ij}|^\alpha}\right)$$
with parameter $\alpha = 1.0$. This balances step sizes across non-uniform sparse matrices.

---

## 5. Presolve Engine (8 Reductions) & Reversible Postsolve Stack

Presolve reduces problem dimensions before optimization. Every reduction pushes a **Postsolve Operation** onto a LIFO stack.

### 5.1 The 8 Reversible Reductions

| Reduction | Condition | Action | Postsolve Dual Recovery |
|---|---|---|---|
| **1. Empty Row** | $\sum_j |A_{ij}| = 0$ | If $l_i \le 0 \le u_i$, drop row; else declare **Infeasible**. | $y_i = 0$. |
| **2. Empty Column** | $\sum_i |A_{ij}| = 0$ | Fix $x_j = l_j$ (if $c_j > 0$) or $u_j$ (if $c_j < 0$). If $c_j \ne 0$ and unconstrained, declare **Unbounded**. | $d_j = c_j$. |
| **3. Fixed Column** | $l_j = u_j$ | Substitute $x_j = l_j$. Adjust row bounds: $l_i \leftarrow l_i - A_{ij} l_j$, $u_i \leftarrow u_i - A_{ij} l_j$. Remove column $j$. | $x_j = l_j$, $d_j = c_j - A_{\cdot j}^T y$. |
| **4. Singleton Row** | $A_{ij} x_j$ is the only term in row $i$ | Implies explicit bounds on $x_j$: $[l_i / A_{ij}, u_i / A_{ij}]$. Tighten $l_j \leftarrow \max(l_j, \text{new\_l})$, $u_j \leftarrow \min(u_j, \text{new\_u})$. Drop row $i$. | Reconstruct dual multiplier $y_i = d_j^{\text{reduced}} / A_{ij}$. |
| **5. Forcing Row** | Sum of lower bounds matches $u_i$, or upper bounds match $l_i$ | All variables in the row are locked to their respective bounds. Fix variables, remove row. | Calculate row dual from active constraint. |
| **6. Redundant Row** | Row bounds are looser than implied extremal bounds $\sum \max(A_{ij} l_j, A_{ij} u_j)$ | Constraint can never be violated. Drop row $i$. | $y_i = 0$. |
| **7. Free Column Singleton** | $x_j$ appears in only row $i$, with $l_j = -\infty, u_j = +\infty$ | Use row $i$ to express $x_j$ in terms of other variables. Substitute $x_j$ out of objective, drop column $j$ and row $i$. | $y_i = c_j / A_{ij}$; recover $x_j$ from row equation. |
| **8. Doubleton Equation** | Row $i$ is $a_1 x_1 + a_2 x_2 = b$ | Substitute $x_1 = (b - a_2 x_2)/a_1$. Update bounds on $x_2$, drop variable $x_1$ and row $i$. | Recover $x_1$, compute duals. |

### 5.2 Exact Dual Fixed-Point Postsolve Reconstruction
Many open-source solvers fail because their postsolve produces invalid dual vectors $y$. **INDUS-OPT** uses a strict fixed-point iteration:
1. Reconstruct all primal values $x$ by unwinding the stack in reverse order.
2. Reconstruct dual multipliers $y$ according to each reduction formula.
3. **Fixed-Point Iteration:** For any column touched by presolve:
   $$d_j = c_j - A_{\cdot j}^T y$$
   Verify that $d_j$ satisfies sign conditions:
   $$\text{violation} = \max(0, -d_j) \quad \text{for } x_j = l_j$$
   If violation $> 10^{-7}$, adjust row duals along the reduction graph to fixed point.
4. Pass the final reconstructed point $(x, y, d)$ to `recompute_quality()` to audit residuals against the **unmodified original model**.
