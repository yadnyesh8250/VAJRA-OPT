# MATHEMATICAL ENGINES & ALGORITHM BLUEPRINT: INDUS-OPT

This document specifies the exact mathematical formulations, algorithmic state machines, recurrence relations, and pseudo-code for all continuous and mixed-integer optimization engines in **INDUS-OPT**.

---

## 1. Bounded Revised Dual Simplex Engine

The dual simplex is the default continuous LP solver and the primary node solver for Branch-and-Bound because child nodes (with narrowed variable bounds) remain **dual feasible**, allowing immediate zero-iteration warm-starts.

### 1.1 Problem Formulation in Bounded Form
$$\min_{x} c^T x \quad \text{subject to} \quad l_{\text{row}} \le A x \le u_{\text{row}}, \quad l_{\text{col}} \le x \le u_{\text{col}}$$
Standard form slack representation:
$$A x + I s = 0, \quad l_{\text{row}} \le -s \le u_{\text{row}}$$
Let $B$ be the $m \times m$ basis matrix formed by basic columns of $[A \quad I]$, and $N$ be the non-basic columns.

### 1.2 Algorithmic State Machine
1. **Factorization:** Factorize $B = L U$ using Markowitz threshold pivoting.
2. **Dual Vector Calculation (BTRAN):**
   Solve $B^T y = c_B$.
3. **Reduced Cost Computation:**
   $$d_N = c_N - N^T y$$
   Verify dual feasibility:
   - If $x_j = l_j$, then $d_j \ge -10^{-7}$.
   - If $x_j = u_j$, then $d_j \le 10^{-7}$.
   - If $l_j < x_j < u_j$ (free/boxed), then $|d_j| \le 10^{-7}$.
4. **Pricing (Selecting Leaving Variable $p$):**
   Identify row $i$ where basic variable $x_{B_i}$ violates bounds ($x_{B_i} < l_{B_i}$ or $x_{B_i} > u_{B_i}$).
   Use **Dual Devex Pricing** (approximated steepest-edge weights $\gamma_i$):
   $$\text{score}_i = \frac{(\text{violation}_i)^2}{\gamma_i}, \quad p = \arg\max_i \text{score}_i$$
   Devex weight update recurrence for row $p$ pivoting with column $q$:
   $$\gamma_i^{\text{new}} = \max\left(\gamma_i, \left(\frac{\alpha_{iq}}{\alpha_{pq}}\right)^2 \gamma_p\right)$$
5. **Pivot Row Generation (BTRAN):**
   Solve $B^T v = e_p$ where $e_p$ is the $p$-th unit vector.
   Compute tableau row entries: $\alpha_{pj} = v^T A_{\cdot j}$ for all non-basic $j$.
6. **Bound-Flipping Ratio Test (Selecting Entering Variable $q$):**
   Instead of stopping at the very first blocking non-basic variable, evaluate ratio $\theta_j = \frac{|d_j|}{|\alpha_{pj}|}$.
   Sort candidates in increasing order of $\theta_j$. As long as the change in objective slope does not cross zero, flip the non-basic variable to its opposite bound ($x_j \leftarrow u_j$ if previously $l_j$) and continue along the search direction. 
   This avoids full matrix refactorization and cuts total iterations by $40\%-60\%$.
7. **Pivot Column Generation (FTRAN):**
   Solve $B \alpha_q = A_{\cdot q}$ using hyper-sparse FTRAN.
8. **Basis Update:**
   Update $B^{-1}$ using Product Form of the Inverse (PFI) with eta vectors. Trigger fresh Markowitz LU refactorization after every 50–100 pivots or when numerical residual error exceeds $10^{-8}$.

---

## 2. Bounded Revised Primal Simplex Engine

Used when starting from a primal-feasible point or when dual simplex encounters artificial bound stalls.

### 2.1 Composite Phase-1 (No Big-M)
Avoids artificial variables with large penalty coefficients $M$ that degrade matrix conditioning:
* If $x_j < l_j$, define objective penalty $c_j^{\text{phase1}} = -1$.
* If $x_j > u_j$, define objective penalty $c_j^{\text{phase1}} = +1$.
* As soon as a variable enters its feasible interval $[l_j, u_j]$, its penalty drops to 0.

### 2.2 Harris Two-Pass Ratio Test
To prevent cycling in degenerate models (e.g. Netlib `degen3`):
* **Pass 1:** Determine maximum feasible step size $\theta_{\max}$ using relaxed tolerance $\epsilon_{\text{feas}} = 10^{-7}$:
  $$\theta_{\max} = \min_{i: \alpha_{iq} > 0} \frac{x_{B_i} - l_{B_i} + \epsilon_{\text{feas}}}{\alpha_{iq}}$$
* **Pass 2:** From all candidate rows achieving step within $\theta_{\max}$, choose the row with the largest pivot magnitude $|\alpha_{iq}|$ to maximize numerical stability:
  $$p = \arg\max_{i \in \text{Candidates}} |\alpha_{iq}|$$

---

## 3. Restarted PDHG (First-Order Matrix-Free Method)

Restarted Primal-Dual Hybrid Gradient is the **core mathematical engine for GPU acceleration**. Because it requires only matrix-vector multiplications ($Ax$ and $A^T y$) and no matrix factorizations, it maps perfectly to parallel hardware.

### 3.1 Saddle-Point Formulation
$$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}} \mathcal{L}(x, y) = c^T x + y^T (A x - b)$$
where $\mathcal{X} = \{x : l_{\text{col}} \le x \le u_{\text{col}}\}$ and $\mathcal{Y} = \{y : y_i \ge 0 \text{ for } \le, y_i \le 0 \text{ for } \ge, y_i \in \mathbb{R} \text{ for } =\}$.

### 3.2 Iteration Recurrence
Given step sizes $\tau > 0$ (primal) and $\sigma > 0$ (dual) satisfying $\tau \sigma \|A\|_2^2 < 1$:
1. **Dual Update:**
   $$y^{k+1} = \text{proj}_{\mathcal{Y}}\left(y^k + \sigma (A \bar{x}^k - b)\right)$$
2. **Primal Update:**
   $$x^{k+1} = \text{proj}_{\mathcal{X}}\left(x^k - \tau (A^T y^{k+1} + c)\right)$$
3. **Extrapolation (Halpern / Malitsky-Pock style):**
   $$\bar{x}^{k+1} = 2 x^{k+1} - x^k$$

### 3.3 Adaptive Restarting Mechanism
Compute the normalized duality gap:
$$\text{gap}(x^k, y^k) = \frac{|c^T x^k + b^T y^k|}{\|c\| \|x^k\| + \|b\| \|y^k\| + 1}$$
When the normalized duality gap or the normalized residual decreases by a factor $\beta = 0.5$, or if the residual stagnates for 200 iterations:
* Reset current iterate average: $(\bar{x}, \bar{y}) \leftarrow (x^k, y^k)$.
* Adaptively rescale $\tau$ and $\sigma$ to balance primal and dual infeasibility.

### 3.4 Hybrid Interior-Point Polish
For high-precision applications ($10^{-10}$ tolerance), take the final iterate $(x^*, y^*)$ from PDHG and warm-start 5 to 10 iterations of the Interior Point Method to eliminate residual drift.

---

## 4. Interior Point Method (Mehrotra Predictor-Corrector)

For large-scale continuous LPs where Simplex pivots are too numerous and sparse Cholesky / LDLᵀ factorizations are affordable.

### 4.1 Normal Equations
At each iteration, solve:
$$\left(A \Theta A^T\right) \Delta y = r$$
where $\Theta = X S^{-1}$ is a positive diagonal matrix of primal-dual slack ratios.
1. Factorize $A \Theta A^T = P^T L D L^T P$ using AMD ordering.
2. **Predictor Step (Affine Scaling):**
   Solve for $(\Delta x_{\text{aff}}, \Delta y_{\text{aff}}, \Delta s_{\text{aff}})$ with $\mu = 0$.
3. **Centering Parameter:**
   Compute affine duality gap $\mu_{\text{aff}}$ and set $\sigma = (\mu_{\text{aff}} / \mu)^3$.
4. **Corrector Step:**
   Add second-order correction $\Delta X_{\text{aff}} \Delta S_{\text{aff}} e - \sigma \mu e$ to the right-hand side.
5. **Step Length:**
   Compute maximum step $\alpha_{\max} \in (0, 1]$ preserving non-negativity $(x, s) > 0$, scaled by boundary buffer $\tau = 0.995$.

---

## 5. Convex Quadratic Programming (Condat-Vũ Primal-Dual)

Solves problems with quadratic objective:
$$f(x) = \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad l \le A x \le u$$

### 5.1 Positive Semi-Definiteness Certification
Before solving, compute LDLᵀ factorization of $Q$:
* If any diagonal element $D_{ii} < -10^{-12}$, **immediately abort and return a Non-Convex Certificate** containing the negative eigenvector $v$ ($v^T Q v < 0$).
* Never solve a non-convex QP to a local stationary point.

### 5.2 Algorithm Loop
Let $\beta = \|Q\|_2$. Choose step sizes such that $\tau \left(\frac{\beta}{2} + \sigma \|A\|_2^2\right) < 1$:
$$x^{k+1} = \text{proj}_{[l, u]}\left(x^k - \tau (Q x^k + c + A^T y^k)\right)$$
$$y^{k+1} = \text{proj}_{[\dots]}\left(y^k + \sigma A (2 x^{k+1} - x^k)\right)$$

---

## 6. Mixed-Integer Linear & Quadratic Programming (Branch & Bound)

### 6.1 Reliability Branching
For each fractional integer variable $x_j$, maintain average objective gain per unit change in down-direction ($\psi_j^-$) and up-direction ($\psi_j^+$).
* If variable $x_j$ has been branched on $< 8$ times (unreliable), run **Strong Branching**:
  - Warm-start dual simplex on down-child and up-child for at most 50 iterations.
  - Record exact objective gains $\Delta z^-$ and $\Delta z^+$.
  - Update pseudocosts: $\psi_j^- \leftarrow \Delta z^- / f_j$, $\psi_j^+ \leftarrow \Delta z^+ / (1 - f_j)$.
* Compute product score:
  $$\text{score}_j = \max(\psi_j^-, 10^{-6}) \times \max(\psi_j^+, 10^{-6})$$
  Branch on candidate with highest score.

### 6.2 Root Diving Heuristic
Before exploring the branch-and-bound tree:
1. Fix the integer variable closest to an integer to its nearest bound.
2. Resolve node LP using warm-started dual simplex.
3. Repeat up to depth 50. If a feasible integer solution is found, record it immediately as the **Incumbent**. This establishes an early cutoff bound that prunes $40\%-70\%$ of subsequent tree nodes.
