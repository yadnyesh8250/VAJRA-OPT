# MPS & LP FILE PARSER SPECIFICATION: INDUS-OPT

This document specifies the exact parsing logic, column layout rules, section state machines, and mathematical conventions for reading **MPS** (Mathematical Programming System) and **LP** files in **INDUS-OPT**.

---

## 1. The MPS File Format Specification

The MPS format originates from IBM punch-card systems (1960s). It supports both **Fixed Format** (column-based) and **Modern Free Format** (whitespace-delimited).

### 1.1 Fixed-Format Column Fields

| Field | Columns (1-based) | Width | Purpose |
|---|---|:---:|---|
| **Field 1** | 2 – 3 | 2 chars | Indicator code (`N`, `L`, `G`, `E`, `UP`, `LO`, `BV`, etc.) |
| **Field 2** | 5 – 12 | 8 chars | Variable, Row, or Section identifier |
| **Field 3** | 15 – 22 | 8 chars | Row name (in `COLUMNS`, `RHS`, `RANGES`) |
| **Field 4** | 25 – 36 | 12 chars | Numerical value (supports `1.23E-04` and Fortran `1.23D-04`) |
| **Field 5** | 40 – 47 | 8 chars | Optional second row name |
| **Field 6** | 50 – 61 | 12 chars | Optional second numerical value |

Any line beginning with an asterisk (`*`) in Column 1 is a **comment** and must be ignored.

---

## 2. Section State Machine & Parsing Rules

The parser transitions through a strict sequence of headers:

```
[START] ──> NAME ──> [OBJSENSE] ──> ROWS ──> COLUMNS ──> RHS ──> [RANGES] ──> [BOUNDS] ──> [QUADOBJ] ──> ENDATA
```

### 2.1 `ROWS` Section
Defines the constraint names and their sense:
* `N`: Objective function (first `N` row encountered) or Free constraint.
* `E`: Equality constraint ($A x = b$).
* `L`: Less-than-or-equal constraint ($A x \le b$).
* `G`: Greater-than-or-equal constraint ($A x \ge b$).

### 2.2 `COLUMNS` Section & Integer Markers
Each column entry specifies non-zero coefficients:
```text
COLUMNS
    X01       ROW01     3.0       ROW02     1.5
    MARK01    'MARKER'            'INTORG'
    X02       ROW01     1.0       ROW03     4.0
    MARK02    'MARKER'            'INTEND'
```
* **Integer Markers:** Any variable defined between `'INTORG'` and `'INTEND'` marker records must have its `col_type[j]` set to `VarType::kInteger`.

### 2.3 `RHS` Section
Provides the right-hand side constant $b_i$ for row $i$:
* If a row is not mentioned in `RHS`, its RHS defaults to $0.0$.
* For an `E` row: $l_i = b_i, u_i = b_i$.
* For an `L` row: $l_i = -\infty, u_i = b_i$.
* For a `G` row: $l_i = b_i, u_i = +\infty$.

### 2.4 `RANGES` Section (Two-Sided Bounds)
The `RANGES` section specifies interval bounds. The sign of the range value $r$ interacts with the row sense according to the official IBM MPS standard:

| Row Type | Range Value $r$ | Lower Bound $l_i$ | Upper Bound $u_i$ |
|:---:|:---:|:---:|:---:|
| **`G`** | $r > 0$ | $b_i$ | $b_i + |r|$ |
| **`L`** | $r > 0$ | $b_i - |r|$ | $b_i$ |
| **`E`** | $r > 0$ | $b_i$ | $b_i + r$ |
| **`E`** | $r < 0$ | $b_i + r$ | $b_i$ |

### 2.5 `BOUNDS` Section
Default bounds for any variable in linear programming are $[0, +\infty)$.
The `BOUNDS` section overrides defaults using these codes:

| Bound Code | Meaning | Resulting $[l_j, u_j]$ |
|---|---|---|
| `LO` | Lower Bound | $l_j = v, \quad u_j$ unchanged |
| `UP` | Upper Bound | If $v < 0$, $l_j = -\infty$; $u_j = v$ |
| `FX` | Fixed Variable | $l_j = v, \quad u_j = v$ |
| `FR` | Free Variable | $l_j = -\infty, \quad u_j = +\infty$ |
| `MI` | Minus Infinity | $l_j = -\infty, \quad u_j$ unchanged |
| `PL` | Plus Infinity | $l_j$ unchanged, $\quad u_j = +\infty$ |
| `BV` | Binary Variable | $l_j = 0.0, \quad u_j = 1.0, \quad \text{type} = \text{kInteger}$ |
| `UI` | Upper Integer | $u_j = v, \quad \text{type} = \text{kInteger}$ |
| `LI` | Lower Integer | $l_j = v, \quad \text{type} = \text{kInteger}$ |

### 2.6 `QUADOBJ` Section (Convex QP Extension)
Specifies the symmetric Hessian matrix $Q$ in the objective $\frac{1}{2} x^T Q x + c^T x$:
```text
QUADOBJ
    X01       X01       2.000000
    X02       X01       0.500000
    X02       X02       4.000000
```
* Only store lower-triangular entries (row $\ge$ col) in `Model::Q`.
* Do not multiply by $0.5$ in the reader; the evaluator applies the $\frac{1}{2}$ factor.

---

## 3. The CPLEX `.lp` Format Parser

The parser also reads standard algebraic `.lp` files:
```text
Minimize
 obj: 42.5 X01 + 38.0 X02 + 45.2 X03 + [ 2 X01^2 + 4 X02 * X03 ] / 2
Subject To
 cdu_capacity: X01 + X02 + X03 <= 300
 diesel_yield: 0.45 X01 + 0.40 X02 + 0.52 X03 >= 130
Bounds
 0 <= X01 <= 120
 0 <= X02 <= 150
 0 <= X03 <= 80
Generals
 X01 X02
End
```
* Tokens are case-insensitive (`minimize`, `MIN`, `max`, `Subject To`, `st`, `s.t.`, `Bounds`, `Generals`, `Binary`, `End`).
* Quadratic terms are enclosed in brackets `[ ... ]` or `[ ... ] / 2`.
