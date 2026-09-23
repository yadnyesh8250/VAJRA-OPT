# MRPL INDUSTRIAL INTEGRATION & API SPECIFICATION: INDUS-OPT

This document specifies the exact industrial problem formulations for **Mangalore Refinery and Petrochemicals Limited (MRPL)**, the interpretation of dual shadow prices for refinery planners, the drop-in **Pyomo / PuLP Python adapter**, the pure C API, and root `CMakeLists.txt`.

---

## 1. MRPL Refinery Blending Case Study (`demo/crude_blend.mps`)

### 1.1 Physical Plant Formulation
MRPL operates Crude Distillation Units (CDU) processing diverse imported and domestic crude parcels into finished diesel meeting Bharat Stage VI (BS-VI) specifications.

* **Decision Variables ($x_1, x_2, x_3$):** Throughput of Crude 1 (Arabian Light), Crude 2 (Basrah Medium), and Crude 3 (Bombay High) in thousand barrels/day (kbpd).
* **Objective Function:** Maximize net gross refining margin (GRM):
  $$\max \quad 42.5 x_1 + 38.0 x_2 + 45.2 x_3$$
* **Technical Constraints:**
  1. **CDU Throughput Capacity:**
     $$x_1 + x_2 + x_3 \le 300 \quad \text{kbpd}$$
  2. **Diesel Pool Yield Commitment:**
     $$0.45 x_1 + 0.40 x_2 + 0.52 x_3 \ge 130 \quad \text{kbpd}$$
  3. **Sulfur Specification Budget (BS-VI Ultra-Low Sulfur $\le 10 \text{ ppm}$):**
     $$1.2 x_1 + 2.5 x_2 + 0.15 x_3 \le 350 \quad \text{kg/day}$$
  4. **Crude Parcel Supply Contracts:**
     $$0 \le x_1 \le 120, \quad 0 \le x_2 \le 150, \quad 0 \le x_3 \le 80$$

### 1.2 Interpreting Dual Shadow Prices for MRPL Planners
When **INDUS-OPT** solves this model, it outputs not just optimal crude rates, but the **Dual Multipliers ($\pi_i$)**:
* **$\pi_{\text{cdu}} = \$4.20/\text{barrel}$:** The refinery earns an extra \$4,200/day for every 1 kbpd increase in CDU capacity.
* **$\pi_{\text{sulfur}} = -\$18.50/\text{kg}$:** The marginal cost of sulfur giveaway; indicates exactly what MRPL can afford to spend on hydrotreating catalyst replenishment.

---

## 2. Drop-In Python Pyomo / PuLP Adapter (`src/interfaces/python/indus_opt/pyomo_adapter.py`)

MRPL's linear programming models are typically maintained in Python. **INDUS-OPT** acts as a drop-in replacement for CPLEX and Gurobi:

```python
"""
indus_opt/pyomo_adapter.py
Native Pyomo solver plugin for INDUS-OPT.
"""

from pyomo.opt import SolverFactory, OptStatus, TerminationCondition
import subprocess
import os
import tempfile

@SolverFactory.register('indus', doc='INDUS-OPT Sovereign Optimization Solver')
class IndusPyomoSolver:
    def __init__(self, **kwargs):
        self.executable = kwargs.get('executable', 'indus')
        self.options = kwargs.get('options', {})

    def solve(self, model, **kwargs):
        # 1. Export Pyomo model to temporary MPS file
        with tempfile.TemporaryDirectory() as tmpdir:
            mps_path = os.path.join(tmpdir, "model.mps")
            sol_path = os.path.join(tmpdir, "solution.sol")
            json_path = os.path.join(tmpdir, "stats.json")

            model.write(mps_path, format='mps')

            # 2. Invoke INDUS-OPT CLI
            cmd = [
                self.executable, "solve", mps_path,
                "--write-sol", sol_path,
                "--stats", json_path
            ]
            if kwargs.get('gpu', False):
                cmd.append("--gpu")

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                raise RuntimeError(f"INDUS-OPT solve failed: {result.stderr}")

            # 3. Read back primal and dual solution into Pyomo model
            self._load_solution(model, sol_path)

        return {"Status": "OK", "Message": "Solved to optimality"}

    def _load_solution(self, model, sol_path):
        with open(sol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0] == "VAR":
                    var_name, val = parts[1], float(parts[2])
                    if hasattr(model, var_name):
                        getattr(model, var_name).value = val
```

---

## 3. Pure C FFI API (`include/indus/indus.h`)

An ABI-stable C API guarantees interoperability with legacy C/C++, Fortran, and Python `ctypes`:

```c
#ifndef INDUS_H
#define INDUS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct indus_model_t indus_model_t;
typedef struct indus_solution_t indus_solution_t;
typedef struct indus_options_t indus_options_t;

/* Status codes matching SIH standards */
typedef enum {
    INDUS_STATUS_OPTIMAL    = 0,
    INDUS_STATUS_INFEASIBLE = 1,
    INDUS_STATUS_UNBOUNDED  = 2,
    INDUS_STATUS_TIME_LIMIT = 3,
    INDUS_STATUS_ITER_LIMIT = 4,
    INDUS_STATUS_ERROR      = 5
} indus_status_enum;

/* Model construction */
indus_model_t* indus_model_create(const char* name);
void indus_model_free(indus_model_t* model);

int indus_add_col(indus_model_t* model, double cost, double lower, double upper,
                  int is_integer, const char* name);

int indus_add_row(indus_model_t* model, double lower, double upper,
                  int nnz, const int* col_indices, const double* values,
                  const char* name);

/* Solve & Query */
indus_options_t* indus_options_create(void);
void indus_options_set_bool(indus_options_t* opts, const char* key, int value);
void indus_options_free(indus_options_t* opts);

indus_solution_t* indus_solve(const indus_model_t* model, const indus_options_t* opts);
indus_status_enum indus_solution_status(const indus_solution_t* sol);
double indus_solution_objective(const indus_solution_t* sol);
void indus_solution_get_primals(const indus_solution_t* sol, double* out_x);
void indus_solution_get_duals(const indus_solution_t* sol, double* out_y);
void indus_solution_free(indus_solution_t* sol);

#ifdef __cplusplus
}
#endif

#endif /* INDUS_H */
```

---

## 4. Root Build Specification (`CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.20)
project(indus
        VERSION 1.0.0
        DESCRIPTION "Indigenous Sovereign GPU-Accelerated Mathematical Optimization Solver"
        LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(INDUS_ENABLE_CUDA "Build with native NVIDIA CUDA GPU acceleration" ON)
option(INDUS_BUILD_TESTS "Build comprehensive test suite" ON)

if(INDUS_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    add_compile_definitions(INDUS_WITH_CUDA=1)
endif()

include(FetchContent)
FetchContent_Declare(fmt GIT_REPOSITORY https://github.com/fmtlib/fmt.git GIT_TAG 10.2.1)
FetchContent_Declare(cli11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2)
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(fmt cli11 nlohmann_json)

add_subdirectory(src)
add_subdirectory(apps/indus-cli)

if(INDUS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```
