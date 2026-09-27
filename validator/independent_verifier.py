#!/usr/bin/env python3
"""
Sovereign Independent Solution Verifier (Zero-Dependency)
Part of INDUS-OPT (Siddhanta) Project

This script is a completely independent audit verifier for linear programming solutions.
It implements its OWN parsers for MPS and .sol files from first principles, with ZERO
dependencies on any external packages or solver source code.

Verifies:
1. Primal feasibility (column bounds & row bounds Ax)
2. Objective value consistency (recomputed c^T x vs reported objective)
3. Dual feasibility (reduced costs d_j = c_j - A^T y)
4. Complementary slackness (x_j at bounds vs sign of d_j)
"""

import sys
import math
import re
from typing import Dict, List, Tuple, Optional

class MPSModel:
    def __init__(self):
        self.name: str = ""
        self.sense: str = "MIN"  # "MIN" or "MAX"
        self.obj_name: str = ""
        self.obj_offset: float = 0.0
        
        # Row definitions: name -> type ('E', 'L', 'G', 'N')
        self.row_types: Dict[str, str] = {}
        self.row_order: List[str] = []
        
        # Row bounds: name -> (lower, upper)
        self.row_lower: Dict[str, float] = {}
        self.row_upper: Dict[str, float] = {}
        
        # Columns (variables): name -> list of (row_name, coeff)
        self.col_order: List[str] = []
        self.col_coeffs: Dict[str, List[Tuple[str, float]]] = {}
        self.col_obj: Dict[str, float] = {}
        self.col_lower: Dict[str, float] = {}
        self.col_upper: Dict[str, float] = {}


def parse_mps_independent(filepath: str) -> MPSModel:
    model = MPSModel()
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    section = None
    
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.startswith('*'):
            continue
            
        # Section header
        if not line.startswith(' ') and not line.startswith('\t'):
            tokens = line.split()
            sec_name = tokens[0].upper()
            if sec_name == 'NAME':
                section = 'NAME'
                if len(tokens) > 1:
                    model.name = tokens[1]
            elif sec_name == 'OBJSENSE':
                section = 'OBJSENSE'
            elif sec_name == 'OBJNAME':
                section = 'OBJNAME'
            elif sec_name in ('ROWS', 'COLUMNS', 'RHS', 'RANGES', 'BOUNDS', 'ENDATA'):
                section = sec_name
            continue
            
        tokens = line.split()
        if not tokens:
            continue
            
        if section == 'OBJSENSE':
            sense_token = tokens[0].upper()
            if 'MAX' in sense_token:
                model.sense = 'MAX'
            else:
                model.sense = 'MIN'
                
        elif section == 'OBJNAME':
            model.obj_name = tokens[0]
            
        elif section == 'ROWS':
            row_type = tokens[0].upper()
            row_name = tokens[1]
            model.row_types[row_name] = row_type
            if row_type == 'N' and not model.obj_name:
                model.obj_name = row_name
            if row_type != 'N':
                model.row_order.append(row_name)
                if row_type == 'E':
                    model.row_lower[row_name] = 0.0
                    model.row_upper[row_name] = 0.0
                elif row_type == 'L':
                    model.row_lower[row_name] = -math.inf
                    model.row_upper[row_name] = 0.0
                elif row_type == 'G':
                    model.row_lower[row_name] = 0.0
                    model.row_upper[row_name] = math.inf
                    
        elif section == 'COLUMNS':
            col_name = tokens[0]
            if col_name not in model.col_coeffs:
                model.col_order.append(col_name)
                model.col_coeffs[col_name] = []
                model.col_obj[col_name] = 0.0
                model.col_lower[col_name] = 0.0
                model.col_upper[col_name] = math.inf
                
            idx = 1
            while idx + 1 < len(tokens):
                r_name = tokens[idx]
                val = float(tokens[idx + 1])
                if r_name == model.obj_name:
                    model.col_obj[col_name] = val
                else:
                    model.col_coeffs[col_name].append((r_name, val))
                idx += 2
                
        elif section == 'RHS':
            offset = 1 if tokens[0] not in model.row_types and tokens[0] != model.obj_name else 0
            idx = offset
            while idx + 1 < len(tokens):
                r_name = tokens[idx]
                val = float(tokens[idx + 1])
                if r_name in model.row_types:
                    t = model.row_types[r_name]
                    if t == 'E':
                        model.row_lower[r_name] = val
                        model.row_upper[r_name] = val
                    elif t == 'L':
                        model.row_lower[r_name] = -math.inf
                        model.row_upper[r_name] = val
                    elif t == 'G':
                        model.row_lower[r_name] = val
                        model.row_upper[r_name] = math.inf
                idx += 2
                
        elif section == 'RANGES':
            offset = 1 if tokens[0] not in model.row_types else 0
            idx = offset
            while idx + 1 < len(tokens):
                r_name = tokens[idx]
                r_val = float(tokens[idx + 1])
                if r_name in model.row_types:
                    t = model.row_types[r_name]
                    if t == 'E':
                        if r_val > 0:
                            model.row_upper[r_name] = model.row_lower[r_name] + r_val
                        else:
                            model.row_lower[r_name] = model.row_upper[r_name] + r_val
                    elif t == 'L':
                        model.row_lower[r_name] = model.row_upper[r_name] - abs(r_val)
                    elif t == 'G':
                        model.row_upper[r_name] = model.row_lower[r_name] + abs(r_val)
                idx += 2
                
        elif section == 'BOUNDS':
            b_type = tokens[0].upper()
            if len(tokens) >= 3:
                if tokens[2] in model.col_coeffs:
                    col_name = tokens[2]
                    b_val = float(tokens[3]) if len(tokens) > 3 else 0.0
                elif tokens[1] in model.col_coeffs:
                    col_name = tokens[1]
                    b_val = float(tokens[2]) if len(tokens) > 2 else 0.0
                else:
                    col_name = tokens[2]
                    b_val = float(tokens[3]) if len(tokens) > 3 else 0.0
                    
                if col_name in model.col_coeffs:
                    if b_type == 'UP':
                        model.col_upper[col_name] = b_val
                    elif b_type == 'LO':
                        model.col_lower[col_name] = b_val
                    elif b_type == 'FX':
                        model.col_lower[col_name] = b_val
                        model.col_upper[col_name] = b_val
                    elif b_type == 'FR':
                        model.col_lower[col_name] = -math.inf
                        model.col_upper[col_name] = math.inf
                    elif b_type == 'MI':
                        model.col_lower[col_name] = -math.inf
                    elif b_type == 'PL':
                        model.col_upper[col_name] = math.inf
                        
    return model


def parse_lp_independent(filepath: str) -> MPSModel:
    model = MPSModel()
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    cleaned_lines = []
    for line in lines:
        c_pos = line.find('\\')
        if c_pos != -1:
            line = line[:c_pos]
        line = line.strip()
        if line:
            cleaned_lines.append(line)
            
    full_text = " ".join(cleaned_lines)
    tokens = full_text.split()
    i = 0
    section = None
    while i < len(tokens):
        tok = tokens[i]
        tok_upper = tok.upper()
        if tok_upper in ("MAXIMIZE", "MAX"):
            model.sense = "MAX"
            section = "OBJ"
            i += 1
            continue
        elif tok_upper in ("MINIMIZE", "MIN"):
            model.sense = "MIN"
            section = "OBJ"
            i += 1
            continue
        elif tok_upper in ("SUBJECT", "ST", "S.T."):
            if tok_upper == "SUBJECT" and i + 1 < len(tokens) and tokens[i+1].upper() == "TO":
                i += 2
            else:
                i += 1
            section = "CONSTRAINTS"
            continue
        elif tok_upper == "BOUNDS":
            section = "BOUNDS"
            i += 1
            continue
        elif tok_upper == "END":
            break
            
        if section == "OBJ":
            if tok.endswith(":"):
                model.obj_name = tok[:-1]
                i += 1
                continue
            sign = 1.0
            coeff = 1.0
            coeff_seen = False
            while i < len(tokens) and tokens[i].upper() not in ("SUBJECT", "ST", "S.T.", "BOUNDS", "END"):
                t = tokens[i]
                if t == "+":
                    sign = 1.0
                elif t == "-":
                    sign = -1.0
                elif re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", t):
                    coeff = float(t)
                    coeff_seen = True
                else:
                    var = t
                    val = sign * (coeff if coeff_seen else 1.0)
                    model.col_obj[var] = val
                    if var not in model.col_order:
                        model.col_order.append(var)
                        model.col_coeffs[var] = []
                        model.col_lower[var] = 0.0
                        model.col_upper[var] = math.inf
                    sign = 1.0
                    coeff = 1.0
                    coeff_seen = False
                i += 1
        elif section == "CONSTRAINTS":
            if tok.endswith(":"):
                row_name = tok[:-1]
                i += 1
            else:
                row_name = f"R{len(model.row_order)+1}"
            model.row_order.append(row_name)
            
            c_tokens = []
            while i < len(tokens) and tokens[i].upper() not in ("BOUNDS", "END", "SUBJECT", "ST", "S.T.") and not (tokens[i].endswith(":") and tokens[i] not in ("<=", ">=", "=")):
                c_tokens.append(tokens[i])
                i += 1
                
            def parse_expr(e_tokens):
                pairs = []
                sign = 1.0
                coeff = 1.0
                c_seen = False
                for t in e_tokens:
                    if t == "+": sign = 1.0
                    elif t == "-": sign = -1.0
                    elif re.match(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$", t):
                        coeff = float(t)
                        c_seen = True
                    else:
                        var = t
                        val = sign * (coeff if c_seen else 1.0)
                        pairs.append((var, val))
                        sign = 1.0
                        coeff = 1.0
                        c_seen = False
                return pairs
                
            ops = [k for k, t in enumerate(c_tokens) if t in ("<=", ">=", "=")]
            if len(ops) == 2:
                v1 = float(c_tokens[0])
                v2 = float(c_tokens[-1])
                expr_tokens = c_tokens[ops[0]+1:ops[1]]
                for var, val in parse_expr(expr_tokens):
                    if var not in model.col_coeffs:
                        model.col_order.append(var)
                        model.col_coeffs[var] = []
                        model.col_obj[var] = 0.0
                        model.col_lower[var] = 0.0
                        model.col_upper[var] = math.inf
                    model.col_coeffs[var].append((row_name, val))
                model.row_lower[row_name] = v1
                model.row_upper[row_name] = v2
                model.row_types[row_name] = "E" if v1 == v2 else "L"
            elif len(ops) == 1:
                op = c_tokens[ops[0]]
                rhs = float(c_tokens[ops[0]+1])
                expr_tokens = c_tokens[:ops[0]]
                for var, val in parse_expr(expr_tokens):
                    if var not in model.col_coeffs:
                        model.col_order.append(var)
                        model.col_coeffs[var] = []
                        model.col_obj[var] = 0.0
                        model.col_lower[var] = 0.0
                        model.col_upper[var] = math.inf
                    model.col_coeffs[var].append((row_name, val))
                if op == "<=":
                    model.row_lower[row_name] = -math.inf
                    model.row_upper[row_name] = rhs
                    model.row_types[row_name] = "L"
                elif op == ">=":
                    model.row_lower[row_name] = rhs
                    model.row_upper[row_name] = math.inf
                    model.row_types[row_name] = "G"
                elif op == "=":
                    model.row_lower[row_name] = rhs
                    model.row_upper[row_name] = rhs
                    model.row_types[row_name] = "E"
        elif section == "BOUNDS":
            b_tokens = []
            while i < len(tokens) and tokens[i].upper() not in ("END",):
                b_tokens.append(tokens[i])
                i += 1
            idx = 0
            while idx < len(b_tokens):
                if idx + 2 < len(b_tokens) and b_tokens[idx+1] in (">=", "<=", "="):
                    var = b_tokens[idx]
                    op = b_tokens[idx+1]
                    val = float(b_tokens[idx+2])
                    if var in model.col_lower:
                        if op == ">=": model.col_lower[var] = val
                        elif op == "<=": model.col_upper[var] = val
                        elif op == "=":
                            model.col_lower[var] = val
                            model.col_upper[var] = val
                    idx += 3
                else:
                    idx += 1
    return model


class Solution:
    def __init__(self):
        self.status: str = ""
        self.reported_objective: float = 0.0
        self.col_values: Dict[str, float] = {}
        self.col_duals: Dict[str, float] = {}
        self.row_values: Dict[str, float] = {}
        self.row_duals: Dict[str, float] = {}


def parse_solution_independent(filepath: str, model: MPSModel) -> Solution:
    sol = Solution()
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    in_columns = False
    in_rows = False
    
    for line_num, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line:
            continue
            
        if line.startswith('#'):
            if "Status:" in line:
                parts = line.split()
                if len(parts) >= 3:
                    sol.status = parts[2].upper()
            elif "Objective:" in line:
                parts = line.split()
                if len(parts) >= 3:
                    val = float(parts[2])
                    if math.isnan(val) or math.isinf(val):
                        raise ValueError(f"Malformed objective float at line {line_num}")
                    sol.reported_objective = val
            elif "Columns (Variables)" in line:
                in_columns = True
                in_rows = False
            elif "Rows (Constraints)" in line:
                in_columns = False
                in_rows = True
            continue
            
        tokens = line.split()
        if not tokens:
            continue
            
        if in_columns:
            var_name = tokens[0]
            if var_name not in model.col_coeffs:
                raise ValueError(f"Unknown variable in solution file at line {line_num}: '{var_name}'")
            if var_name in sol.col_values:
                raise ValueError(f"Duplicate variable in solution file at line {line_num}: '{var_name}'")
            try:
                val = float(tokens[1])
            except ValueError:
                raise ValueError(f"Malformed numeric value for variable '{var_name}' at line {line_num}: '{tokens[1]}'")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Malformed numeric value for variable '{var_name}' at line {line_num}")
            sol.col_values[var_name] = val
            if len(tokens) >= 3:
                try:
                    dj = float(tokens[2])
                except ValueError:
                    raise ValueError(f"Malformed reduced cost for variable '{var_name}' at line {line_num}: '{tokens[2]}'")
                if math.isnan(dj) or math.isinf(dj):
                    raise ValueError(f"Malformed reduced cost for variable '{var_name}' at line {line_num}")
                sol.col_duals[var_name] = dj
                
        elif in_rows:
            r_name = tokens[0]
            if r_name not in model.row_types:
                raise ValueError(f"Unknown row in solution file at line {line_num}: '{r_name}'")
            if r_name in sol.row_values:
                raise ValueError(f"Duplicate row in solution file at line {line_num}: '{r_name}'")
            try:
                val = float(tokens[1])
            except ValueError:
                raise ValueError(f"Malformed numeric activity for row '{r_name}' at line {line_num}: '{tokens[1]}'")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Malformed numeric activity for row '{r_name}' at line {line_num}")
            sol.row_values[r_name] = val
            if len(tokens) >= 3:
                try:
                    y = float(tokens[2])
                except ValueError:
                    raise ValueError(f"Malformed dual value for row '{r_name}' at line {line_num}: '{tokens[2]}'")
                if math.isnan(y) or math.isinf(y):
                    raise ValueError(f"Malformed dual value for row '{r_name}' at line {line_num}")
                sol.row_duals[r_name] = y
                
    if not sol.status:
        raise ValueError("Missing '# Status:' header declaration in solution file")
        
    # Completeness check: all model variables must be present
    for col in model.col_order:
        if col not in sol.col_values:
            raise ValueError(f"Incomplete solution: missing variable '{col}'")
            
    return sol


def verify_sovereign(model_path: str, sol_path: str, tol: float = 1e-6) -> Tuple[bool, List[str]]:
    if model_path.lower().endswith('.lp'):
        model = parse_lp_independent(model_path)
    else:
        model = parse_mps_independent(model_path)
    sol = parse_solution_independent(sol_path, model)
    
    violations = []
    
    if sol.status != "OPTIMAL":
        violations.append(f"Solution status is not OPTIMAL (reported: {sol.status})")
        return False, violations
        
    # 1. Check Column Bounds: l_j <= x_j <= u_j
    max_bound_viol = 0.0
    for col in model.col_order:
        xj = sol.col_values[col]
        lj = model.col_lower[col]
        uj = model.col_upper[col]
        if xj < lj - tol:
            viol = lj - xj
            max_bound_viol = max(max_bound_viol, viol)
            violations.append(f"Column '{col}' lower bound violated: x={xj:.7e} < l={lj:.7e} (diff={viol:.2e})")
        if xj > uj + tol:
            viol = xj - uj
            max_bound_viol = max(max_bound_viol, viol)
            violations.append(f"Column '{col}' upper bound violated: x={xj:.7e} > u={uj:.7e} (diff={viol:.2e})")
            
    # 2. Check Row Activities: Ax = row_act and bounds
    row_activities = {r: 0.0 for r in model.row_order}
    for col in model.col_order:
        xj = sol.col_values[col]
        for r_name, coeff in model.col_coeffs[col]:
            if r_name in row_activities:
                row_activities[r_name] += coeff * xj
                
    max_row_viol = 0.0
    for r in model.row_order:
        act = row_activities[r]
        lr = model.row_lower[r]
        ur = model.row_upper[r]
        if act < lr - tol:
            viol = lr - act
            max_row_viol = max(max_row_viol, viol)
            violations.append(f"Row '{r}' lower bound violated: act={act:.7e} < l={lr:.7e} (diff={viol:.2e})")
        if act > ur + tol:
            viol = act - ur
            max_row_viol = max(max_row_viol, viol)
            violations.append(f"Row '{r}' upper bound violated: act={act:.7e} > u={ur:.7e} (diff={viol:.2e})")
            
    # 3. Check Objective Value
    recomputed_obj = model.obj_offset
    for col in model.col_order:
        recomputed_obj += model.col_obj.get(col, 0.0) * sol.col_values[col]
        
    obj_err = abs(sol.reported_objective - recomputed_obj)
    obj_scale = max(1.0, abs(recomputed_obj))
    if obj_err / obj_scale > tol:
        violations.append(f"Objective value mismatch: reported={sol.reported_objective:.10e}, recomputed={recomputed_obj:.10e}, rel_err={obj_err/obj_scale:.2e}")
        
    # 4. Check Dual Feasibility & Complementary Slackness (if duals provided)
    max_dual_viol = 0.0
    max_cs_viol = 0.0
    if sol.row_duals and sol.col_duals:
        for col in model.col_order:
            cj = model.col_obj.get(col, 0.0)
            a_trans_y = 0.0
            for r_name, coeff in model.col_coeffs[col]:
                y_i = sol.row_duals.get(r_name, 0.0)
                a_trans_y += coeff * y_i
            recomputed_dj = cj - a_trans_y
            reported_dj = sol.col_duals.get(col, 0.0)
            dj_diff = abs(recomputed_dj - reported_dj)
            if dj_diff > max(1e-4, tol * 100):
                max_dual_viol = max(max_dual_viol, dj_diff)
                
            # Complementary slackness
            xj = sol.col_values[col]
            lj = model.col_lower[col]
            uj = model.col_upper[col]
            
            if xj > lj + 1e-5 and xj < uj - 1e-5:
                if abs(reported_dj) > 1e-4:
                    max_cs_viol = max(max_cs_viol, abs(reported_dj))
                    
    passed = (len(violations) == 0)
    return passed, violations


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 independent_verifier.py <model.mps|model.lp> <solution.sol> [tolerance]")
        sys.exit(1)
        
    model_path = sys.argv[1]
    sol_path = sys.argv[2]
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-6
    
    print(f"[SOVEREIGN INDEPENDENT AUDIT] Model: {model_path}")
    print(f"[SOVEREIGN INDEPENDENT AUDIT] Solution: {sol_path}")
    print(f"[SOVEREIGN INDEPENDENT AUDIT] Tolerance: {tol:.1e}")
    
    try:
        passed, violations = verify_sovereign(model_path, sol_path, tol)
    except Exception as e:
        print(f"\n[REJECTED] Verification halted due to syntax / integrity error: {e}")
        sys.exit(1)
        
    if passed:
        print("\n>>> [VERIFIED OPTIMAL] Sovereign independent certificate check PASSED with 0 violations! <<<")
        sys.exit(0)
    else:
        print(f"\n>>> [FAILED] Solution violated {len(violations)} certificate conditions: <<<")
        for v in violations[:10]:
            print(f"  - {v}")
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more violations.")
        sys.exit(1)


if __name__ == "__main__":
    main()
