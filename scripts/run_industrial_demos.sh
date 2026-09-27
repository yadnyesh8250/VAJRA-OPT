#!/usr/bin/env bash
# ==============================================================================
# INDUSTRIAL DEMONSTRATION SUITE FOR MRPL PROBLEM STATEMENT 26119
# VAJRA-OPT (Siddhanta) Indigenous Solver
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOLVER_BIN=""
OUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --solver-bin)
            SOLVER_BIN="$2"
            shift 2
            ;;
        --output-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--solver-bin <path>] [--output-dir <path>]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$SOLVER_BIN" ]]; then
    if [[ -n "${INDUS_SOLVER_BIN:-}" && -x "${INDUS_SOLVER_BIN}" ]]; then
        SOLVER_BIN="${INDUS_SOLVER_BIN}"
    else
        for b in "${ROOT_DIR}/build/indus_solve" "${ROOT_DIR}/build-cpu/indus_solve" "indus_solve"; do
            if [[ -x "$b" ]]; then
                SOLVER_BIN="$b"
                break
            fi
        done
    fi
fi

if [[ -z "$SOLVER_BIN" ]]; then
    echo "[ERROR] indus_solve executable not found. Please build the project or pass --solver-bin."
    exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="${ROOT_DIR}/build/industrial_demos"
fi
mkdir -p "$OUT_DIR"

echo "================================================================================"
echo "  SIDDHANTA (INDUS-OPT): INDUSTRIAL CASE STUDY DEMONSTRATIONS"
echo "  MRPL Problem Statement 26119 | Refinery Economics & Supply Optimization"
echo "================================================================================"
echo "  Solver Binary : ${SOLVER_BIN}"
echo "  Artifacts Dir : ${OUT_DIR}"
echo ""
echo "  [DATA PROVENANCE NOTICE]"
echo "  All models evaluated below are industrial-style synthetic demonstrations"
echo "  engineered to reflect refinery process topologies, CDU blending modes, lot"
echo "  sizing, and power dispatch. They do NOT contain confidential or real MRPL"
echo "  operational telemetry."
echo ""

run_demo() {
    local label="$1"
    local model_type="$2"
    local model_file="$3"
    local sol_file="${OUT_DIR}/$(basename "${model_file}" .mps).sol"
    local json_file="${OUT_DIR}/$(basename "${model_file}" .mps).json"

    echo "--------------------------------------------------------------------------------"
    echo "  [DEMO] ${label}"
    echo "  Category     : ${model_type}"
    echo "  Model Path   : ${model_file}"
    echo "--------------------------------------------------------------------------------"

    "${SOLVER_BIN}" --input "${model_file}" --output "${sol_file}" --tol 1e-4

    echo ""
    echo "  >> Independent Python Verification Audit:"
    python3 "${ROOT_DIR}/validator/independent_verifier.py" "${model_file}" "${sol_file}" 1e-4
    echo ""
}

# 1. Crude Blending (LP)
run_demo "MRPL CDU Crude Blending (3-Crude Margin Optimization)" \
         "Industrial-style synthetic demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend.mps"

# 2. Mode-Switch Crude Blending (MILP)
run_demo "MRPL Crude Blending with Discrete CDU Operating Mode Selection" \
         "Industrial-style synthetic demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/blend_milp.mps"

# 3. Refinery Production Lot Sizing (MILP)
run_demo "Multi-Period Refinery Production Lot Sizing with Setup Binaries" \
         "Industrial-style synthetic demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/lot_sizing.mps"

# 4. Depot Distribution Logistics (LP)
run_demo "Refinery to Depot Petroleum Product Distribution (Transportation LP)" \
         "Industrial-style synthetic demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/supply_chain.mps"

# 5. Cogeneration Power Dispatch (MILP)
run_demo "Refinery Captive Power Plant Unit Commitment & Economic Dispatch" \
         "Industrial-style synthetic demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/power_dispatch.mps"

# 6. Convex Crude Blending with Non-Linear Blending Penalties (QP)
run_demo "MRPL Quadratic Blending Pool (Convex Quality Giveaway Minimization)" \
         "Industrial-style synthetic demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/crude_blend_qp.mps"

# 7. Extreme Matrix Condition Stress Test (LP)
run_demo "Ill-Conditioned Constraint Matrix Numerical Robustness Test" \
         "Synthetic numerical stress demonstration" \
         "${ROOT_DIR}/SOVEREIGN_SOLVER_BLUEPRINT/test_models/ill_conditioned.mps"

echo "================================================================================"
echo "  ALL INDUSTRIAL DEMONSTRATIONS SUCCESSFULLY COMPLETED & VERIFIED"
echo "  Artifacts saved to: ${OUT_DIR}"
echo "================================================================================"
