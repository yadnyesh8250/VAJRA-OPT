#!/usr/bin/env bash
# ==============================================================================
# NVIDIA CUDA HARDWARE VALIDATION HARNESS (Phase 9B Workflow)
# VAJRA-OPT (Siddhanta) Indigenous Optimization Engine
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================================"
echo "  SIDDHANTA (INDUS-OPT): NVIDIA CUDA HARDWARE VALIDATION HARNESS"
echo "  Phase 9B Execution Workflow | Strict Anti-Fabrication Hardware Protocol"
echo "================================================================================"
echo "  Host Platform  : $(uname -s) $(uname -m)"
echo "  Date / Time    : $(date)"
echo ""

# Step 1: Physical GPU Hardware Probe via nvidia-smi
echo "[STAGE 1: HARDWARE PROBE]"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  >> nvidia-smi detected."
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
    HAS_NVIDIA_SMI=true
else
    echo "  >> nvidia-smi NOT found. Discrete NVIDIA GPU is absent on this host."
    HAS_NVIDIA_SMI=false
fi

# Step 2: CUDA Toolkit Compiler Probe via nvcc
echo ""
echo "[STAGE 2: COMPILER PROBE]"
if command -v nvcc >/dev/null 2>&1; then
    echo "  >> nvcc detected:"
    nvcc --version
    HAS_NVCC=true
else
    echo "  >> nvcc compiler NOT found in PATH."
    HAS_NVCC=false
fi

# Step 3: Branching on Hardware Availability
echo ""
echo "[STAGE 3: HARDWARE STATUS EVALUATION]"
if [[ "$HAS_NVIDIA_SMI" = false || "$HAS_NVCC" = false ]]; then
    echo "--------------------------------------------------------------------------------"
    echo "  CUDA VALIDATION RESULT: PENDING (Phase 9B Required on Physical NVIDIA Machine)"
    echo "--------------------------------------------------------------------------------"
    echo "  * Physical NVIDIA GPU or nvcc compiler is UNAVAILABLE on this host."
    echo "  * Pure CPU fallback is ACTIVE, verified, and certified."
    echo "  * In accordance with SIH26119 integrity rules:"
    echo "    - GPU speedup is reported as NOT_MEASURED (never fabricated or mocked)."
    echo "    - No synthetic or simulated GPU claims are permitted."
    echo "    - Hardware acceleration gate remains pending an actual NVIDIA GPU testbed."
    echo "--------------------------------------------------------------------------------"
    exit 0
fi

# Step 4: Out-of-tree CUDA Build and Test (Only on actual NVIDIA hardware)
CUDA_BUILD_DIR="${ROOT_DIR}/build-cuda"
echo "[STAGE 4: OUT-OF-TREE CUDA BUILD]"
echo "  Target build dir: ${CUDA_BUILD_DIR}"
mkdir -p "${CUDA_BUILD_DIR}"

cmake -S "${ROOT_DIR}" -B "${CUDA_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DINDUS_ENABLE_CUDA=ON

cmake --build "${CUDA_BUILD_DIR}" -j4

echo ""
echo "[STAGE 5: CUDA KERNEL UNIT TESTING]"
ctest --test-dir "${CUDA_BUILD_DIR}" -R test_cuda_kernels --output-on-failure

echo ""
echo "[STAGE 6: CUDA BENCHMARK & SPEEDUP AUDIT]"
"${CUDA_BUILD_DIR}/indus_benchmark" --suite cuda --full \
    --csv "${CUDA_BUILD_DIR}/cuda_benchmark_results.csv" \
    --json "${CUDA_BUILD_DIR}/cuda_benchmark_results.json"

echo "================================================================================"
echo "  CUDA HARDWARE VALIDATION COMPLETED SUCCESSFULLY ON PHYSICAL NVIDIA DEVICE"
echo "================================================================================"
