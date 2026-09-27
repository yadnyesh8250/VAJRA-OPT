# NVIDIA CUDA Hardware Validation Workflow (Phase 9B Protocol)

## 1. Executive Summary & Hardware Notice

VAJRA-OPT (Siddhanta) contains custom native NVIDIA CUDA kernels for first-order operator-splitting algorithms:
- `cuda_warp_spmv`: Warp-synchronous sparse matrix-vector multiplication using `__shfl_down_sync`.
- `cuda_pdhg_step`: Fused primal-dual vector updates with coordinate projection.
- `cuda_vector_ops`: Fused AXPY, coordinate clamping, and dot products.
- `cuda_halpn_accel`: High-order Halpern fixed-point acceleration.

**Current Host Hardware Status**:
- Evaluated on **Apple Silicon (macOS ARM64)** without discrete NVIDIA GPU hardware.
- `nvidia-smi` and `nvcc` are absent on this host.
- In strict adherence to SIH26119 rules:
  - **No CUDA execution is claimed on this machine.**
  - **GPU speedup is reported as `NOT_MEASURED` (never 1.00x, 0.00x, or fabricated).**
  - **Execution backend is reported as `CPU FALLBACK`.**
  - **CUDA status is reported as `NOT AVAILABLE`.**
  - **Physical hardware validation is marked as `PENDING` (Phase 9B).**

---

## 2. Phase 9B Execution Workflow on Physical NVIDIA Hardware

When deploying to a physical NVIDIA testbed (e.g. NVIDIA A100, H100, RTX 4090, or V100 GPU server):

### Step 1: Pre-Flight Hardware & Compiler Checks
```bash
# Verify NVIDIA GPU driver and hardware availability
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2     |
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# |   0  NVIDIA RTX 4090      Off | 00000000:01:00.0 Off |                  N/A |
# +-----------------------------------------------------------------------------+

# Verify CUDA Toolkit compiler
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Cuda compilation tools, release 12.x, V12.x.xxx
```

### Step 2: Clean CUDA Build
```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DINDUS_ENABLE_CUDA=ON

cmake --build build-cuda -j4
```

### Step 3: Run CUDA Unit Tests
```bash
ctest --test-dir build-cuda -R test_cuda_kernels --output-on-failure
```

### Step 4: Full Telemetry & Speedup Benchmark
```bash
./build-cuda/indus_benchmark \
  --suite cuda \
  --full \
  --csv build-cuda/cuda_benchmark_results.csv \
  --json build-cuda/cuda_benchmark_results.json
```

### Step 5: Execute Automated Harness
Run the provided automated script:
```bash
./scripts/validate_cuda.sh
```

---

## 3. Strict Verification & Speedup Reporting Contract

A GPU speedup ratio ($T_{\text{CPU}} / T_{\text{GPU}}$) may be recorded **only if and only if all of the following conditions are met**:
1. `nvcc` and a physical NVIDIA device are present.
2. A real CUDA kernel executes via `cudaLaunchKernel`.
3. Host-device memory transfers are non-zero (`transfers > 0`).
4. Both CPU and GPU solvers use identical tolerances ($10^{-4}$) and iteration limits.
5. Both CPU and GPU solutions pass independent primal feasibility verification.
6. The objective values match within tolerance ($10^{-6}$).

If any of these conditions fails or if hardware is absent, speedup MUST be recorded as `NOT_MEASURED`.
