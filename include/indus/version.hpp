#pragma once

namespace indus {

inline constexpr const char* kSolverName = "INDUS-OPT";
inline constexpr const char* kSolverAlias = "SIDDHANTA";
inline constexpr int kVersionMajor = 1;
inline constexpr int kVersionMinor = 0;
inline constexpr int kVersionPatch = 0;
inline constexpr const char* kVersionString = "1.0.0";
inline constexpr const char* kTargetProblemStatement = "SIH26119";
inline constexpr const char* kIssuingOrganization = "Mangalore Refinery and Petrochemicals Limited (MRPL)";

#if defined(__NVCC__) || defined(INDUS_ENABLE_CUDA)
inline constexpr bool kCudaEnabled = true;
#else
inline constexpr bool kCudaEnabled = false;
#endif

} // namespace indus
