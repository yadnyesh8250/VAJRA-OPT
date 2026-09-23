#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <limits>

namespace indus {

using Index = int;
using BigIndex = int64_t;
using Float = double;

inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

enum class ObjSense : uint8_t { 
    kMinimize = 0, 
    kMaximize = 1 
};

enum class VarType : uint8_t { 
    kContinuous = 0, 
    kInteger = 1 
};

enum class BasisStatus : uint8_t {
    kUnknown = 0,
    kBasic = 1,
    kAtLower = 2,
    kAtUpper = 3,
    kNonbasicFree = 4,
    kFixed = 5
};

enum class SolveStatus : uint8_t {
    kNotSolved = 0,
    kOptimal = 1,
    kInfeasible = 2,
    kUnbounded = 3,
    kInfeasibleOrUnbounded = 4,
    kFeasible = 5,
    kIterationLimit = 6,
    kTimeLimit = 7,
    kNodeLimit = 8,
    kNumericalError = 9,
    kModelError = 10
};

[[nodiscard]] inline constexpr bool claims_a_point(SolveStatus status) noexcept {
    switch (status) {
        case SolveStatus::kOptimal:
        case SolveStatus::kFeasible:
        case SolveStatus::kUnbounded:
        case SolveStatus::kIterationLimit:
        case SolveStatus::kTimeLimit:
        case SolveStatus::kNodeLimit: 
            return true;
        default: 
            return false;
    }
}

inline const char* to_string(SolveStatus status) noexcept {
    switch (status) {
        case SolveStatus::kNotSolved: return "NOT_SOLVED";
        case SolveStatus::kOptimal: return "OPTIMAL";
        case SolveStatus::kInfeasible: return "INFEASIBLE";
        case SolveStatus::kUnbounded: return "UNBOUNDED";
        case SolveStatus::kInfeasibleOrUnbounded: return "INFEASIBLE_OR_UNBOUNDED";
        case SolveStatus::kFeasible: return "FEASIBLE";
        case SolveStatus::kIterationLimit: return "ITERATION_LIMIT";
        case SolveStatus::kTimeLimit: return "TIME_LIMIT";
        case SolveStatus::kNodeLimit: return "NODE_LIMIT";
        case SolveStatus::kNumericalError: return "NUMERICAL_ERROR";
        case SolveStatus::kModelError: return "MODEL_ERROR";
        default: return "UNKNOWN";
    }
}

inline const char* to_string(BasisStatus status) noexcept {
    switch (status) {
        case BasisStatus::kUnknown: return "UNKNOWN";
        case BasisStatus::kBasic: return "BASIC";
        case BasisStatus::kAtLower: return "AT_LOWER";
        case BasisStatus::kAtUpper: return "AT_UPPER";
        case BasisStatus::kNonbasicFree: return "NONBASIC_FREE";
        case BasisStatus::kFixed: return "FIXED";
        default: return "UNKNOWN";
    }
}

inline const char* to_string(VarType type) noexcept {
    switch (type) {
        case VarType::kContinuous: return "CONTINUOUS";
        case VarType::kInteger: return "INTEGER";
        default: return "UNKNOWN";
    }
}

inline const char* to_string(ObjSense sense) noexcept {
    switch (sense) {
        case ObjSense::kMinimize: return "MINIMIZE";
        case ObjSense::kMaximize: return "MAXIMIZE";
        default: return "UNKNOWN";
    }
}

} // namespace indus
