#pragma once

#include "indus/model.hpp"
#include "indus/options.hpp"
#include <string>

namespace indus::qp {

enum class QpConvexity {
    kPositiveDefinite,
    kPositiveSemidefinite,
    kIndefinite,
    kNegativeDefinite,
    kInvalid
};

inline const char* to_string(QpConvexity c) noexcept {
    switch (c) {
        case QpConvexity::kPositiveDefinite: return "POSITIVE_DEFINITE";
        case QpConvexity::kPositiveSemidefinite: return "POSITIVE_SEMIDEFINITE";
        case QpConvexity::kIndefinite: return "INDEFINITE";
        case QpConvexity::kNegativeDefinite: return "NEGATIVE_DEFINITE";
        default: return "INVALID";
    }
}

// Checks convexity of Q matrix via sparse LDL^T inertia analysis
QpConvexity check_convexity(const Model& model, std::string* message = nullptr);

// Native Continuous Convex QP Solver using Primal Active-Set and sparse factorizations
Solution solve_qp(const Model& model, const Options& options = Options());

} // namespace indus::qp
