#pragma once

#include <string>
#include "indus/model.hpp"

namespace indus::io {

// High-performance MPS Reader (Fixed and Free format)
Model read_mps(const std::string& filepath);

// CPLEX-style Algebraic LP Reader
Model read_lp(const std::string& filepath);

// Solution exporter (.sol text format)
void write_solution(const Solution& solution, const Model& model, const std::string& filepath);

// Telemetry & Statistics exporter (.json format)
void write_json(const Solution& solution, const Model& model, const std::string& filepath);

} // namespace indus::io
