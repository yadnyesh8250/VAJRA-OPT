#pragma once

#include <string>
#include <unordered_map>
#include <variant>
#include <cstdint>

namespace indus {

class Options {
public:
    Options();

    void set(const std::string& key, const std::string& value);
    void set(const std::string& key, double value);
    void set(const std::string& key, int64_t value);
    void set(const std::string& key, bool value);

    [[nodiscard]] std::string get_string(const std::string& key, const std::string& default_val = "") const;
    [[nodiscard]] double get_double(const std::string& key, double default_val = 0.0) const;
    [[nodiscard]] int64_t get_int(const std::string& key, int64_t default_val = 0) const;
    [[nodiscard]] bool get_bool(const std::string& key, bool default_val = false) const;

    // Built-in shortcut accessors
    double time_limit = 1e20;
    int64_t iteration_limit = 1000000;
    int64_t node_limit = 500000;
    double mip_relative_gap = 1e-4;
    double mip_absolute_gap = 1e-6;
    bool enable_presolve = true;
    bool enable_scaling = true;
    bool enable_root_cuts = false;
    bool use_gpu = false;
    std::string algorithm = "auto"; // "auto", "simplex", "dual_simplex", "primal_simplex", "pdhg", "ipm"

private:
    std::unordered_map<std::string, std::variant<std::string, double, int64_t, bool>> values_;
};

} // namespace indus
