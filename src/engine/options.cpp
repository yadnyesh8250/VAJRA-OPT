#include "indus/options.hpp"
#include <algorithm>

namespace indus {

Options::Options() {
    set("time_limit", time_limit);
    set("iteration_limit", iteration_limit);
    set("node_limit", node_limit);
    set("mip_relative_gap", mip_relative_gap);
    set("mip_absolute_gap", mip_absolute_gap);
    set("enable_presolve", enable_presolve);
    set("enable_scaling", enable_scaling);
    set("enable_root_cuts", enable_root_cuts);
    set("use_gpu", use_gpu);
    set("algorithm", algorithm);
}

void Options::set(const std::string& key, const std::string& value) {
    values_[key] = value;
    if (key == "algorithm") algorithm = value;
}

void Options::set(const std::string& key, double value) {
    values_[key] = value;
    if (key == "time_limit") time_limit = value;
    else if (key == "mip_relative_gap") mip_relative_gap = value;
    else if (key == "mip_absolute_gap") mip_absolute_gap = value;
}

void Options::set(const std::string& key, int64_t value) {
    values_[key] = value;
    if (key == "iteration_limit") iteration_limit = value;
    else if (key == "node_limit") node_limit = value;
}

void Options::set(const std::string& key, bool value) {
    values_[key] = value;
    if (key == "enable_presolve") enable_presolve = value;
    else if (key == "enable_scaling") enable_scaling = value;
    else if (key == "enable_root_cuts") enable_root_cuts = value;
    else if (key == "use_gpu") use_gpu = value;
}

std::string Options::get_string(const std::string& key, const std::string& default_val) const {
    auto it = values_.find(key);
    if (it != values_.end() && std::holds_alternative<std::string>(it->second)) {
        return std::get<std::string>(it->second);
    }
    return default_val;
}

double Options::get_double(const std::string& key, double default_val) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        if (std::holds_alternative<double>(it->second)) {
            return std::get<double>(it->second);
        } else if (std::holds_alternative<int64_t>(it->second)) {
            return static_cast<double>(std::get<int64_t>(it->second));
        }
    }
    return default_val;
}

int64_t Options::get_int(const std::string& key, int64_t default_val) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        if (std::holds_alternative<int64_t>(it->second)) {
            return std::get<int64_t>(it->second);
        } else if (std::holds_alternative<double>(it->second)) {
            return static_cast<int64_t>(std::get<double>(it->second));
        }
    }
    return default_val;
}

bool Options::get_bool(const std::string& key, bool default_val) const {
    auto it = values_.find(key);
    if (it != values_.end() && std::holds_alternative<bool>(it->second)) {
        return std::get<bool>(it->second);
    }
    return default_val;
}

} // namespace indus
