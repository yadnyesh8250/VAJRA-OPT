#include "indus/io.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <iostream>
#include <stdexcept>

namespace indus::io {

namespace {

// Convert Fortran scientific notation (1.23D-04) to standard C++ (1.23E-04)
double parse_number(std::string str) {
    for (char& ch : str) {
        if (ch == 'D' || ch == 'd') ch = 'E';
    }
    return std::stod(str);
}

// Split string into tokens by whitespace
std::vector<std::string> tokenize(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

enum class MpsSection {
    kNone,
    kName,
    kObjsense,
    kRows,
    kColumns,
    kRhs,
    kRanges,
    kBounds,
    kQuadobj,
    kEndata
};

} // namespace

Model read_mps(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MPS file: " + filepath);
    }

    Model model;
    model.source_path = filepath;
    model.sense = ObjSense::kMinimize; // Default MPS sense

    MpsSection section = MpsSection::kNone;
    std::string obj_row_name = "";

    // Internal row tracking
    std::vector<std::string> row_names;
    std::unordered_map<std::string, int> row_name_to_idx;
    std::vector<char> row_types; // 'E', 'L', 'G'

    // Internal column tracking
    std::vector<std::string> col_names;
    std::unordered_map<std::string, int> col_name_to_idx;

    std::vector<la::Triplet> A_triplets;
    std::vector<la::Triplet> Q_triplets;

    std::unordered_map<int, double> rhs_values;
    std::unordered_map<int, double> range_values;

    bool in_integer_block = false;

    // Track which columns had bounds explicitly set
    std::unordered_map<int, double> col_lb_set;
    std::unordered_map<int, double> col_ub_set;
    std::unordered_map<int, VarType> col_var_types;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Skip comment lines
        if (line[0] == '*') continue;

        // Header detection (starts in column 1 without whitespace)
        if (!std::isspace(static_cast<unsigned char>(line[0]))) {
            const auto tokens = tokenize(line);
            if (tokens.empty()) continue;

            std::string header = tokens[0];
            std::transform(header.begin(), header.end(), header.begin(), ::toupper);

            if (header == "NAME") {
                section = MpsSection::kName;
                if (tokens.size() > 1) model.name = tokens[1];
                else model.name = "unnamed";
                continue;
            } else if (header == "OBJSENSE") {
                section = MpsSection::kObjsense;
                continue;
            } else if (header == "ROWS") {
                section = MpsSection::kRows;
                continue;
            } else if (header == "COLUMNS") {
                section = MpsSection::kColumns;
                continue;
            } else if (header == "RHS") {
                section = MpsSection::kRhs;
                continue;
            } else if (header == "RANGES") {
                section = MpsSection::kRanges;
                continue;
            } else if (header == "BOUNDS") {
                section = MpsSection::kBounds;
                continue;
            } else if (header == "QUADOBJ" || header == "QMATRIX") {
                section = MpsSection::kQuadobj;
                continue;
            } else if (header == "ENDATA") {
                section = MpsSection::kEndata;
                break;
            }
        }

        // Section data parsing
        const auto tokens = tokenize(line);
        if (tokens.empty()) continue;

        switch (section) {
            case MpsSection::kObjsense: {
                std::string s = tokens[0];
                std::transform(s.begin(), s.end(), s.begin(), ::toupper);
                if (s == "MAX" || s == "MAXIMIZE") {
                    model.sense = ObjSense::kMaximize;
                } else if (s == "MIN" || s == "MINIMIZE") {
                    model.sense = ObjSense::kMinimize;
                }
                break;
            }

            case MpsSection::kRows: {
                // Tokens: <type> <name>
                if (tokens.size() < 2) continue;
                char type = std::toupper(static_cast<unsigned char>(tokens[0][0]));
                std::string rname = tokens[1];

                if (type == 'N') {
                    if (obj_row_name.empty()) {
                        obj_row_name = rname;
                    }
                } else if (type == 'E' || type == 'L' || type == 'G') {
                    if (row_name_to_idx.find(rname) == row_name_to_idx.end()) {
                        const int idx = static_cast<int>(row_names.size());
                        row_names.push_back(rname);
                        row_name_to_idx[rname] = idx;
                        row_types.push_back(type);
                    }
                }
                break;
            }

            case MpsSection::kColumns: {
                // Tokens: <col_name> <row_name1> <val1> [<row_name2> <val2>]
                if (tokens.size() < 3) continue;

                const std::string cname = tokens[0];
                const std::string r1 = tokens[1];

                // Check for integer marker
                if (r1 == "'MARKER'" || r1 == "MARKER") {
                    if (tokens.size() >= 3) {
                        std::string marker_type = tokens[2];
                        if (marker_type.find("INTORG") != std::string::npos) {
                            in_integer_block = true;
                        } else if (marker_type.find("INTEND") != std::string::npos) {
                            in_integer_block = false;
                        }
                    }
                    continue;
                }

                int col_idx = -1;
                auto it = col_name_to_idx.find(cname);
                if (it == col_name_to_idx.end()) {
                    col_idx = static_cast<int>(col_names.size());
                    col_names.push_back(cname);
                    col_name_to_idx[cname] = col_idx;
                    col_var_types[col_idx] = in_integer_block ? VarType::kInteger : VarType::kContinuous;
                } else {
                    col_idx = it->second;
                    if (in_integer_block) {
                        col_var_types[col_idx] = VarType::kInteger;
                    }
                }

                // Field 1
                try {
                    const double val1 = parse_number(tokens[2]);
                    if (r1 == obj_row_name) {
                        // Objective coefficient
                        if (static_cast<size_t>(col_idx) >= model.c.size()) {
                            model.c.resize(static_cast<size_t>(col_idx + 1), 0.0);
                        }
                        model.c[static_cast<size_t>(col_idx)] += val1;
                    } else {
                        auto rit = row_name_to_idx.find(r1);
                        if (rit != row_name_to_idx.end()) {
                            A_triplets.push_back({rit->second, col_idx, val1});
                        }
                    }
                } catch (...) {}

                // Optional Field 2: <row_name2> <val2>
                if (tokens.size() >= 5) {
                    const std::string r2 = tokens[3];
                    try {
                        const double val2 = parse_number(tokens[4]);
                        if (r2 == obj_row_name) {
                            if (static_cast<size_t>(col_idx) >= model.c.size()) {
                                model.c.resize(static_cast<size_t>(col_idx + 1), 0.0);
                            }
                            model.c[static_cast<size_t>(col_idx)] += val2;
                        } else {
                            auto rit = row_name_to_idx.find(r2);
                            if (rit != row_name_to_idx.end()) {
                                A_triplets.push_back({rit->second, col_idx, val2});
                            }
                        }
                    } catch (...) {}
                }
                break;
            }

            case MpsSection::kRhs: {
                // Tokens: [<rhs_name>] <row1> <val1> [<row2> <val2>]
                size_t offset = 0;
                if (tokens.size() == 3 || tokens.size() == 5) {
                    offset = 1; // First token is rhs vector name
                }

                if (offset + 1 < tokens.size()) {
                    const std::string r1 = tokens[offset];
                    try {
                        const double v1 = parse_number(tokens[offset + 1]);
                        auto rit = row_name_to_idx.find(r1);
                        if (rit != row_name_to_idx.end()) {
                            rhs_values[rit->second] = v1;
                        }
                    } catch (...) {}
                }

                if (offset + 3 < tokens.size()) {
                    const std::string r2 = tokens[offset + 2];
                    try {
                        const double v2 = parse_number(tokens[offset + 3]);
                        auto rit = row_name_to_idx.find(r2);
                        if (rit != row_name_to_idx.end()) {
                            rhs_values[rit->second] = v2;
                        }
                    } catch (...) {}
                }
                break;
            }

            case MpsSection::kRanges: {
                // Tokens: [<rng_name>] <row1> <val1> [<row2> <val2>]
                size_t offset = 0;
                if (tokens.size() == 3 || tokens.size() == 5) {
                    offset = 1;
                }

                if (offset + 1 < tokens.size()) {
                    const std::string r1 = tokens[offset];
                    try {
                        const double v1 = parse_number(tokens[offset + 1]);
                        auto rit = row_name_to_idx.find(r1);
                        if (rit != row_name_to_idx.end()) {
                            range_values[rit->second] = v1;
                        }
                    } catch (...) {}
                }

                if (offset + 3 < tokens.size()) {
                    const std::string r2 = tokens[offset + 2];
                    try {
                        const double v2 = parse_number(tokens[offset + 3]);
                        auto rit = row_name_to_idx.find(r2);
                        if (rit != row_name_to_idx.end()) {
                            range_values[rit->second] = v2;
                        }
                    } catch (...) {}
                }
                break;
            }

            case MpsSection::kBounds: {
                // Tokens: <type> [<bnd_name>] <col_name> [<val>]
                if (tokens.size() < 2) continue;
                std::string btype = tokens[0];
                std::transform(btype.begin(), btype.end(), btype.begin(), ::toupper);

                std::string cname;
                double bval = 0.0;

                if (tokens.size() == 2) {
                    cname = tokens[1];
                } else if (tokens.size() == 3) {
                    // Could be <type> <col> <val> OR <type> <bnd_id> <col>
                    auto it_col = col_name_to_idx.find(tokens[1]);
                    if (it_col != col_name_to_idx.end()) {
                        cname = tokens[1];
                        try { bval = parse_number(tokens[2]); } catch (...) {}
                    } else {
                        cname = tokens[2];
                    }
                } else if (tokens.size() >= 4) {
                    cname = tokens[2];
                    try { bval = parse_number(tokens[3]); } catch (...) {}
                }

                auto cit = col_name_to_idx.find(cname);
                if (cit == col_name_to_idx.end()) continue;
                const int col_idx = cit->second;

                if (btype == "LO") {
                    col_lb_set[col_idx] = bval;
                } else if (btype == "UP") {
                    col_ub_set[col_idx] = bval;
                    if (bval < 0.0 && col_lb_set.find(col_idx) == col_lb_set.end()) {
                        col_lb_set[col_idx] = -1e20; // Default lower bound becomes -inf if upper < 0
                    }
                } else if (btype == "FX") {
                    col_lb_set[col_idx] = bval;
                    col_ub_set[col_idx] = bval;
                } else if (btype == "FR") {
                    col_lb_set[col_idx] = -1e20;
                    col_ub_set[col_idx] = 1e20;
                } else if (btype == "MI") {
                    col_lb_set[col_idx] = -1e20;
                } else if (btype == "PL") {
                    col_ub_set[col_idx] = 1e20;
                } else if (btype == "BV") {
                    col_lb_set[col_idx] = 0.0;
                    col_ub_set[col_idx] = 1.0;
                    col_var_types[col_idx] = VarType::kInteger;
                } else if (btype == "UI") {
                    col_ub_set[col_idx] = bval;
                    col_var_types[col_idx] = VarType::kInteger;
                } else if (btype == "LI") {
                    col_lb_set[col_idx] = bval;
                    col_var_types[col_idx] = VarType::kInteger;
                }
                break;
            }

            case MpsSection::kQuadobj: {
                // Tokens: <col1> <col2> <val>
                if (tokens.size() >= 3) {
                    auto c1_it = col_name_to_idx.find(tokens[0]);
                    auto c2_it = col_name_to_idx.find(tokens[1]);
                    if (c1_it != col_name_to_idx.end() && c2_it != col_name_to_idx.end()) {
                        try {
                            const double qval = parse_number(tokens[2]);
                            int r = c1_it->second;
                            int c = c2_it->second;
                            if (r < c) std::swap(r, c); // Enforce lower triangular
                            Q_triplets.push_back({r, c, qval});
                        } catch (...) {}
                    }
                }
                break;
            }

            default:
                break;
        }
    }

    const int m = static_cast<int>(row_names.size());
    const int n = static_cast<int>(col_names.size());

    model.num_rows = m;
    model.num_cols = n;
    model.row_names = std::move(row_names);
    model.col_names = std::move(col_names);

    // Objective vector c
    if (static_cast<int>(model.c.size()) < n) {
        model.c.resize(static_cast<size_t>(n), 0.0);
    }

    // Constraint row bounds
    model.row_lower.assign(static_cast<size_t>(m), -1e20);
    model.row_upper.assign(static_cast<size_t>(m), 1e20);

    for (int i = 0; i < m; ++i) {
        double b = 0.0;
        auto rit = rhs_values.find(i);
        if (rit != rhs_values.end()) {
            b = rit->second;
        }

        const char type = row_types[static_cast<size_t>(i)];
        auto rng_it = range_values.find(i);
        const bool has_range = (rng_it != range_values.end());
        const double r = has_range ? rng_it->second : 0.0;

        if (type == 'E') {
            if (!has_range) {
                model.row_lower[static_cast<size_t>(i)] = b;
                model.row_upper[static_cast<size_t>(i)] = b;
            } else if (r > 0.0) {
                model.row_lower[static_cast<size_t>(i)] = b;
                model.row_upper[static_cast<size_t>(i)] = b + r;
            } else {
                model.row_lower[static_cast<size_t>(i)] = b + r;
                model.row_upper[static_cast<size_t>(i)] = b;
            }
        } else if (type == 'L') {
            if (!has_range) {
                model.row_lower[static_cast<size_t>(i)] = -1e20;
                model.row_upper[static_cast<size_t>(i)] = b;
            } else {
                model.row_lower[static_cast<size_t>(i)] = b - std::abs(r);
                model.row_upper[static_cast<size_t>(i)] = b;
            }
        } else if (type == 'G') {
            if (!has_range) {
                model.row_lower[static_cast<size_t>(i)] = b;
                model.row_upper[static_cast<size_t>(i)] = 1e20;
            } else {
                model.row_lower[static_cast<size_t>(i)] = b;
                model.row_upper[static_cast<size_t>(i)] = b + std::abs(r);
            }
        }
    }

    // Column bounds and integrality
    model.col_lower.assign(static_cast<size_t>(n), 0.0);
    model.col_upper.assign(static_cast<size_t>(n), 1e20);
    model.col_type.assign(static_cast<size_t>(n), VarType::kContinuous);

    for (int j = 0; j < n; ++j) {
        auto lbit = col_lb_set.find(j);
        if (lbit != col_lb_set.end()) {
            model.col_lower[static_cast<size_t>(j)] = lbit->second;
        }

        auto ubit = col_ub_set.find(j);
        if (ubit != col_ub_set.end()) {
            model.col_upper[static_cast<size_t>(j)] = ubit->second;
        }

        auto typeit = col_var_types.find(j);
        if (typeit != col_var_types.end()) {
            model.col_type[static_cast<size_t>(j)] = typeit->second;
        }
    }

    // Assemble matrix A
    model.A.set_from_triplets(m, n, A_triplets, true);

    // Assemble matrix Q
    if (!Q_triplets.empty()) {
        model.Q.set_from_triplets(n, n, Q_triplets, true);
    }

    return model;
}

} // namespace indus::io
