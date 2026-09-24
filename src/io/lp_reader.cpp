#include "indus/io.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace indus::io {

namespace {

enum class TokenType {
    kIdent,
    kNumber,
    kColon,
    kPlus,
    kMinus,
    kLe,
    kGe,
    kEq,
    kLt,
    kGt,
    kLBracket,
    kRBracket,
    kAsterisk,
    kSlash,
    kCaret,
    kEof
};

struct Token {
    TokenType type = TokenType::kEof;
    std::string text;
    double num_val = 0.0;
    int line = 1;
};

std::string to_upper(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return s;
}

bool is_infinity_str(const std::string& s) {
    const std::string u = to_upper(s);
    return (u == "INF" || u == "INFINITY" || u == "+INF" || u == "-INF");
}

class LpLexer {
public:
    explicit LpLexer(std::string content) : src_(std::move(content)), pos_(0), line_(1) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (pos_ < src_.size()) {
            skip_whitespace_and_comments();
            if (pos_ >= src_.size()) break;

            const char c = src_[pos_];
            const int cur_line = line_;

            if (c == ':') {
                tokens.push_back({TokenType::kColon, ":", 0.0, cur_line});
                ++pos_;
            } else if (c == '+') {
                tokens.push_back({TokenType::kPlus, "+", 0.0, cur_line});
                ++pos_;
            } else if (c == '-') {
                tokens.push_back({TokenType::kMinus, "-", 0.0, cur_line});
                ++pos_;
            } else if (c == '<') {
                if (pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                    tokens.push_back({TokenType::kLe, "<=", 0.0, cur_line});
                    pos_ += 2;
                } else {
                    tokens.push_back({TokenType::kLt, "<", 0.0, cur_line});
                    ++pos_;
                }
            } else if (c == '>') {
                if (pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                    tokens.push_back({TokenType::kGe, ">=", 0.0, cur_line});
                    pos_ += 2;
                } else {
                    tokens.push_back({TokenType::kGt, ">", 0.0, cur_line});
                    ++pos_;
                }
            } else if (c == '=') {
                if (pos_ + 1 < src_.size() && src_[pos_ + 1] == '<') {
                    tokens.push_back({TokenType::kLe, "=<", 0.0, cur_line});
                    pos_ += 2;
                } else if (pos_ + 1 < src_.size() && src_[pos_ + 1] == '>') {
                    tokens.push_back({TokenType::kGe, "=>", 0.0, cur_line});
                    pos_ += 2;
                } else if (pos_ + 1 < src_.size() && src_[pos_ + 1] == '=') {
                    tokens.push_back({TokenType::kEq, "==", 0.0, cur_line});
                    pos_ += 2;
                } else {
                    tokens.push_back({TokenType::kEq, "=", 0.0, cur_line});
                    ++pos_;
                }
            } else if (c == '[') {
                tokens.push_back({TokenType::kLBracket, "[", 0.0, cur_line});
                ++pos_;
            } else if (c == ']') {
                tokens.push_back({TokenType::kRBracket, "]", 0.0, cur_line});
                ++pos_;
            } else if (c == '*') {
                tokens.push_back({TokenType::kAsterisk, "*", 0.0, cur_line});
                ++pos_;
            } else if (c == '/') {
                tokens.push_back({TokenType::kSlash, "/", 0.0, cur_line});
                ++pos_;
            } else if (c == '^') {
                tokens.push_back({TokenType::kCaret, "^", 0.0, cur_line});
                ++pos_;
            } else if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && pos_ + 1 < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_ + 1])))) {
                tokens.push_back(scan_number());
            } else {
                tokens.push_back(scan_ident());
            }
        }
        tokens.push_back({TokenType::kEof, "", 0.0, line_});
        return tokens;
    }

private:
    void skip_whitespace_and_comments() {
        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (c == '\n') {
                ++line_;
                ++pos_;
            } else if (std::isspace(static_cast<unsigned char>(c))) {
                ++pos_;
            } else if (c == '\\') {
                // Line comment
                ++pos_;
                while (pos_ < src_.size() && src_[pos_] != '\n') {
                    ++pos_;
                }
            } else {
                break;
            }
        }
    }

    Token scan_number() {
        const size_t start = pos_;
        const int cur_line = line_;
        bool has_dot = false;
        bool has_exp = false;

        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (std::isdigit(static_cast<unsigned char>(c))) {
                ++pos_;
            } else if (c == '.' && !has_dot && !has_exp) {
                has_dot = true;
                ++pos_;
            } else if ((c == 'e' || c == 'E') && !has_exp) {
                has_exp = true;
                ++pos_;
                if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) {
                    ++pos_;
                }
            } else {
                break;
            }
        }

        const std::string text = src_.substr(start, pos_ - start);
        double val = 0.0;
        try {
            val = std::stod(text);
        } catch (...) {
            val = 0.0;
        }
        return {TokenType::kNumber, text, val, cur_line};
    }

    Token scan_ident() {
        const size_t start = pos_;
        const int cur_line = line_;

        while (pos_ < src_.size()) {
            const char c = src_[pos_];
            if (std::isspace(static_cast<unsigned char>(c)) ||
                c == ':' || c == '+' || c == '-' || c == '<' || c == '>' ||
                c == '=' || c == '[' || c == ']' || c == '*' || c == '/' ||
                c == '^' || c == '\\') {
                break;
            }
            ++pos_;
        }

        const std::string text = src_.substr(start, pos_ - start);
        return {TokenType::kIdent, text, 0.0, cur_line};
    }

    std::string src_;
    size_t pos_;
    int line_;
};

enum class LpSection {
    kStart,
    kObjective,
    kConstraints,
    kBounds,
    kGenerals,
    kBinaries,
    kEnd
};

} // namespace

Model read_lp(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open LP file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    LpLexer lexer(buffer.str());
    const std::vector<Token> tokens = lexer.tokenize();

    Model model;
    model.source_path = filepath;
    model.sense = ObjSense::kMinimize;

    size_t cursor = 0;
    auto peek = [&](size_t offset = 0) -> const Token& {
        if (cursor + offset < tokens.size()) return tokens[cursor + offset];
        return tokens.back();
    };
    auto consume = [&]() -> const Token& {
        const Token& t = peek();
        if (cursor < tokens.size()) ++cursor;
        return t;
    };

    // Tracking
    std::vector<std::string> col_names;
    std::unordered_map<std::string, int> col_name_to_idx;
    auto get_or_create_col = [&](const std::string& name) -> int {
        auto it = col_name_to_idx.find(name);
        if (it != col_name_to_idx.end()) return it->second;
        const int idx = static_cast<int>(col_names.size());
        col_names.push_back(name);
        col_name_to_idx[name] = idx;
        return idx;
    };

    std::vector<std::string> row_names;
    std::unordered_map<std::string, int> row_name_to_idx;
    auto create_row = [&](const std::string& name) -> int {
        const int idx = static_cast<int>(row_names.size());
        row_names.push_back(name);
        row_name_to_idx[name] = idx;
        return idx;
    };

    std::vector<la::Triplet> A_triplets;
    std::vector<la::Triplet> Q_triplets;
    std::vector<double> row_lhs;
    std::vector<double> row_rhs;

    std::unordered_map<int, double> col_lb_map;
    std::unordered_map<int, double> col_ub_map;
    std::unordered_map<int, VarType> col_type_map;

    LpSection section = LpSection::kStart;

    auto check_section_header = [&](const Token& tok, LpSection& next_sec) -> bool {
        if (tok.type != TokenType::kIdent) return false;
        const std::string u = to_upper(tok.text);
        if (u == "MAXIMIZE" || u == "MAXIMUM" || u == "MAX") {
            model.sense = ObjSense::kMaximize;
            next_sec = LpSection::kObjective;
            return true;
        }
        if (u == "MINIMIZE" || u == "MINIMUM" || u == "MIN") {
            model.sense = ObjSense::kMinimize;
            next_sec = LpSection::kObjective;
            return true;
        }
        if (u == "SUBJECT" || u == "SUCH") {
            // Check for "TO" or "THAT"
            next_sec = LpSection::kConstraints;
            return true;
        }
        if (u == "ST" || u == "S.T.") {
            next_sec = LpSection::kConstraints;
            return true;
        }
        if (u == "BOUNDS" || u == "BOUND") {
            next_sec = LpSection::kBounds;
            return true;
        }
        if (u == "GENERALS" || u == "GENERAL" || u == "GEN" || u == "INTEGERS" || u == "INTEGER" || u == "INT") {
            next_sec = LpSection::kGenerals;
            return true;
        }
        if (u == "BINARY" || u == "BINARIES" || u == "BIN") {
            next_sec = LpSection::kBinaries;
            return true;
        }
        if (u == "END") {
            next_sec = LpSection::kEnd;
            return true;
        }
        return false;
    };

    while (cursor < tokens.size() && peek().type != TokenType::kEof) {
        LpSection detected_sec = section;
        if (check_section_header(peek(), detected_sec)) {
            consume();
            if (detected_sec == LpSection::kConstraints) {
                // If "SUBJECT TO", consume "TO" if present
                if (peek().type == TokenType::kIdent && to_upper(peek().text) == "TO") {
                    consume();
                }
            }
            section = detected_sec;
            if (section == LpSection::kEnd) break;
            continue;
        }

        switch (section) {
            case LpSection::kStart: {
                // If not section header, consume
                consume();
                break;
            }

            case LpSection::kObjective: {
                // Check if optional objective name, e.g. "obj:" or "margin:"
                if (peek().type == TokenType::kIdent && peek(1).type == TokenType::kColon) {
                    model.name = peek().text;
                    consume(); // name
                    consume(); // ':'
                }

                // Parse objective expression terms until next section
                while (cursor < tokens.size() && peek().type != TokenType::kEof) {
                    LpSection next_sec;
                    if (check_section_header(peek(), next_sec)) break;

                    // Handle quadratic bracket: [ ... ] / 2
                    if (peek().type == TokenType::kLBracket) {
                        consume(); // '['
                        double q_sign = 1.0;
                        while (cursor < tokens.size() && peek().type != TokenType::kRBracket && peek().type != TokenType::kEof) {
                            if (peek().type == TokenType::kPlus) {
                                q_sign = 1.0;
                                consume();
                                continue;
                            }
                            if (peek().type == TokenType::kMinus) {
                                q_sign = -1.0;
                                consume();
                                continue;
                            }

                            double q_coeff = 1.0;
                            if (peek().type == TokenType::kNumber) {
                                q_coeff = consume().num_val;
                            }

                            if (peek().type == TokenType::kIdent) {
                                const std::string var1 = consume().text;
                                const int idx1 = get_or_create_col(var1);

                                if (peek().type == TokenType::kCaret) {
                                    consume(); // '^'
                                    if (peek().type == TokenType::kNumber) consume(); // '2'
                                    Q_triplets.push_back({idx1, idx1, q_sign * q_coeff});
                                } else if (peek().type == TokenType::kAsterisk) {
                                    consume(); // '*'
                                    if (peek().type == TokenType::kIdent) {
                                        const std::string var2 = consume().text;
                                        const int idx2 = get_or_create_col(var2);
                                        int r = idx1, c = idx2;
                                        if (r < c) std::swap(r, c);
                                        Q_triplets.push_back({r, c, q_sign * q_coeff});
                                    }
                                } else {
                                    Q_triplets.push_back({idx1, idx1, q_sign * q_coeff});
                                }
                            } else {
                                consume();
                            }
                        }

                        if (peek().type == TokenType::kRBracket) {
                            consume(); // ']'
                        }
                        if (peek().type == TokenType::kSlash) {
                            consume(); // '/'
                            if (peek().type == TokenType::kNumber) consume(); // '2'
                        }
                        continue;
                    }

                    // Linear objective term
                    double sign = 1.0;
                    if (peek().type == TokenType::kPlus) {
                        sign = 1.0;
                        consume();
                    } else if (peek().type == TokenType::kMinus) {
                        sign = -1.0;
                        consume();
                    }

                    double coeff = 1.0;
                    if (peek().type == TokenType::kNumber) {
                        coeff = consume().num_val;
                    }

                    if (peek().type == TokenType::kIdent) {
                        // Check if this ident is actually the start of the next section!
                        if (check_section_header(peek(), next_sec)) {
                            break;
                        }
                        const std::string var_name = consume().text;
                        const int col_idx = get_or_create_col(var_name);
                        if (static_cast<size_t>(col_idx) >= model.c.size()) {
                            model.c.resize(static_cast<size_t>(col_idx + 1), 0.0);
                        }
                        model.c[static_cast<size_t>(col_idx)] += sign * coeff;
                    } else {
                        // Unexpected token in objective, skip
                        if (peek().type != TokenType::kEof) consume();
                    }
                }
                break;
            }

            case LpSection::kConstraints: {
                // Check if constraint name present, e.g. "THRUPUT:"
                std::string cname;
                if (peek().type == TokenType::kIdent && peek(1).type == TokenType::kColon) {
                    cname = peek().text;
                    consume(); // name
                    consume(); // ':'
                } else {
                    cname = "R" + std::to_string(row_names.size());
                }

                // Check for range constraint LHS: e.g. "90 <= ..."
                double lhs_val = -1e20;
                bool has_lhs = false;
                TokenType lhs_op = TokenType::kLe;

                // Peek if number followed by <=, <, >=, >, =
                size_t lookahead = 0;
                double leading_sign = 1.0;
                if (peek(lookahead).type == TokenType::kPlus) {
                    leading_sign = 1.0;
                    lookahead++;
                } else if (peek(lookahead).type == TokenType::kMinus) {
                    leading_sign = -1.0;
                    lookahead++;
                }

                if (peek(lookahead).type == TokenType::kNumber &&
                    (peek(lookahead + 1).type == TokenType::kLe ||
                     peek(lookahead + 1).type == TokenType::kLt ||
                     peek(lookahead + 1).type == TokenType::kGe ||
                     peek(lookahead + 1).type == TokenType::kGt ||
                     peek(lookahead + 1).type == TokenType::kEq)) {
                    // Consume the leading sign if present
                    if (lookahead > 0) consume();
                    lhs_val = leading_sign * consume().num_val;
                    lhs_op = consume().type;
                    has_lhs = true;
                }

                const int row_idx = create_row(cname);

                // Parse terms of linear expression until relational operator
                while (cursor < tokens.size() && peek().type != TokenType::kEof) {
                    LpSection next_sec;
                    if (check_section_header(peek(), next_sec)) break;

                    const TokenType tt = peek().type;
                    if (tt == TokenType::kLe || tt == TokenType::kLt ||
                        tt == TokenType::kGe || tt == TokenType::kGt ||
                        tt == TokenType::kEq) {
                        break;
                    }

                    double term_sign = 1.0;
                    if (peek().type == TokenType::kPlus) {
                        term_sign = 1.0;
                        consume();
                    } else if (peek().type == TokenType::kMinus) {
                        term_sign = -1.0;
                        consume();
                    }

                    double coeff = 1.0;
                    if (peek().type == TokenType::kNumber) {
                        coeff = consume().num_val;
                    }

                    if (peek().type == TokenType::kIdent) {
                        const std::string var = consume().text;
                        const int col_idx = get_or_create_col(var);
                        A_triplets.push_back({row_idx, col_idx, term_sign * coeff});
                    } else {
                        // Skip unrecognized token
                        break;
                    }
                }

                // Expect relational operator and RHS
                double rhs_val = 1e20;
                TokenType rel_op = TokenType::kLe;

                if (peek().type == TokenType::kLe || peek().type == TokenType::kLt ||
                    peek().type == TokenType::kGe || peek().type == TokenType::kGt ||
                    peek().type == TokenType::kEq) {
                    rel_op = consume().type;

                    double rhs_sign = 1.0;
                    if (peek().type == TokenType::kPlus) {
                        rhs_sign = 1.0;
                        consume();
                    } else if (peek().type == TokenType::kMinus) {
                        rhs_sign = -1.0;
                        consume();
                    }

                    if (peek().type == TokenType::kNumber) {
                        rhs_val = rhs_sign * consume().num_val;
                    }
                }

                // Determine row bounds based on LHS, relation, RHS
                double l_bound = -1e20;
                double u_bound = 1e20;

                if (has_lhs) {
                    // LHS <= expr <= RHS
                    if (lhs_op == TokenType::kLe || lhs_op == TokenType::kLt) {
                        l_bound = lhs_val;
                    } else if (lhs_op == TokenType::kGe || lhs_op == TokenType::kGt) {
                        u_bound = lhs_val;
                    }

                    if (rel_op == TokenType::kLe || rel_op == TokenType::kLt) {
                        u_bound = rhs_val;
                    } else if (rel_op == TokenType::kGe || rel_op == TokenType::kGt) {
                        l_bound = rhs_val;
                    }
                } else {
                    if (rel_op == TokenType::kEq) {
                        l_bound = rhs_val;
                        u_bound = rhs_val;
                    } else if (rel_op == TokenType::kLe || rel_op == TokenType::kLt) {
                        l_bound = -1e20;
                        u_bound = rhs_val;
                    } else if (rel_op == TokenType::kGe || rel_op == TokenType::kGt) {
                        l_bound = rhs_val;
                        u_bound = 1e20;
                    }
                }

                row_lhs.push_back(l_bound);
                row_rhs.push_back(u_bound);
                break;
            }

            case LpSection::kBounds: {
                // Forms:
                // 1) var >= num  or  var <= num  or  var = num  or  var free
                // 2) num <= var <= num  or  num <= var
                LpSection next_sec;
                if (check_section_header(peek(), next_sec)) {
                    break;
                }

                // Check if starting with a number (e.g. 10 <= x <= 20)
                double leading_sign = 1.0;
                size_t lookahead = 0;
                if (peek(lookahead).type == TokenType::kPlus) {
                    leading_sign = 1.0;
                    lookahead++;
                } else if (peek(lookahead).type == TokenType::kMinus) {
                    leading_sign = -1.0;
                    lookahead++;
                }

                if (peek(lookahead).type == TokenType::kNumber ||
                    (peek(lookahead).type == TokenType::kIdent && is_infinity_str(peek(lookahead).text))) {
                    if (lookahead > 0) consume();
                    double num1 = 0.0;
                    if (peek().type == TokenType::kNumber) {
                        num1 = leading_sign * consume().num_val;
                    } else {
                        // Infinity
                        const std::string inf_str = consume().text;
                        num1 = (leading_sign < 0.0 || inf_str[0] == '-') ? -1e20 : 1e20;
                    }

                    TokenType op1 = consume().type;
                    if (peek().type == TokenType::kIdent) {
                        const std::string var_name = consume().text;
                        const int col_idx = get_or_create_col(var_name);

                        if (op1 == TokenType::kLe || op1 == TokenType::kLt) {
                            col_lb_map[col_idx] = num1;
                        } else if (op1 == TokenType::kGe || op1 == TokenType::kGt) {
                            col_ub_map[col_idx] = num1;
                        }

                        // Check if followed by second op and number: <= 20
                        if (peek().type == TokenType::kLe || peek().type == TokenType::kLt ||
                            peek().type == TokenType::kGe || peek().type == TokenType::kGt) {
                            TokenType op2 = consume().type;
                            double sign2 = 1.0;
                            if (peek().type == TokenType::kPlus) {
                                sign2 = 1.0;
                                consume();
                            } else if (peek().type == TokenType::kMinus) {
                                sign2 = -1.0;
                                consume();
                            }

                            if (peek().type == TokenType::kNumber) {
                                double num2 = sign2 * consume().num_val;
                                if (op2 == TokenType::kLe || op2 == TokenType::kLt) {
                                    col_ub_map[col_idx] = num2;
                                } else {
                                    col_lb_map[col_idx] = num2;
                                }
                            } else if (peek().type == TokenType::kIdent && is_infinity_str(peek().text)) {
                                double num2 = (sign2 < 0.0) ? -1e20 : 1e20;
                                consume();
                                if (op2 == TokenType::kLe || op2 == TokenType::kLt) {
                                    col_ub_map[col_idx] = num2;
                                } else {
                                    col_lb_map[col_idx] = num2;
                                }
                            }
                        }
                    }
                } else if (peek().type == TokenType::kIdent) {
                    const std::string var_name = consume().text;
                    const int col_idx = get_or_create_col(var_name);

                    if (peek().type == TokenType::kIdent && to_upper(peek().text) == "FREE") {
                        consume(); // "free"
                        col_lb_map[col_idx] = -1e20;
                        col_ub_map[col_idx] = 1e20;
                    } else if (peek().type == TokenType::kGe || peek().type == TokenType::kGt) {
                        consume(); // >=
                        double s = 1.0;
                        if (peek().type == TokenType::kPlus) { s = 1.0; consume(); }
                        else if (peek().type == TokenType::kMinus) { s = -1.0; consume(); }

                        if (peek().type == TokenType::kNumber) {
                            col_lb_map[col_idx] = s * consume().num_val;
                        } else if (peek().type == TokenType::kIdent && is_infinity_str(peek().text)) {
                            col_lb_map[col_idx] = (s < 0.0) ? -1e20 : 1e20;
                            consume();
                        }
                    } else if (peek().type == TokenType::kLe || peek().type == TokenType::kLt) {
                        consume(); // <=
                        double s = 1.0;
                        if (peek().type == TokenType::kPlus) { s = 1.0; consume(); }
                        else if (peek().type == TokenType::kMinus) { s = -1.0; consume(); }

                        if (peek().type == TokenType::kNumber) {
                            col_ub_map[col_idx] = s * consume().num_val;
                        } else if (peek().type == TokenType::kIdent && is_infinity_str(peek().text)) {
                            col_ub_map[col_idx] = (s < 0.0) ? -1e20 : 1e20;
                            consume();
                        }
                    } else if (peek().type == TokenType::kEq) {
                        consume(); // =
                        double s = 1.0;
                        if (peek().type == TokenType::kPlus) { s = 1.0; consume(); }
                        else if (peek().type == TokenType::kMinus) { s = -1.0; consume(); }

                        if (peek().type == TokenType::kNumber) {
                            const double val = s * consume().num_val;
                            col_lb_map[col_idx] = val;
                            col_ub_map[col_idx] = val;
                        }
                    }
                } else {
                    consume();
                }
                break;
            }

            case LpSection::kGenerals: {
                LpSection next_sec;
                if (check_section_header(peek(), next_sec)) break;
                if (peek().type == TokenType::kIdent) {
                    const std::string var_name = consume().text;
                    const int col_idx = get_or_create_col(var_name);
                    col_type_map[col_idx] = VarType::kInteger;
                } else {
                    consume();
                }
                break;
            }

            case LpSection::kBinaries: {
                LpSection next_sec;
                if (check_section_header(peek(), next_sec)) break;
                if (peek().type == TokenType::kIdent) {
                    const std::string var_name = consume().text;
                    const int col_idx = get_or_create_col(var_name);
                    col_type_map[col_idx] = VarType::kInteger;
                    if (col_lb_map.find(col_idx) == col_lb_map.end()) {
                        col_lb_map[col_idx] = 0.0;
                    }
                    if (col_ub_map.find(col_idx) == col_ub_map.end()) {
                        col_ub_map[col_idx] = 1.0;
                    }
                } else {
                    consume();
                }
                break;
            }

            case LpSection::kEnd: {
                cursor = tokens.size();
                break;
            }
        }
    }

    const int m = static_cast<int>(row_names.size());
    const int n = static_cast<int>(col_names.size());

    model.num_rows = m;
    model.num_cols = n;
    model.row_names = std::move(row_names);
    model.col_names = std::move(col_names);

    if (static_cast<int>(model.c.size()) < n) {
        model.c.resize(static_cast<size_t>(n), 0.0);
    }

    model.row_lower = std::move(row_lhs);
    model.row_upper = std::move(row_rhs);

    // Default variable bounds in LP standard: 0 <= x <= +inf
    model.col_lower.assign(static_cast<size_t>(n), 0.0);
    model.col_upper.assign(static_cast<size_t>(n), 1e20);
    model.col_type.assign(static_cast<size_t>(n), VarType::kContinuous);

    for (int j = 0; j < n; ++j) {
        auto lbit = col_lb_map.find(j);
        if (lbit != col_lb_map.end()) {
            model.col_lower[static_cast<size_t>(j)] = lbit->second;
        }

        auto ubit = col_ub_map.find(j);
        if (ubit != col_ub_map.end()) {
            model.col_upper[static_cast<size_t>(j)] = ubit->second;
        }

        auto typeit = col_type_map.find(j);
        if (typeit != col_type_map.end()) {
            model.col_type[static_cast<size_t>(j)] = typeit->second;
        }
    }

    model.A.set_from_triplets(m, n, A_triplets, true);

    if (!Q_triplets.empty()) {
        model.Q.set_from_triplets(n, n, Q_triplets, true);
    }

    return model;
}

} // namespace indus::io
