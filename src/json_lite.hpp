#ifndef JSON_LITE_HPP
#define JSON_LITE_HPP

// Small JSON reader for the Alpaca messages the live feed consumes (objects, arrays,
// strings, numbers, booleans, null). Not a general-purpose library: no writer, and \u
// escapes outside ASCII are replaced by '?', which no field we read contains.

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Json> items;
    std::map<std::string, Json> fields;

    static Json parse(std::string_view text) {
        size_t i = 0;
        Json v = parse_value(text, i);
        skip_ws(text, i);
        if (i != text.size()) throw std::runtime_error("json: trailing characters");
        return v;
    }

    bool is_object() const { return type == Type::Object; }
    bool is_array() const { return type == Type::Array; }
    bool has(const std::string& k) const { return type == Type::Object && fields.count(k) > 0; }
    const Json& operator[](const std::string& k) const {
        static const Json null_value;
        auto it = fields.find(k);
        return it == fields.end() ? null_value : it->second;
    }
    double num(const std::string& k, double fallback = 0.0) const {
        const Json& v = (*this)[k];
        return v.type == Type::Number ? v.number : fallback;
    }
    std::string text(const std::string& k) const {
        const Json& v = (*this)[k];
        return v.type == Type::String ? v.str : std::string();
    }

private:
    static void skip_ws(std::string_view s, size_t& i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    }

    static void expect(std::string_view s, size_t& i, std::string_view word) {
        if (s.substr(i, word.size()) != word) throw std::runtime_error("json: unexpected token");
        i += word.size();
    }

    static std::string parse_string(std::string_view s, size_t& i) {
        if (i >= s.size() || s[i] != '"') throw std::runtime_error("json: expected string");
        ++i;
        std::string out;
        while (i < s.size() && s[i] != '"') {
            char c = s[i++];
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i >= s.size()) break;
            char e = s[i++];
            switch (e) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    if (i + 4 > s.size()) throw std::runtime_error("json: bad \\u escape");
                    const long cp = std::strtol(std::string(s.substr(i, 4)).c_str(), nullptr, 16);
                    out += cp < 0x80 ? static_cast<char>(cp) : '?';
                    i += 4;
                    break;
                }
                default: out += e; // \" \\ \/
            }
        }
        if (i >= s.size()) throw std::runtime_error("json: unterminated string");
        ++i;
        return out;
    }

    static Json parse_value(std::string_view s, size_t& i) {
        skip_ws(s, i);
        if (i >= s.size()) throw std::runtime_error("json: unexpected end");
        Json v;
        const char c = s[i];
        if (c == '{') {
            v.type = Type::Object;
            ++i;
            skip_ws(s, i);
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            while (true) {
                skip_ws(s, i);
                std::string key = parse_string(s, i);
                skip_ws(s, i);
                if (i >= s.size() || s[i] != ':') throw std::runtime_error("json: expected ':'");
                ++i;
                v.fields[std::move(key)] = parse_value(s, i);
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return v; }
                throw std::runtime_error("json: expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.type = Type::Array;
            ++i;
            skip_ws(s, i);
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            while (true) {
                v.items.push_back(parse_value(s, i));
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return v; }
                throw std::runtime_error("json: expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.type = Type::String;
            v.str = parse_string(s, i);
            return v;
        }
        if (c == 't') { expect(s, i, "true"); v.type = Type::Bool; v.boolean = true; return v; }
        if (c == 'f') { expect(s, i, "false"); v.type = Type::Bool; return v; }
        if (c == 'n') { expect(s, i, "null"); return v; }
        const std::string num(s.substr(i, std::min<size_t>(64, s.size() - i)));
        char* end = nullptr;
        v.number = std::strtod(num.c_str(), &end);
        if (end == num.c_str()) throw std::runtime_error("json: bad value");
        v.type = Type::Number;
        i += static_cast<size_t>(end - num.c_str());
        return v;
    }
};

#endif
