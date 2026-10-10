#ifndef MARKET_DATA_HPP
#define MARKET_DATA_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// CSV schema written by python/bar_schema.py and python/generate_ticks.py. An optional 9th
// column, quoted_spread, carries the measured bid-ask spread in $ (-1 or absent: unknown).
inline constexpr const char* kTickCsvHeader =
    "timestamp,ticker,raw_price,raw_spread,raw_ofi,raw_delta,raw_vol,target";

struct MarketTick {
    int64_t timestamp = 0; // Bar time, unix seconds. Rows sharing a timestamp form one bar.
    int id = 0;            // Row number in the file
    char ticker[8] = {0};
    float raw_price = 0.0f;
    float raw_spread = 0.0f;
    float raw_ofi = 0.0f;
    float raw_delta = 0.0f;
    float raw_vol = 0.0f;
    float quoted_spread = -1.0f; // Measured bid-ask spread in $, -1 when unknown
    int8_t target = -1;    // 0 SELL, 1 HOLD, 2 BUY, -1 unknown (last bars of a series)
};

inline bool is_tick_csv_header(const std::string& line) {
    return line.rfind(kTickCsvHeader, 0) == 0;
}

// Parse one data row. Returns false for malformed rows.
inline bool parse_tick_row(const std::string& line, MarketTick& tick) {
    const char* p = line.c_str();
    char* end = nullptr;

    tick.timestamp = std::strtoll(p, &end, 10);
    if (end == p || *end != ',') return false;
    p = end + 1;

    const char* comma = std::strchr(p, ',');
    if (!comma) return false;
    size_t len = static_cast<size_t>(comma - p);
    if (len == 0 || len >= sizeof(tick.ticker)) return false;
    std::memcpy(tick.ticker, p, len);
    tick.ticker[len] = '\0';
    p = comma + 1;

    float* fields[5] = {&tick.raw_price, &tick.raw_spread, &tick.raw_ofi, &tick.raw_delta, &tick.raw_vol};
    for (float* f : fields) {
        *f = std::strtof(p, &end);
        if (end == p) return false;
        if (*end != ',' && *end != '\0' && *end != '\r') return false;
        p = (*end == ',') ? end + 1 : end;
    }

    long t = std::strtol(p, &end, 10);
    tick.target = (end != p && t >= 0 && t <= 2) ? static_cast<int8_t>(t) : int8_t{-1};
    tick.quoted_spread = -1.0f;
    if (end != p && *end == ',') {
        p = end + 1;
        const float q = std::strtof(p, &end);
        if (end != p && q > 0.0f) tick.quoted_spread = q;
    }
    return true;
}

#endif // MARKET_DATA_HPP
