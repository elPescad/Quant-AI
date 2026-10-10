#ifndef LIVE_BARS_HPP
#define LIVE_BARS_HPP

// Network-free pieces of the live feed:
//   * RFC 3339 timestamps -> unix seconds, New York session test (US DST rules)
//   * 1-minute bars -> 5-minute bars (the training bar size)
//   * per-bar features computed exactly like python/bar_schema.py, so live ticks match
//     the CSV rows the model was trained on

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "market_data.hpp"

namespace live {

constexpr int64_t kBarSeconds = 300;

// Days since 1970-01-01 for a proleptic Gregorian date (H. Hinnant's algorithm)
inline int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

inline int weekday(int64_t days) { return static_cast<int>((days % 7 + 11) % 7); } // 0 = Sunday

// "2026-10-12T13:35:00Z", "2026-10-12T13:35:00.123456789Z", "2026-10-12T09:30:00-04:00"
inline std::optional<int64_t> parse_rfc3339(const std::string& s) {
    int y, mo, d, h, mi, se;
    if (s.size() < 19 || std::sscanf(s.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d", &y, &mo, &d, &h, &mi, &se) != 6) return std::nullopt;
    size_t i = 19;
    if (i < s.size() && s[i] == '.') {
        ++i;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    int64_t offset = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) {
        int oh = 0, om = 0;
        if (std::sscanf(s.c_str() + i + 1, "%2d:%2d", &oh, &om) != 2) return std::nullopt;
        offset = (s[i] == '+' ? 1 : -1) * (oh * 3600 + om * 60);
    } else if (i >= s.size() || (s[i] != 'Z' && s[i] != 'z')) {
        return std::nullopt;
    }
    return days_from_civil(y, mo, d) * 86400 + h * 3600 + mi * 60 + se - offset;
}

inline std::string to_rfc3339(int64_t t) {
    const int64_t days = t >= 0 ? t / 86400 : (t - 86399) / 86400;
    const int64_t secs = t - days * 86400;
    // civil_from_days
    const int64_t z = days + 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned d = doy - (153 * mp + 2) / 5 + 1;
    const unsigned m = mp < 10 ? mp + 3 : mp - 9;
    const int64_t y = static_cast<int64_t>(yoe) + era * 400 + (m <= 2);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04lld-%02u-%02uT%02lld:%02lld:%02lldZ", static_cast<long long>(y), m, d,
                  static_cast<long long>(secs / 3600), static_cast<long long>(secs % 3600 / 60), static_cast<long long>(secs % 60));
    return buf;
}

// New York UTC offset (seconds): EDT from the 2nd Sunday of March 02:00 local to the
// 1st Sunday of November 02:00 local (rule in force since 2007), EST otherwise.
inline int64_t new_york_offset(int64_t utc) {
    const int64_t days = utc >= 0 ? utc / 86400 : (utc - 86399) / 86400;
    // Year of this UTC instant
    int64_t y = 1970 + days / 366;
    while (days_from_civil(y + 1, 1, 1) <= days) ++y;
    while (days_from_civil(y, 1, 1) > days) --y;
    auto nth_sunday = [&](unsigned month, int n) {
        const int64_t first = days_from_civil(y, month, 1);
        return first + (7 - weekday(first)) % 7 + 7 * (n - 1);
    };
    const int64_t dst_start = nth_sunday(3, 2) * 86400 + 7 * 3600; // 02:00 EST = 07:00 UTC
    const int64_t dst_end = nth_sunday(11, 1) * 86400 + 6 * 3600;  // 02:00 EDT = 06:00 UTC
    return (utc >= dst_start && utc < dst_end) ? -4 * 3600 : -5 * 3600;
}

// Same rule as python/bar_schema.py: bar starts inside 09:30-16:00 New York time, Mon-Fri
inline bool in_regular_session(int64_t bar_start_utc) {
    const int64_t local = bar_start_utc + new_york_offset(bar_start_utc);
    const int64_t days = local >= 0 ? local / 86400 : (local - 86399) / 86400;
    const int wd = weekday(days);
    const int64_t minute = (local - days * 86400) / 60;
    return wd >= 1 && wd <= 5 && minute >= 9 * 60 + 30 && minute < 16 * 60;
}

struct Bar {
    std::string symbol;
    int64_t start = 0; // unix seconds, start of the bar
    double open = 0, high = 0, low = 0, close = 0, volume = 0;
};

// Per-ticker feature state: python/bar_schema.py bars_to_ticks, one bar at a time
class FeatureState {
public:
    MarketTick make_tick(const Bar& b, float quoted_spread) {
        double& prev = prev_close_[b.symbol];
        const double delta = prev > 0.0 ? b.close - prev : 0.0; // np.diff(prepend=close[0])
        prev = b.close;
        MarketTick t;
        t.timestamp = b.start;
        std::strncpy(t.ticker, b.symbol.c_str(), sizeof(t.ticker) - 1);
        t.raw_price = static_cast<float>(b.close);
        t.raw_spread = static_cast<float>(std::clamp(b.high - b.low, 0.01, 2.0));
        const double sign = delta > 0 ? 1.0 : (delta < 0 ? -1.0 : 0.0);
        t.raw_ofi = static_cast<float>(sign * std::log1p(b.volume));
        t.raw_delta = static_cast<float>(delta);
        t.raw_vol = static_cast<float>(std::abs(delta));
        t.target = -1;
        t.quoted_spread = quoted_spread;
        return t;
    }

private:
    std::map<std::string, double> prev_close_;
};

// Builds 5-minute bars from 1-minute bars. A 5-minute bar is complete when every symbol
// has delivered its last minute, when a minute of a later bar arrives, or (in real time)
// when the caller's clock passes the bar's end plus a grace period.
class FiveMinuteAggregator {
public:
    explicit FiveMinuteAggregator(std::vector<std::string> symbols) : symbols_(std::move(symbols)) {}

    // Returns the bars completed by this minute bar (in symbol order), possibly none
    std::vector<Bar> add_minute(const Bar& m) {
        std::vector<Bar> done;
        const int64_t bucket = m.start - ((m.start % kBarSeconds) + kBarSeconds) % kBarSeconds;
        if (emitted_through_ != kNone && bucket <= emitted_through_) return done; // already delivered
        if (bucket_ != kNone && bucket < bucket_) return done;
        if (bucket_ != kNone && bucket > bucket_) done = flush();
        bucket_ = bucket;
        auto [it, fresh] = current_.try_emplace(m.symbol, m);
        Bar& b = it->second;
        if (fresh) {
            b.start = bucket;
        } else {
            b.high = std::max(b.high, m.high);
            b.low = std::min(b.low, m.low);
            b.close = m.close;
            b.volume += m.volume;
        }
        if (m.start == bucket + kBarSeconds - 60) last_minute_seen_[m.symbol] = true;
        bool all_last = true;
        for (const auto& s : symbols_) all_last &= last_minute_seen_.count(s) > 0;
        if (all_last) {
            auto more = flush();
            done.insert(done.end(), more.begin(), more.end());
        }
        return done;
    }

    // Emit the open bar if `now` is past its end + grace (real-time use)
    std::vector<Bar> on_clock(int64_t now, int64_t grace) {
        if (bucket_ != kNone && !current_.empty() && now >= bucket_ + kBarSeconds + grace) return flush();
        return {};
    }

    std::vector<Bar> flush() {
        std::vector<Bar> out;
        for (const auto& s : symbols_) {
            auto it = current_.find(s);
            if (it != current_.end()) out.push_back(it->second);
        }
        current_.clear();
        last_minute_seen_.clear();
        if (bucket_ != kNone && !out.empty()) emitted_through_ = std::max(emitted_through_, bucket_);
        return out;
    }

    // Start of the newest bar handed out, so a reconnect does not repeat it
    int64_t emitted_through() const { return emitted_through_; }
    void skip_through(int64_t bucket) {
        emitted_through_ = std::max(emitted_through_, bucket);
        if (bucket_ != kNone && bucket_ <= bucket) { current_.clear(); last_minute_seen_.clear(); }
    }

private:
    static constexpr int64_t kNone = INT64_MIN;
    std::vector<std::string> symbols_;
    int64_t bucket_ = kNone;
    int64_t emitted_through_ = kNone;
    std::map<std::string, Bar> current_;
    std::map<std::string, bool> last_minute_seen_;
};

} // namespace live

#endif
