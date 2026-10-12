#ifndef CONTEXT_FEATURES_HPP
#define CONTEXT_FEATURES_HPP

// Market context inputs (feature_version 3): where this bar sits in the session and in the
// last days, in units of the ticker's recent 5-minute volatility. The microstructure inputs
// only see the last bars' range, signed volume and price change; python/signal_lab.py found
// these slower, price-only quantities (time of day, today's move, the 1- and 5-day moves,
// distance from the 20/50-day averages, the volatility ratio) carrying most of the
// out-of-sample information on five years of bars. Price-only, so they mean the same on IEX
// and SIP bars. python/quantum_features.py (ContextFeatures) mirrors this step for step;
// tests/fixtures/feature_parity_v3.csv checks they agree.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>

#include "live_bars.hpp"

inline constexpr int kNumContextFeatures = 8;

class ContextFeatures {
public:
    // Order of the outputs (all clipped to [-4, 4])
    enum { kTimeOfDay, kToday, kGap, kDay1, kDay5, kTrend20, kTrend50, kVolRatio };

    void update(int64_t bar_start, double price, double* out) {
        const double lc = std::log(price);
        const int64_t local = bar_start + live::new_york_offset(bar_start);
        const int64_t day = local >= 0 ? local / 86400 : (local - 86399) / 86400;
        const double minute = static_cast<double>(local - day * 86400) / 60.0;
        if (!has_day_ || day != day_) {
            if (has_day_) {
                closes_.push_back(last_price_);
                if (closes_.size() > 50) closes_.pop_front();
                prev_close_ = lc_prev_;
                has_prev_close_ = true;
            }
            has_day_ = true;
            day_ = day;
            lc_first_ = lc;
            k_ = 0;
        } else {
            k_++;
            const double r = lc - lc_prev_;
            if (n_ret_ == 0) {
                var5_ = r * r;
                var_short_ = r * r;
            } else {
                var5_ = (1.0 - kAlpha5) * var5_ + kAlpha5 * r * r;
                var_short_ = (1.0 - kAlphaShort) * var_short_ + kAlphaShort * r * r;
            }
            n_ret_++;
        }

        for (int i = 0; i < kNumContextFeatures; ++i) out[i] = 0.0;
        out[kTimeOfDay] = 2.0 * std::clamp((minute - 570.0) / 5.0 / 77.0, 0.0, 1.0) - 1.0;
        if (n_ret_ >= 20 && var5_ > 0.0) {
            const double s5 = std::sqrt(var5_);
            const double s_day = s5 * std::sqrt(78.0);
            if (k_ > 0) out[kToday] = (lc - lc_first_) / (s5 * std::sqrt(static_cast<double>(k_)));
            if (has_prev_close_) out[kGap] = (lc_first_ - prev_close_) / s_day;
            if (ring_.size() >= 78) out[kDay1] = (lc - ring_[ring_.size() - 78]) / s_day;
            if (ring_.size() >= 390) out[kDay5] = (lc - ring_[ring_.size() - 390]) / (s5 * std::sqrt(390.0));
            if (closes_.size() >= 20) out[kTrend20] = (lc - std::log(mean_last(20))) / s_day;
            if (closes_.size() >= 50) out[kTrend50] = (lc - std::log(mean_last(50))) / s_day;
            if (var_short_ > 0.0) out[kVolRatio] = 0.5 * std::log(var_short_ / var5_);
        }
        for (int i = 0; i < kNumContextFeatures; ++i) out[i] = std::clamp(out[i], -4.0, 4.0);

        ring_.push_back(lc);
        if (ring_.size() > 390) ring_.pop_front();
        lc_prev_ = lc;
        last_price_ = price;
    }

private:
    static constexpr double kAlpha5 = 2.0 / (390.0 + 1.0);   // ~5 days of 5-minute returns
    static constexpr double kAlphaShort = 2.0 / (12.0 + 1.0); // ~1 hour

    // Average of the last n completed daily closes, summed oldest first (as the Python mirror)
    double mean_last(size_t n) const {
        double s = 0.0;
        for (size_t i = closes_.size() - n; i < closes_.size(); ++i) s += closes_[i];
        return s / static_cast<double>(n);
    }

    bool has_day_ = false, has_prev_close_ = false;
    int64_t day_ = 0;
    int k_ = 0;          // bars since the first bar of today
    long n_ret_ = 0;     // intraday returns seen
    double lc_prev_ = 0.0, lc_first_ = 0.0, prev_close_ = 0.0, last_price_ = 0.0;
    double var5_ = 0.0, var_short_ = 0.0;
    std::deque<double> ring_;   // last 390 log closes, oldest first
    std::deque<double> closes_; // last 50 completed daily closes (prices), oldest first
};

#endif
