#ifndef REGIME_HPP
#define REGIME_HPP

// Slow, per-ticker market regime from the engine's own bar closes, for the risk layer:
//   trend  : +1 while the last daily close is above its N-day average, -1 below, 0 unknown
//   vol    : annualised volatility of the last M daily close-to-close returns
// Both are decided from completed New York trading days only and stay fixed through the
// next day (decided on day t's close, held over day t+1), like python/regime_study.py,
// where on 20-90 years of real index data these rules roughly halved drawdowns.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <numeric>

#include "live_bars.hpp"

class DailyRegime {
public:
    DailyRegime(int trend_days = 0, int vol_days = 20) : trend_days_(trend_days), vol_days_(vol_days) {}

    // Every tick of this ticker (bar start time, price). Returns true on the first tick of a
    // new trading day, after the previous day's close has been taken into account.
    bool on_tick(int64_t bar_start, double price) {
        if (!std::isfinite(price) || price <= 0.0) return false;
        const int64_t local = bar_start + live::new_york_offset(bar_start);
        const int64_t day = local >= 0 ? local / 86400 : (local - 86399) / 86400;
        bool new_day = false;
        if (day != day_) {
            if (day_ != kNoDay) close_day(last_price_);
            day_ = day;
            new_day = true;
        }
        last_price_ = price;
        return new_day;
    }

    int trend() const { return trend_; }
    double vol() const { return vol_; } // NaN until vol_days + 1 closes
    int days() const { return static_cast<int>(closes_.size()); }

    // Position size multiplier for a target annual volatility: min(1, target / vol)
    double size_for(double target) const {
        if (target <= 0.0) return 1.0;
        if (!std::isfinite(vol_) || vol_ <= 0.0) return 0.0; // unknown risk: no new position
        return std::min(1.0, target / vol_);
    }

private:
    static constexpr int64_t kNoDay = std::numeric_limits<int64_t>::min();

    void close_day(double close) {
        closes_.push_back(close);
        const size_t keep = static_cast<size_t>(std::max(trend_days_, vol_days_ + 1));
        while (closes_.size() > keep) closes_.pop_front();
        if (trend_days_ > 0 && static_cast<int>(closes_.size()) >= trend_days_) {
            const double avg = std::accumulate(closes_.end() - trend_days_, closes_.end(), 0.0) / trend_days_;
            trend_ = close > avg ? 1 : -1;
        }
        if (static_cast<int>(closes_.size()) >= vol_days_ + 1) {
            double sum = 0.0, sq = 0.0;
            for (size_t i = closes_.size() - vol_days_; i < closes_.size(); ++i) {
                const double r = std::log(closes_[i] / closes_[i - 1]);
                sum += r;
                sq += r * r;
            }
            const double mean = sum / vol_days_;
            vol_ = std::sqrt(std::max(0.0, (sq - vol_days_ * mean * mean) / (vol_days_ - 1)) * 252.0);
        }
    }

    int trend_days_, vol_days_;
    std::deque<double> closes_; // completed days, oldest first
    int64_t day_ = kNoDay;
    double last_price_ = 0.0;
    int trend_ = 0;
    double vol_ = std::numeric_limits<double>::quiet_NaN();
};

#endif
