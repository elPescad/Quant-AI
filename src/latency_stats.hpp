#ifndef LATENCY_STATS_HPP
#define LATENCY_STATS_HPP

// Fixed-memory latency summary: mean plus quantiles from a log-spaced histogram
// (buckets 5% wide, 10 ns to ~3 s; slower samples land in the top bucket). Constant size,
// so a long live run cannot grow it.

#include <array>
#include <cmath>
#include <cstdint>

class LatencyStats {
public:
    void add(double us) {
        sum_ += us;
        count_++;
        int b = us <= kMinUs ? 0 : static_cast<int>(std::log(us / kMinUs) / kLogStep) + 1;
        hist_[b < kBuckets ? b : kBuckets - 1]++;
    }

    uint64_t count() const { return count_; }
    double mean() const { return count_ ? sum_ / count_ : 0.0; }

    // Upper edge of the bucket holding the q-quantile (within 5%)
    double quantile(double q) const {
        if (count_ == 0) return 0.0;
        const uint64_t rank = static_cast<uint64_t>(q * static_cast<double>(count_ - 1)) + 1;
        uint64_t seen = 0;
        for (int b = 0; b < kBuckets; ++b) {
            seen += hist_[b];
            if (seen >= rank) return kMinUs * std::exp(kLogStep * b);
        }
        return kMinUs * std::exp(kLogStep * (kBuckets - 1));
    }

private:
    static constexpr double kMinUs = 0.01;
    static constexpr double kLogStep = 0.04879; // ln(1.05)
    static constexpr int kBuckets = 400;
    std::array<uint64_t, kBuckets> hist_{};
    double sum_ = 0.0;
    uint64_t count_ = 0;
};

#endif
