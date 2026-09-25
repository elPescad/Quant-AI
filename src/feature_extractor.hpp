#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include <cmath>
#include <vector>
#include <algorithm>

class DynamicFeatureExtractor {
private:
    double alpha_;           // EWMA smoothing factor
    double mean_ofi_;        // Rolling mean of OFI
    double var_ofi_;         // Rolling variance of OFI
    double cov_ofi_;         // Rolling autocovariance (lag 1)
    double prev_ofi_;        // OFI from previous tick
    bool initialized_;

public:
    DynamicFeatureExtractor(double smoothing_factor = 0.001)
        : alpha_(smoothing_factor),
          mean_ofi_(0.0),
          var_ofi_(1.0),
          cov_ofi_(0.0),
          prev_ofi_(0.0),
          initialized_(false) {}

    // Process tick and return normalized features + dynamic phi
    struct FeatureVector {
        float norm_spread;
        float norm_ofi;
        float norm_delta;
        float norm_vol;
        float dynamic_phi; // Live-calculated AR(1) coefficient
    };

    FeatureVector update(float spread, float raw_ofi, float price_delta, float volatility) {
        if (!initialized_) {
            prev_ofi_ = raw_ofi;
            mean_ofi_ = raw_ofi;
            initialized_ = true;
            return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        // 1. EWMA Mean & Variance Update
        double diff = raw_ofi - mean_ofi_;
        mean_ofi_ += alpha_ * diff;
        var_ofi_ = (1.0 - alpha_) * (var_ofi_ + alpha_ * diff * diff);

        // 2. EWMA Lag-1 Autocovariance Update
        double prev_diff = prev_ofi_ - mean_ofi_;
        cov_ofi_ = (1.0 - alpha_) * cov_ofi_ + alpha_ * (diff * prev_diff);

        // 3. Dynamic AR(1) Coefficient Estimation: Phi = Cov(t, t-1) / Var(t)
        double current_phi = 0.0;
        if (var_ofi_ > 1e-6) {
            current_phi = cov_ofi_ / var_ofi_;
        }
        current_phi = std::clamp(current_phi, -0.95, 0.95);

        prev_ofi_ = raw_ofi;

        // Return normalized feature vector for neural network input
        float norm_spread = (spread - 0.02f) / 0.005f;
        float norm_ofi = static_cast<float>((raw_ofi - mean_ofi_) / std::sqrt(var_ofi_ + 1e-6));
        float norm_delta = price_delta / 0.01f;
        float norm_vol = (volatility - 0.01f) / 0.002f;

        return {norm_spread, norm_ofi, norm_delta, norm_vol, static_cast<float>(current_phi)};
    }
};

#endif // FEATURE_EXTRACTOR_HPP