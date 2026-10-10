#ifndef FEATURE_PIPELINE_HPP
#define FEATURE_PIPELINE_HPP

// Per-ticker feature pipeline shared by the trading engine, model_runner and the tests.
// python/quantum_features.py is the training-side mirror; tests/engine_tests.cpp checks
// that both produce the same numbers (tests/fixtures/feature_parity.csv).
//
//   raw tick -> EWMA z-scores (spread, ofi, delta, vol) + RLS AR(1) phi of OFI
//            -> [optional] ZZ quantum feature map lift (5 qubits -> 62 Pauli expectations)
//            -> rolling window [seq_len, input_dim] for the GRU

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "ar_estimator.hpp"
#include "market_data.hpp"
#include "quantum_sim.hpp"

inline constexpr int kNumClassicalFeatures = 5; // spread, ofi, delta, vol, ar_phi

struct FeatureConfig {
    int feature_version = 2;
    int seq_len = 30;
    int ewma_span = 500;
    int min_warmup = 30;
    double ar_forgetting = 0.995;
    int quantum_lift = 1;
    int n_qubits = kNumClassicalFeatures;
    int reps = 2;
    double bandwidth = 0.5;
    int label_horizon = 6;
    double label_hurdle = 0.0004; // 6-bar return beyond +/- this = BUY / SELL (python/bar_schema.py)
    int64_t val_start_ts = 0;  // Written by train_and_export.py: first timestamp NOT used for training
    int64_t test_start_ts = 0; // First timestamp of the held-out test period

    int input_dim() const {
        return kNumClassicalFeatures + (quantum_lift ? 2 * ((1 << n_qubits) - 1) : 0);
    }

    // key=value file written by python/train_and_export.py next to the model
    static FeatureConfig load(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("Missing feature config '" + path +
                                     "'. Re-run python/train_and_export.py to export the model and its config.");
        }
        std::map<std::string, std::string> kv;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            kv[line.substr(0, eq)] = line.substr(eq + 1);
        }

        FeatureConfig c;
        auto get_i = [&](const char* k, int& v) { if (kv.count(k)) v = std::stoi(kv[k]); };
        auto get_l = [&](const char* k, int64_t& v) { if (kv.count(k)) v = std::stoll(kv[k]); };
        auto get_d = [&](const char* k, double& v) { if (kv.count(k)) v = std::stod(kv[k]); };
        get_i("feature_version", c.feature_version);
        get_i("seq_len", c.seq_len);
        get_i("ewma_span", c.ewma_span);
        get_i("min_warmup", c.min_warmup);
        get_d("ar_forgetting", c.ar_forgetting);
        get_i("quantum_lift", c.quantum_lift);
        get_i("n_qubits", c.n_qubits);
        get_i("reps", c.reps);
        get_d("bandwidth", c.bandwidth);
        get_i("label_horizon", c.label_horizon);
        get_d("label_hurdle", c.label_hurdle);
        get_l("val_start_ts", c.val_start_ts);
        get_l("test_start_ts", c.test_start_ts);

        if (c.feature_version != 2) throw std::runtime_error("Unsupported feature_version in " + path);
        if (c.n_qubits != kNumClassicalFeatures) throw std::runtime_error("n_qubits must equal the 5 classical features");
        if (kv.count("input_dim") && std::stoi(kv["input_dim"]) != c.input_dim()) {
            throw std::runtime_error("input_dim in " + path + " does not match the configured pipeline");
        }
        return c;
    }
};

// EWMA z-score. Updates the moments with x first, then standardises x (matches training).
class EWMStandardizer {
public:
    explicit EWMStandardizer(int span = 500) : alpha_(2.0 / (span + 1.0)) {}

    float normalize_and_update(double x) {
        count_++;
        if (!initialized_) {
            mean_ = x;
            variance_ = 0.0;
            initialized_ = true;
            return 0.0f;
        }
        const double delta = x - mean_;
        mean_ += alpha_ * delta;
        variance_ = (1.0 - alpha_) * (variance_ + alpha_ * delta * delta);

        const double stddev = std::sqrt(variance_);
        if (stddev > 1e-6 && count_ >= 10) {
            return static_cast<float>(std::clamp((x - mean_) / stddev, -4.0, 4.0));
        }
        return 0.0f;
    }

    uint64_t count() const { return count_; }

private:
    double alpha_;
    double mean_ = 0.0;
    double variance_ = 0.0;
    bool initialized_ = false;
    uint64_t count_ = 0;
};

// Chronological ring of feature vectors for the GRU window
class RollingWindow {
public:
    RollingWindow(int seq_len, int dim) : seq_len_(seq_len), dim_(dim), data_(static_cast<size_t>(seq_len) * dim, 0.0f) {}

    void push(const float* row) {
        std::copy(row, row + dim_, data_.begin() + static_cast<size_t>(head_) * dim_);
        head_ = (head_ + 1) % seq_len_;
        if (count_ < seq_len_) count_++;
    }

    bool full() const { return count_ == seq_len_; }

    // Oldest -> newest into a [seq_len, dim] buffer
    void copy_to(float* dst) const {
        int idx = full() ? head_ : 0;
        for (int i = 0; i < count_; ++i) {
            std::copy_n(data_.begin() + static_cast<size_t>(idx) * dim_, dim_, dst + static_cast<size_t>(i) * dim_);
            idx = (idx + 1) % seq_len_;
        }
    }

private:
    int seq_len_, dim_;
    std::vector<float> data_;
    int head_ = 0;
    int count_ = 0;
};

class TickerFeaturePipeline {
public:
    explicit TickerFeaturePipeline(const FeatureConfig& cfg)
        : cfg_(cfg),
          spread_(cfg.ewma_span), ofi_(cfg.ewma_span), delta_(cfg.ewma_span), vol_(cfg.ewma_span),
          ar_(cfg.ar_forgetting),
          qmap_(cfg.n_qubits, cfg.reps, cfg.bandwidth),
          window_(cfg.seq_len, cfg.input_dim()),
          row_(cfg.input_dim()),
          lifted_(cfg.quantum_lift ? qmap_.output_dim() : 0) {}

    // Returns true once the window holds a full sequence the model may be run on
    bool update(const MarketTick& t) {
        double classical[kNumClassicalFeatures];
        compute_row(t, classical, row_.data());
        window_.push(row_.data());
        count_++;
        return ready();
    }

    bool ready() const { return count_ >= static_cast<uint64_t>(cfg_.min_warmup) && window_.full(); }
    void copy_window(float* dst) const { window_.copy_to(dst); }
    const std::vector<float>& latest_row() const { return row_; }
    const ArRlsEstimator& ar() const { return ar_; }

    // AR coefficient as fed to the model: 0 during warm-up, scaled x2 so its range
    // is comparable to the z-scored inputs before quantum angle encoding
    static double phi_feature(const ArRlsEstimator& ar) {
        if (ar.count() < 30) return 0.0;
        return 2.0 * std::clamp(ar.phi(), -1.0, 1.0);
    }

private:
    void compute_row(const MarketTick& t, double* classical, float* out) {
        classical[0] = spread_.normalize_and_update(t.raw_spread);
        classical[1] = ofi_.normalize_and_update(t.raw_ofi);
        classical[2] = delta_.normalize_and_update(t.raw_delta);
        classical[3] = vol_.normalize_and_update(t.raw_vol);
        ar_.update(static_cast<double>(t.raw_ofi));
        classical[4] = phi_feature(ar_);

        for (int i = 0; i < kNumClassicalFeatures; ++i) out[i] = static_cast<float>(classical[i]);
        if (cfg_.quantum_lift) {
            qmap_.lift(classical, lifted_.data());
            for (size_t i = 0; i < lifted_.size(); ++i) out[kNumClassicalFeatures + i] = static_cast<float>(lifted_[i]);
        }
    }

    FeatureConfig cfg_;
    EWMStandardizer spread_, ofi_, delta_, vol_;
    ArRlsEstimator ar_;
    quantum::ZZFeatureMap qmap_;
    RollingWindow window_;
    std::vector<float> row_;
    std::vector<double> lifted_;
    uint64_t count_ = 0;
};

#endif // FEATURE_PIPELINE_HPP
