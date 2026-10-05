#ifndef QUBO_ALLOCATOR_HPP
#define QUBO_ALLOCATOR_HPP

// Cross-sectional position selection as a QUBO.
//
// Each bar, every ticker with a fresh model signal gets two binary variables:
//     L_i = 1 -> hold long,  S_i = 1 -> hold short,  both 0 -> flat.
// Positions are equal-weight (w = position_weight of equity), so the
// mean-variance objective per unit of w is
//     maximise  sum_i mu_i d_i  -  costs  -  (gamma/2) * w * H * d' Sigma d,   d_i = L_i - S_i
// with
//     mu_i     model edge (p_buy - p_sell) x typical |H-bar move| of ticker i
//     costs    linear in L/S given the current holding (entry, exit, flip)
//     Sigma    EWMA covariance of 1-bar returns, scaled to the H-bar horizon
// Constraints become penalties:
//     P * L_i * S_i                                    never long and short at once
//     A * (sum_i (L_i + S_i) + sum_k 2^k s_k - K)^2    at most K positions (binary slack s)
// The resulting QUBO is solved by simulated annealing (qubo.hpp).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <vector>

#include "qubo.hpp"

// Exponentially weighted covariance of per-bar returns (zero-mean assumption)
class RiskModel {
public:
    explicit RiskModel(double decay = 0.99) : decay_(decay) {}

    int add_asset() {
        const int old_n = n_;
        std::vector<double> next(static_cast<size_t>(n_ + 1) * (n_ + 1), 0.0);
        for (int i = 0; i < old_n; ++i)
            for (int j = 0; j < old_n; ++j) next[static_cast<size_t>(i) * (n_ + 1) + j] = cov_[static_cast<size_t>(i) * old_n + j];
        cov_.swap(next);
        n_++;
        return n_ - 1;
    }

    int size() const { return n_; }

    // returns[i] is ignored unless valid[i]
    void update(const std::vector<double>& returns, const std::vector<uint8_t>& valid) {
        weight_ = decay_ * weight_ + (1.0 - decay_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                double& c = cov_[static_cast<size_t>(i) * n_ + j];
                const double rr = (valid[i] && valid[j]) ? returns[i] * returns[j] : 0.0;
                c = decay_ * c + (1.0 - decay_) * rr;
            }
        }
        updates_++;
    }

    double cov(int i, int j) const {
        if (weight_ <= 0.0) return 0.0;
        return cov_[static_cast<size_t>(i) * n_ + j] / weight_; // bias-corrected
    }

    bool warm() const { return updates_ >= 30; }

private:
    double decay_;
    int n_ = 0;
    std::vector<double> cov_;
    double weight_ = 0.0;
    uint64_t updates_ = 0;
};

// Typical absolute return over the label horizon, from realised bar-close prices
class HorizonMoveTracker {
public:
    explicit HorizonMoveTracker(int horizon = 6, int span = 200)
        : horizon_(horizon), alpha_(2.0 / (span + 1.0)) {}

    void on_bar_close(double price) {
        if (!(price > 0.0)) return;
        closes_.push_back(price);
        if (static_cast<int>(closes_.size()) > horizon_ + 1) closes_.pop_front();
        if (static_cast<int>(closes_.size()) == horizon_ + 1) {
            const double move = std::abs(closes_.back() / closes_.front() - 1.0);
            mean_abs_move_ = (n_ == 0) ? move : mean_abs_move_ + alpha_ * (move - mean_abs_move_);
            n_++;
        }
    }

    bool ready() const { return n_ >= 20; }
    double typical_move() const { return mean_abs_move_; }

private:
    int horizon_;
    double alpha_;
    std::deque<double> closes_;
    double mean_abs_move_ = 0.0;
    uint64_t n_ = 0;
};

struct AssetView {
    int asset = -1;               // Index into the RiskModel
    double expected_return = 0.0; // Signed, fractional, over the horizon
    double cost = 0.0;            // One-way trading cost as a fraction (half spread + fee)
    int current = 0;              // Current holding: -1, 0, +1
};

struct AllocatorParams {
    int max_positions = 4;
    double risk_aversion = 100.0;
    double position_weight = 0.2;
    int horizon = 6;
    qubo::AnnealParams anneal;
    bool verify_with_brute_force = false;
};

struct AllocatorStats {
    long solves = 0;
    long verified = 0;
    long matched_optimum = 0;
    double max_energy_gap = 0.0;
    long variables_total = 0;
};

class QuboAllocator {
public:
    QuboAllocator(AllocatorParams p, const RiskModel& risk)
        : params_(p), risk_(risk), annealer_(p.anneal) {}

    // Linear cost terms for (L_i, S_i) given the current holding.
    // Opening pays entry + eventual exit (2c); keeping pays nothing; dropping a held
    // position pays exit (c), encoded as a -c bonus for keeping it.
    static void cost_terms(const AssetView& a, double& cost_long, double& cost_short) {
        const double c = a.cost;
        if (a.current > 0) { cost_long = -c; cost_short = 2.0 * c; }
        else if (a.current < 0) { cost_long = 2.0 * c; cost_short = -c; }
        else { cost_long = 2.0 * c; cost_short = 2.0 * c; }
    }

    // Builds the QUBO. Variables: [L_0, S_0, L_1, S_1, ..., slack bits]
    qubo::Qubo build(const std::vector<AssetView>& assets, int capacity, int& n_slack) const {
        const int n = static_cast<int>(assets.size());
        const bool constrained = capacity < n;
        n_slack = 0;
        if (constrained) while (((1 << n_slack) - 1) < capacity) n_slack++;

        // Normalise to O(1) coefficients; the argmin is unchanged
        double unit = 1e-12;
        for (const auto& a : assets) unit = std::max({unit, std::abs(a.expected_return), a.cost});
        const double risk_scale = params_.risk_aversion * params_.position_weight * params_.horizon;

        qubo::Qubo q(2 * n + n_slack);
        double max_gain = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& a = assets[i];
            const int li = 2 * i, si = 2 * i + 1;
            double cl, cs;
            cost_terms(a, cl, cs);
            q.add_linear(li, (-a.expected_return + cl) / unit);
            q.add_linear(si, (a.expected_return + cs) / unit);

            double risk_row = 0.0;
            for (int j = 0; j < n; ++j) {
                const double cov = (a.asset >= 0 && assets[j].asset >= 0) ? risk_.cov(a.asset, assets[j].asset) : 0.0;
                const double r = risk_scale * cov / unit;
                risk_row += std::abs(r);
                if (j == i) {
                    // (L - S)^2 = L + S - 2 L S
                    q.add_linear(li, 0.5 * r);
                    q.add_linear(si, 0.5 * r);
                    q.add_quadratic(li, si, -r);
                } else if (j > i) {
                    const int lj = 2 * j, sj = 2 * j + 1;
                    q.add_quadratic(li, lj, r);
                    q.add_quadratic(si, sj, r);
                    q.add_quadratic(li, sj, -r);
                    q.add_quadratic(si, lj, -r);
                }
            }
            max_gain = std::max(max_gain, (std::abs(a.expected_return) + 3.0 * a.cost) / unit + 2.0 * risk_row);
        }

        const double penalty = 2.0 * max_gain + 1.0;
        for (int i = 0; i < n; ++i) q.add_quadratic(2 * i, 2 * i + 1, penalty);

        if (constrained) {
            // A * (sum_v a_v x_v - K)^2, a_v = 1 for position bits, 2^k for slack bits
            const int nv = 2 * n + n_slack;
            std::vector<double> coef(nv, 1.0);
            for (int k = 0; k < n_slack; ++k) coef[2 * n + k] = static_cast<double>(1 << k);
            const double K = capacity;
            for (int u = 0; u < nv; ++u) {
                q.add_linear(u, penalty * (coef[u] * coef[u] - 2.0 * K * coef[u]));
                for (int v = u + 1; v < nv; ++v) q.add_quadratic(u, v, 2.0 * penalty * coef[u] * coef[v]);
            }
        }
        return q;
    }

    static std::vector<int> decode(const std::vector<uint8_t>& x, int n_assets) {
        std::vector<int> d(n_assets, 0);
        for (int i = 0; i < n_assets; ++i) {
            const bool l = x[2 * i], s = x[2 * i + 1];
            d[i] = (l && !s) ? 1 : (s && !l) ? -1 : 0;
        }
        return d;
    }

    // capacity = how many of these assets may hold a position (max_positions minus
    // positions held by tickers outside this decision)
    std::vector<int> allocate(const std::vector<AssetView>& assets, int capacity) {
        const int n = static_cast<int>(assets.size());
        if (n == 0) return {};
        if (capacity <= 0) return std::vector<int>(n, 0);

        int n_slack = 0;
        qubo::Qubo q = build(assets, capacity, n_slack);

        // Warm start from current holdings, with slack set to the unused capacity
        std::vector<uint8_t> warm(q.size(), 0);
        int held = 0;
        for (int i = 0; i < n; ++i) {
            if (assets[i].current > 0) warm[2 * i] = 1;
            if (assets[i].current < 0) warm[2 * i + 1] = 1;
            held += assets[i].current != 0;
        }
        int spare = std::max(0, capacity - held);
        for (int k = 0; k < n_slack; ++k) warm[2 * n + k] = (spare >> k) & 1;

        qubo::Solution sol = annealer_.solve(q, &warm);
        stats_.solves++;
        stats_.variables_total += q.size();

        if (params_.verify_with_brute_force && q.size() <= 22) {
            qubo::Solution exact = qubo::brute_force(q);
            stats_.verified++;
            const double gap = sol.energy - exact.energy;
            if (gap <= 1e-9 * std::max(1.0, std::abs(exact.energy))) stats_.matched_optimum++;
            stats_.max_energy_gap = std::max(stats_.max_energy_gap, gap);
        }

        std::vector<int> d = decode(sol.x, n);
        enforce_capacity(assets, capacity, d);
        return d;
    }

    // Baseline: best direction per asset by net expected return, keep the top `capacity`.
    // This is the exact QUBO optimum when risk_aversion == 0.
    static std::vector<int> greedy(const std::vector<AssetView>& assets, int capacity) {
        const int n = static_cast<int>(assets.size());
        std::vector<int> d(n, 0);
        std::vector<std::pair<double, int>> scored;
        for (int i = 0; i < n; ++i) {
            double cl, cs;
            cost_terms(assets[i], cl, cs);
            const double vl = assets[i].expected_return - cl;
            const double vs = -assets[i].expected_return - cs;
            if (std::max(vl, vs) > 0.0) {
                d[i] = vl >= vs ? 1 : -1;
                scored.push_back({std::max(vl, vs), i});
            }
        }
        std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (size_t k = static_cast<size_t>(std::max(0, capacity)); k < scored.size(); ++k) d[scored[k].second] = 0;
        return d;
    }

    const AllocatorStats& stats() const { return stats_; }

private:
    // Safety net: SA is heuristic, so never act on a solution that breaks the cap
    static void enforce_capacity(const std::vector<AssetView>& assets, int capacity, std::vector<int>& d) {
        std::vector<std::pair<double, int>> held;
        for (size_t i = 0; i < d.size(); ++i)
            if (d[i] != 0) held.push_back({d[i] * assets[i].expected_return, static_cast<int>(i)});
        if (static_cast<int>(held.size()) <= capacity) return;
        std::sort(held.begin(), held.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (size_t k = static_cast<size_t>(capacity); k < held.size(); ++k) d[held[k].second] = 0;
    }

    AllocatorParams params_;
    const RiskModel& risk_;
    qubo::SimulatedAnnealer annealer_;
    AllocatorStats stats_;
};

#endif // QUBO_ALLOCATOR_HPP
