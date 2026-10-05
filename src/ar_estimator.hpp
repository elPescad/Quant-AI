#ifndef AR_ESTIMATOR_HPP
#define AR_ESTIMATOR_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

// Online AR(1) estimator for   x_t = c + phi * x_{t-1} + e_t
//
// Recursive least squares (RLS) with exponential forgetting factor lambda.
// After every update, theta = [c, phi] is the exact minimiser of
//     sum_k lambda^(t-k) * (x_k - c - phi * x_{k-1})^2
// i.e. an exponentially weighted OLS fit. Compared with the old ratio of
// separately smoothed EWMA moments, this
//   * fits the intercept jointly (no mean-mismatch bias between lag and lead),
//   * has no arbitrary variance prior that takes ~1/alpha ticks to wash out,
//   * gives a standard error for phi (sigma^2 * P[1][1]).
// python/quantum_features.py mirrors this class operation-for-operation.
class ArRlsEstimator {
public:
    explicit ArRlsEstimator(double forgetting = 0.995, double init_cov = 1e3)
        : lambda_(forgetting), init_cov_(init_cov) {
        reset();
    }

    void reset() {
        c_ = 0.0;
        phi_ = 0.0;
        p00_ = init_cov_;
        p01_ = 0.0;
        p11_ = init_cov_;
        sse_ = 0.0;
        weight_sum_ = 0.0;
        prev_ = 0.0;
        has_prev_ = false;
        n_ = 0;
    }

    void update(double x) {
        if (!std::isfinite(x)) return;
        if (!has_prev_) {
            prev_ = x;
            has_prev_ = true;
            return;
        }

        // Regressor z = [1, x_{t-1}]
        const double z1 = prev_;
        const double pz0 = p00_ + p01_ * z1;
        const double pz1 = p01_ + p11_ * z1;
        const double denom = lambda_ + pz0 + z1 * pz1;
        const double k0 = pz0 / denom;
        const double k1 = pz1 / denom;

        const double prior_err = x - (c_ + phi_ * z1);
        c_ += k0 * prior_err;
        phi_ += k1 * prior_err;
        const double post_err = x - (c_ + phi_ * z1);

        // P <- (P - k (Pz)^T) / lambda, kept symmetric
        p00_ = (p00_ - k0 * pz0) / lambda_;
        p01_ = (p01_ - k0 * pz1) / lambda_;
        p11_ = (p11_ - k1 * pz1) / lambda_;

        // Exact recursion for the weighted residual sum of squares
        sse_ = lambda_ * sse_ + prior_err * post_err;
        weight_sum_ = lambda_ * weight_sum_ + 1.0;

        // Without excitation (e.g. a flat OFI series) P grows by 1/lambda each tick.
        // Re-arm it rather than letting the gain explode.
        if (p00_ + p11_ > 1e6 || !std::isfinite(p00_ + p11_)) {
            p00_ = init_cov_;
            p01_ = 0.0;
            p11_ = init_cov_;
        }

        prev_ = x;
        n_++;
    }

    double phi() const { return phi_; }
    double intercept() const { return c_; }
    uint64_t count() const { return n_; }

    // Residual variance with 2 fitted parameters
    double residual_variance() const {
        double dof = weight_sum_ - 2.0;
        return dof > 1.0 ? std::max(0.0, sse_) / dof : 0.0;
    }

    double phi_stderr() const {
        return std::sqrt(std::max(0.0, residual_variance() * p11_));
    }

    // One-step-ahead forecast of x_{t+1}
    double forecast() const { return c_ + phi_ * prev_; }

    // Bars for a shock to decay by half; infinity for |phi| >= 1
    double half_life() const {
        double a = std::abs(phi_);
        if (a <= 0.0) return 0.0;
        if (a >= 1.0) return std::numeric_limits<double>::infinity();
        return std::log(0.5) / std::log(a);
    }

private:
    double lambda_;
    double init_cov_;
    double c_, phi_;
    double p00_, p01_, p11_;
    double sse_, weight_sum_;
    double prev_;
    bool has_prev_;
    uint64_t n_;
};

#endif // AR_ESTIMATOR_HPP
