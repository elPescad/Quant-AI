#ifndef QUBO_HPP
#define QUBO_HPP

// Quadratic Unconstrained Binary Optimisation:
//     minimise  E(x) = sum_i h_i x_i + sum_{i<j} J_ij x_i x_j,   x in {0,1}^n
// This is the problem class quantum annealers solve natively. Here it is solved
// classically with simulated annealing (Metropolis single-bit flips under a cooling
// schedule), plus an exact Gray-code enumerator for small n used to verify SA.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace qubo {

class Qubo {
public:
    explicit Qubo(int n = 0) { resize(n); }

    void resize(int n) {
        n_ = n;
        linear_.assign(n, 0.0);
        coupling_.assign(static_cast<size_t>(n) * n, 0.0);
    }

    int size() const { return n_; }

    void add_linear(int i, double v) { linear_[i] += v; }

    // Adds v * x_i * x_j. For i == j this is linear because x^2 = x for binaries.
    void add_quadratic(int i, int j, double v) {
        if (i == j) {
            linear_[i] += v;
            return;
        }
        coupling_[idx(i, j)] += v;
        coupling_[idx(j, i)] += v;
    }

    double linear(int i) const { return linear_[i]; }
    double coupling(int i, int j) const { return coupling_[idx(i, j)]; }

    double energy(const std::vector<uint8_t>& x) const {
        double e = 0.0;
        for (int i = 0; i < n_; ++i) {
            if (!x[i]) continue;
            e += linear_[i];
            for (int j = i + 1; j < n_; ++j) {
                if (x[j]) e += coupling_[idx(i, j)];
            }
        }
        return e;
    }

    // Largest possible |dE| of a single flip: an upper bound used to set the start temperature
    double max_flip_delta() const {
        double m = 0.0;
        for (int i = 0; i < n_; ++i) {
            double s = std::abs(linear_[i]);
            for (int j = 0; j < n_; ++j) if (j != i) s += std::abs(coupling_[idx(i, j)]);
            m = std::max(m, s);
        }
        return m;
    }

private:
    size_t idx(int i, int j) const { return static_cast<size_t>(i) * n_ + j; }

    int n_ = 0;
    std::vector<double> linear_;
    std::vector<double> coupling_; // symmetric, zero diagonal
};

struct Solution {
    std::vector<uint8_t> x;
    double energy = std::numeric_limits<double>::infinity();
};

struct AnnealParams {
    int sweeps = 300;           // Temperature steps; each step proposes n single-bit flips
    int restarts = 12;          // Independent runs; restart 0 uses the warm start if given
    double t_end_ratio = 1e-6;  // Final temperature as a fraction of the start temperature
    uint64_t seed = 0x5eed;
};

// Simulated annealing with incremental local fields: proposing a flip is O(1),
// accepting it is O(n).
class SimulatedAnnealer {
public:
    explicit SimulatedAnnealer(AnnealParams p = {}) : params_(p), rng_(p.seed) {}

    Solution solve(const Qubo& q, const std::vector<uint8_t>* warm_start = nullptr) {
        const int n = q.size();
        Solution best;
        if (n == 0) {
            best.energy = 0.0;
            return best;
        }

        const double t0 = std::max(q.max_flip_delta(), 1e-12);
        const double t1 = t0 * params_.t_end_ratio;
        const int sweeps = std::max(1, params_.sweeps);
        const double cool = std::pow(t1 / t0, 1.0 / std::max(1, sweeps - 1));

        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::vector<uint8_t> x(n);
        std::vector<double> field(n); // field_i = h_i + sum_{j != i} J_ij x_j

        for (int r = 0; r < std::max(1, params_.restarts); ++r) {
            if (r == 0 && warm_start && static_cast<int>(warm_start->size()) == n) {
                x = *warm_start;
            } else {
                for (int i = 0; i < n; ++i) x[i] = unif(rng_) < 0.5 ? 1 : 0;
            }
            for (int i = 0; i < n; ++i) {
                double f = q.linear(i);
                for (int j = 0; j < n; ++j) if (j != i && x[j]) f += q.coupling(i, j);
                field[i] = f;
            }
            double e = q.energy(x);
            if (e < best.energy) { best.energy = e; best.x = x; }

            double t = t0;
            for (int s = 0; s < sweeps; ++s, t *= cool) {
                for (int i = 0; i < n; ++i) {
                    const double de = x[i] ? -field[i] : field[i];
                    if (de <= 0.0 || unif(rng_) < std::exp(-de / t)) {
                        const double sign = x[i] ? -1.0 : 1.0;
                        x[i] ^= 1;
                        e += de;
                        for (int j = 0; j < n; ++j) {
                            if (j != i) field[j] += sign * q.coupling(j, i);
                        }
                        if (e < best.energy - 1e-15) { best.energy = e; best.x = x; }
                    }
                }
            }
        }
        polish(q, best.x);
        best.energy = q.energy(best.x); // Remove accumulated round-off
        return best;
    }

    // Zero-temperature finish: apply improving 1- and 2-bit flips until none is left.
    // Constraint penalties (e.g. a cardinality slack) create minima that single flips
    // cannot leave, such as swapping which asset holds a position.
    static void polish(const Qubo& q, std::vector<uint8_t>& x) {
        const int n = q.size();
        std::vector<double> field(n);
        auto refresh = [&]() {
            for (int i = 0; i < n; ++i) {
                double f = q.linear(i);
                for (int j = 0; j < n; ++j) if (j != i && x[j]) f += q.coupling(i, j);
                field[i] = f;
            }
        };
        refresh();
        for (bool improved = true; improved;) {
            improved = false;
            for (int i = 0; i < n && !improved; ++i) {
                const double di = x[i] ? -field[i] : field[i];
                if (di < -1e-12) { x[i] ^= 1; improved = true; break; }
                for (int j = i + 1; j < n; ++j) {
                    const double dj = x[j] ? -field[j] : field[j];
                    const double si = x[i] ? -1.0 : 1.0, sj = x[j] ? -1.0 : 1.0;
                    if (di + dj + si * sj * q.coupling(i, j) < -1e-12) {
                        x[i] ^= 1;
                        x[j] ^= 1;
                        improved = true;
                        break;
                    }
                }
            }
            if (improved) refresh();
        }
    }

private:
    AnnealParams params_;
    std::mt19937_64 rng_;
};

// Exact minimum by enumerating all 2^n states in Gray-code order (one flip per step).
inline Solution brute_force(const Qubo& q) {
    const int n = q.size();
    if (n > 26) throw std::invalid_argument("brute_force: n too large");
    Solution best;
    std::vector<uint8_t> x(n, 0);
    std::vector<double> field(n);
    for (int i = 0; i < n; ++i) field[i] = q.linear(i);
    double e = 0.0;
    best.x = x;
    best.energy = 0.0;

    const uint64_t total = uint64_t{1} << n;
    for (uint64_t k = 1; k < total; ++k) {
        const int i = __builtin_ctzll(k); // Gray code: bit to flip at step k
        const double de = x[i] ? -field[i] : field[i];
        const double sign = x[i] ? -1.0 : 1.0;
        x[i] ^= 1;
        e += de;
        for (int j = 0; j < n; ++j) if (j != i) field[j] += sign * q.coupling(j, i);
        if (e < best.energy) { best.energy = e; best.x = x; }
    }
    best.energy = q.energy(best.x);
    return best;
}

} // namespace qubo

#endif // QUBO_HPP
