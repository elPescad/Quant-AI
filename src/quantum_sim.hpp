#ifndef QUANTUM_SIM_HPP
#define QUANTUM_SIM_HPP

// Classical simulation of a small gate-based quantum circuit.
//
// The full 2^n complex amplitude vector is stored and every gate is applied as the
// exact linear operation a quantum processor would perform. This is only feasible
// for a handful of qubits (memory doubles per qubit), which is all the feature map
// needs: 5 qubits = 32 amplitudes per tick.
//
// Convention: basis index b holds qubit q in bit q (little-endian, as in Qiskit).

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace quantum {

using cplx = std::complex<double>;

class StateVector {
public:
    explicit StateVector(int n_qubits)
        : n_(n_qubits), amp_(size_t{1} << n_qubits) {
        if (n_qubits < 1 || n_qubits > 20) throw std::invalid_argument("n_qubits must be in [1, 20]");
        reset();
    }

    // |00...0>
    void reset() {
        std::fill(amp_.begin(), amp_.end(), cplx(0.0, 0.0));
        amp_[0] = cplx(1.0, 0.0);
    }

    int num_qubits() const { return n_; }
    size_t dim() const { return amp_.size(); }
    const std::vector<cplx>& amplitudes() const { return amp_; }
    std::vector<cplx>& amplitudes() { return amp_; }

    // Generic single-qubit unitary [[m00, m01], [m10, m11]] on qubit q
    void apply_1q(int q, cplx m00, cplx m01, cplx m10, cplx m11) {
        const size_t stride = size_t{1} << q;
        const size_t n = amp_.size();
        for (size_t base = 0; base < n; base += 2 * stride) {
            for (size_t off = 0; off < stride; ++off) {
                const size_t i0 = base + off;
                const size_t i1 = i0 + stride;
                const cplx a0 = amp_[i0];
                const cplx a1 = amp_[i1];
                amp_[i0] = m00 * a0 + m01 * a1;
                amp_[i1] = m10 * a0 + m11 * a1;
            }
        }
    }

    // Hadamard: |0> -> (|0>+|1>)/sqrt2, |1> -> (|0>-|1>)/sqrt2
    void h(int q) {
        const size_t stride = size_t{1} << q;
        const size_t n = amp_.size();
        const double s = 1.0 / std::sqrt(2.0);
        for (size_t base = 0; base < n; base += 2 * stride) {
            for (size_t off = 0; off < stride; ++off) {
                const size_t i0 = base + off;
                const size_t i1 = i0 + stride;
                const cplx a0 = amp_[i0];
                const cplx a1 = amp_[i1];
                amp_[i0] = (a0 + a1) * s;
                amp_[i1] = (a0 - a1) * s;
            }
        }
    }

    // Pauli-X (bit flip)
    void x(int q) {
        const size_t mask = size_t{1} << q;
        for (size_t b = 0; b < amp_.size(); ++b) {
            if (!(b & mask)) std::swap(amp_[b], amp_[b | mask]);
        }
    }

    // RZ(theta) = diag(e^{-i theta/2}, e^{+i theta/2})
    void rz(int q, double theta) {
        const size_t mask = size_t{1} << q;
        const cplx phase0 = std::polar(1.0, -0.5 * theta);
        const cplx phase1 = std::polar(1.0, 0.5 * theta);
        for (size_t b = 0; b < amp_.size(); ++b) {
            amp_[b] *= (b & mask) ? phase1 : phase0;
        }
    }

    // RY(theta) = [[cos t/2, -sin t/2], [sin t/2, cos t/2]]
    void ry(int q, double theta) {
        const double c = std::cos(0.5 * theta), s = std::sin(0.5 * theta);
        apply_1q(q, c, -s, s, c);
    }

    // CNOT: flip `target` when `control` is |1>
    void cnot(int control, int target) {
        const size_t cmask = size_t{1} << control;
        const size_t tmask = size_t{1} << target;
        for (size_t b = 0; b < amp_.size(); ++b) {
            if ((b & cmask) && !(b & tmask)) std::swap(amp_[b], amp_[b | tmask]);
        }
    }

    // RZZ(theta) = exp(-i theta/2 Z_a Z_b), built from native gates: CNOT . RZ . CNOT
    void rzz(int a, int b, double theta) {
        cnot(a, b);
        rz(b, theta);
        cnot(a, b);
    }

    double norm_sq() const {
        double s = 0.0;
        for (const auto& a : amp_) s += std::norm(a);
        return s;
    }

    void probabilities(std::vector<double>& out) const {
        out.resize(amp_.size());
        for (size_t b = 0; b < amp_.size(); ++b) out[b] = std::norm(amp_[b]);
    }

private:
    int n_;
    std::vector<cplx> amp_;
};

// In-place unnormalised Walsh-Hadamard transform:
//   a[mask] <- sum_b a[b] * (-1)^popcount(b & mask)
// Applied to a probability vector this yields <Z_S> for every subset S of qubits at once.
inline void walsh_hadamard(std::vector<double>& a) {
    const size_t n = a.size();
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += 2 * len) {
            for (size_t j = i; j < i + len; ++j) {
                const double u = a[j];
                const double v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
        }
    }
}

// ZZ feature map (Havlicek et al., "Supervised learning with quantum-enhanced feature
// spaces", Nature 2019). Each repetition applies
//     H on every qubit
//     RZ(2 x_i) on every qubit
//     RZZ(2 (pi - x_i)(pi - x_j)) on every pair i < j
// so the data is encoded into phases of a 2^n-dimensional state. The readout measures
// every Pauli string Z_S and X_S (S a non-empty subset of qubits), giving a
// 2 * (2^n - 1) dimensional embedding of the n inputs. In that space the classes can
// be separated by a linear readout even when they are not separable in the raw inputs.
class ZZFeatureMap {
public:
    ZZFeatureMap(int n_qubits, int reps, double bandwidth)
        : n_(n_qubits), reps_(reps), bandwidth_(bandwidth), state_(n_qubits) {}

    int num_qubits() const { return n_; }
    int output_dim() const { return 2 * ((1 << n_) - 1); }

    // Prepare U_phi(x) |0...0>
    void encode(const double* inputs, StateVector& sv) const {
        double x[20];
        for (int i = 0; i < n_; ++i) x[i] = bandwidth_ * inputs[i];

        sv.reset();
        for (int r = 0; r < reps_; ++r) {
            for (int q = 0; q < n_; ++q) sv.h(q);
            for (int q = 0; q < n_; ++q) sv.rz(q, 2.0 * x[q]);
            for (int i = 0; i < n_; ++i) {
                for (int j = i + 1; j < n_; ++j) {
                    sv.rzz(i, j, 2.0 * (std::numbers::pi - x[i]) * (std::numbers::pi - x[j]));
                }
            }
        }
    }

    // out[0 .. 2^n-2]           = <Z_S>, S = 1 .. 2^n-1 (bitmask of qubits)
    // out[2^n-1 .. 2(2^n-1)-1]  = <X_S>, same ordering
    void lift(const double* inputs, double* out) {
        encode(inputs, state_);
        const size_t dim = state_.dim();

        state_.probabilities(probs_);
        walsh_hadamard(probs_);
        for (size_t m = 1; m < dim; ++m) out[m - 1] = probs_[m];

        // Measuring in the X basis = rotate with H on every qubit, then measure Z
        for (int q = 0; q < n_; ++q) state_.h(q);
        state_.probabilities(probs_);
        walsh_hadamard(probs_);
        for (size_t m = 1; m < dim; ++m) out[(dim - 1) + (m - 1)] = probs_[m];
    }

private:
    int n_;
    int reps_;
    double bandwidth_;
    StateVector state_;
    std::vector<double> probs_;
};

} // namespace quantum

#endif // QUANTUM_SIM_HPP
