"""Training-side mirror of src/feature_pipeline.hpp.

Every step here must match the C++ engine operation-for-operation, otherwise the model
is trained on different numbers than it sees live:

    raw tick -> EWMA z-scores (spread, ofi, delta, vol) + RLS AR(1) phi of OFI
             -> [optional] simulated ZZ quantum feature map (5 qubits -> 62 Pauli expectations)

`python quantum_features.py` writes tests/fixtures/feature_parity.csv, which
tests/engine_tests.cpp uses to check the C++ pipeline reproduces these numbers.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

N_CLASSICAL = 5  # spread, ofi, delta, vol, ar_phi
CSV_HEADER = ["timestamp", "ticker", "raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol", "target"]


@dataclass
class FeatureConfig:
    feature_version: int = 2
    seq_len: int = 30
    ewma_span: int = 500
    min_warmup: int = 30
    ar_forgetting: float = 0.995
    quantum_lift: int = 1
    n_qubits: int = N_CLASSICAL
    reps: int = 2
    bandwidth: float = 0.5
    label_horizon: int = 6
    label_hurdle: float = 0.0004  # BUY/SELL return threshold; the engine's online learning labels bars with it
    val_start_ts: int = 0
    test_start_ts: int = 0

    @property
    def input_dim(self) -> int:
        return N_CLASSICAL + (2 * (2**self.n_qubits - 1) if self.quantum_lift else 0)

    def write(self, path: Path) -> None:
        lines = [f"{f.name}={getattr(self, f.name)!r}".replace("'", "") for f in fields(self)]
        lines.append(f"input_dim={self.input_dim}")
        Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Classical features
# --------------------------------------------------------------------------------------
class EWMStandardizer:
    """EWMA z-score; updates with x first, then standardises x. Mirrors the C++ class."""

    def __init__(self, span: int = 500):
        self.alpha = 2.0 / (span + 1.0)
        self.mean = 0.0
        self.variance = 0.0
        self.initialized = False
        self.count = 0

    def normalize_and_update(self, x: float) -> float:
        self.count += 1
        if not self.initialized:
            self.mean = x
            self.variance = 0.0
            self.initialized = True
            return 0.0
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * delta * delta)
        stddev = np.sqrt(self.variance)
        if stddev > 1e-6 and self.count >= 10:
            z = min(max((x - self.mean) / stddev, -4.0), 4.0)
            return float(np.float32(z))  # C++ returns float
        return 0.0


class ArRlsEstimator:
    """RLS with exponential forgetting for x_t = c + phi x_{t-1} + e_t. Mirrors src/ar_estimator.hpp."""

    def __init__(self, forgetting: float = 0.995, init_cov: float = 1e3):
        self.lam = forgetting
        self.init_cov = init_cov
        self.c = 0.0
        self.phi = 0.0
        self.p00 = init_cov
        self.p01 = 0.0
        self.p11 = init_cov
        self.sse = 0.0
        self.weight_sum = 0.0
        self.prev = 0.0
        self.has_prev = False
        self.n = 0

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        if not self.has_prev:
            self.prev = x
            self.has_prev = True
            return
        z1 = self.prev
        pz0 = self.p00 + self.p01 * z1
        pz1 = self.p01 + self.p11 * z1
        denom = self.lam + pz0 + z1 * pz1
        k0 = pz0 / denom
        k1 = pz1 / denom

        prior_err = x - (self.c + self.phi * z1)
        self.c += k0 * prior_err
        self.phi += k1 * prior_err
        post_err = x - (self.c + self.phi * z1)

        self.p00 = (self.p00 - k0 * pz0) / self.lam
        self.p01 = (self.p01 - k0 * pz1) / self.lam
        self.p11 = (self.p11 - k1 * pz1) / self.lam

        self.sse = self.lam * self.sse + prior_err * post_err
        self.weight_sum = self.lam * self.weight_sum + 1.0

        if self.p00 + self.p11 > 1e6 or not np.isfinite(self.p00 + self.p11):
            self.p00 = self.init_cov
            self.p01 = 0.0
            self.p11 = self.init_cov

        self.prev = x
        self.n += 1

    def phi_stderr(self) -> float:
        dof = self.weight_sum - 2.0
        var = max(0.0, self.sse) / dof if dof > 1.0 else 0.0
        return float(np.sqrt(max(0.0, var * self.p11)))


def phi_feature(ar: ArRlsEstimator) -> float:
    if ar.n < 30:
        return 0.0
    return 2.0 * min(max(ar.phi, -1.0), 1.0)


def classical_features(spread, ofi, delta, vol, cfg: FeatureConfig) -> np.ndarray:
    """Per-ticker classical features [T, 5] (float64 holding the exact C++ values)."""
    n = len(spread)
    out = np.zeros((n, N_CLASSICAL), dtype=np.float64)
    st = [EWMStandardizer(cfg.ewma_span) for _ in range(4)]
    ar = ArRlsEstimator(cfg.ar_forgetting)
    # C++ parses floats, then widens to double
    cols = [np.asarray(c, dtype=np.float32).astype(np.float64) for c in (spread, ofi, delta, vol)]
    for t in range(n):
        for k in range(4):
            out[t, k] = st[k].normalize_and_update(float(cols[k][t]))
        ar.update(float(cols[1][t]))
        out[t, 4] = phi_feature(ar)
    return out


# --------------------------------------------------------------------------------------
# Batched statevector simulator (one circuit per row). Qubit q is bit q of the index.
# --------------------------------------------------------------------------------------
def _h(state: np.ndarray, q: int, n: int) -> np.ndarray:
    b = state.shape[0]
    s = state.reshape(b, 2 ** (n - q - 1), 2, 2**q)
    a0, a1 = s[:, :, 0, :], s[:, :, 1, :]
    r = 1.0 / np.sqrt(2.0)
    return np.stack([(a0 + a1) * r, (a0 - a1) * r], axis=2).reshape(b, -1)


def _rz(state: np.ndarray, q: int, theta: np.ndarray, n: int) -> np.ndarray:
    bit = (np.arange(2**n) >> q) & 1
    half = 0.5 * theta[:, None]
    angle = np.where(bit[None, :] == 1, half, -half)
    return state * (np.cos(angle) + 1j * np.sin(angle))


def _cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    idx = np.arange(2**n)
    perm = idx ^ (((idx >> control) & 1) << target)
    return state[:, perm]


def _rzz(state: np.ndarray, a: int, b: int, theta: np.ndarray, n: int) -> np.ndarray:
    state = _cnot(state, a, b, n)
    state = _rz(state, b, theta, n)
    return _cnot(state, a, b, n)


def _walsh_hadamard(p: np.ndarray) -> np.ndarray:
    b, dim = p.shape
    a = p.copy()
    length = 1
    while length < dim:
        v = a.reshape(b, dim // (2 * length), 2, length)
        u, w = v[:, :, 0, :], v[:, :, 1, :]
        a = np.stack([u + w, u - w], axis=2).reshape(b, dim)
        length *= 2
    return a


def zz_feature_map(inputs: np.ndarray, reps: int, bandwidth: float) -> np.ndarray:
    """Simulated ZZ feature map circuit + Pauli-Z/X string readout. [B, n] -> [B, 2(2^n - 1)]."""
    bsz, n = inputs.shape
    x = bandwidth * inputs
    state = np.zeros((bsz, 2**n), dtype=np.complex128)
    state[:, 0] = 1.0
    for _ in range(reps):
        for q in range(n):
            state = _h(state, q, n)
        for q in range(n):
            state = _rz(state, q, 2.0 * x[:, q], n)
        for i in range(n):
            for j in range(i + 1, n):
                state = _rzz(state, i, j, 2.0 * (np.pi - x[:, i]) * (np.pi - x[:, j]), n)

    z_strings = _walsh_hadamard(np.abs(state) ** 2)[:, 1:]
    for q in range(n):
        state = _h(state, q, n)
    x_strings = _walsh_hadamard(np.abs(state) ** 2)[:, 1:]
    return np.concatenate([z_strings, x_strings], axis=1)


def model_inputs(classical: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Per-tick model input rows [T, input_dim] float32, exactly as the C++ window holds them."""
    rows = [classical]
    if cfg.quantum_lift:
        rows.append(zz_feature_map(classical, cfg.reps, cfg.bandwidth))
    return np.concatenate(rows, axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------
# Parity fixture for the C++ tests
# --------------------------------------------------------------------------------------
def write_parity_fixture(out_dir: Path, rows_per_ticker: int = 150, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    cfg = FeatureConfig(seq_len=10, min_warmup=30, bandwidth=0.5)
    out_dir.mkdir(parents=True, exist_ok=True)

    tickers = ["AAA", "BBB"]
    raw = {}
    for sym in tickers:
        ofi = np.zeros(rows_per_ticker)
        for t in range(1, rows_per_ticker):
            ofi[t] = 0.8 * ofi[t - 1] + rng.normal()
        price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, rows_per_ticker)))
        delta = np.diff(price, prepend=price[0])
        raw[sym] = {
            "raw_price": price.astype(np.float32),
            "raw_spread": np.abs(rng.normal(0.02, 0.005, rows_per_ticker)).astype(np.float32),
            "raw_ofi": ofi.astype(np.float32),
            "raw_delta": delta.astype(np.float32),
            "raw_vol": np.abs(delta).astype(np.float32),
        }
        c = raw[sym]
        raw[sym]["features"] = model_inputs(
            classical_features(c["raw_spread"], c["raw_ofi"], c["raw_delta"], c["raw_vol"], cfg), cfg
        )

    with open(out_dir / "feature_parity.csv", "w") as f:
        f.write(",".join(CSV_HEADER + [f"f{k}" for k in range(cfg.input_dim)]) + "\n")
        for t in range(rows_per_ticker):
            for sym in tickers:
                c = raw[sym]
                # str(np.float32) is the shortest text that parses back to the same float
                vals = [str(1700000000 + 300 * t), sym] + [
                    str(c[k][t]) for k in ["raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol"]
                ]
                vals.append("1")
                vals += [repr(float(v)) for v in c["features"][t]]
                f.write(",".join(vals) + "\n")
    cfg.write(out_dir / "feature_parity_config.txt")
    print(f"[+] Wrote parity fixture to {out_dir}")


if __name__ == "__main__":
    write_parity_fixture(Path(__file__).resolve().parent.parent / "tests" / "fixtures")
