"""Training-side mirror of src/feature_pipeline.hpp.

Every step here must match the C++ engine operation-for-operation, otherwise the model
is trained on different numbers than it sees live:

    raw tick -> EWMA z-scores (spread, ofi, delta, vol) + RLS AR(1) phi of OFI
             -> [feature_version 3] 8 market context inputs (src/context_features.hpp)
             -> [optional] simulated ZZ quantum feature map of the 5 (5 qubits -> 62 Pauli expectations)
Row layout: classical | context | quantum (the raw GRU reads the prefix without the quantum part).

`python quantum_features.py` writes tests/fixtures/feature_parity.csv (and _v3.csv), which
tests/engine_tests.cpp uses to check the C++ pipeline reproduces these numbers.
"""

import math
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

N_CLASSICAL = 5  # spread, ofi, delta, vol, ar_phi
N_CONTEXT = 8    # feature_version 3: time of day, today, gap, 1d, 5d, 20d/50d trend, vol ratio
CONTEXT_NAMES = ["time_of_day", "r_today", "gap", "r_1d", "r_5d", "trend_20d", "trend_50d", "vol_ratio"]
CSV_HEADER = ["timestamp", "ticker", "raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol", "target"]


@dataclass
class FeatureConfig:
    feature_version: int = 2  # 3 = with the market context inputs
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
        return N_CLASSICAL + (2 * (2**self.n_qubits - 1) if self.quantum_lift else 0) + (N_CONTEXT if self.has_context else 0)

    @property
    def has_context(self) -> bool:
        return self.feature_version >= 3

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
# Market context (feature_version 3): mirrors src/context_features.hpp operation for operation
# --------------------------------------------------------------------------------------
def _days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def new_york_offset(utc: int) -> int:
    """src/live_bars.hpp new_york_offset: EDT from the 2nd Sunday of March to the 1st Sunday of November."""
    days = utc // 86400
    y = 1970 + days // 366
    while _days_from_civil(y + 1, 1, 1) <= days:
        y += 1
    while _days_from_civil(y, 1, 1) > days:
        y -= 1

    def nth_sunday(month, n):
        first = _days_from_civil(y, month, 1)
        weekday = (first % 7 + 11) % 7  # 0 = Sunday
        return first + (7 - weekday) % 7 + 7 * (n - 1)

    dst_start = nth_sunday(3, 2) * 86400 + 7 * 3600
    dst_end = nth_sunday(11, 1) * 86400 + 6 * 3600
    return -4 * 3600 if dst_start <= utc < dst_end else -5 * 3600


class ContextFeatures:
    ALPHA5 = 2.0 / (390.0 + 1.0)
    ALPHA_SHORT = 2.0 / (12.0 + 1.0)

    def __init__(self):
        self.has_day = self.has_prev_close = False
        self.day = 0
        self.k = 0
        self.n_ret = 0
        self.lc_prev = self.lc_first = self.prev_close = self.last_price = 0.0
        self.var5 = self.var_short = 0.0
        self.ring = deque(maxlen=390)
        self.closes = deque(maxlen=50)

    def _mean_last(self, n):
        s = 0.0
        for v in list(self.closes)[-n:]:  # oldest first, plain left-to-right sum like C++
            s += v
        return s / n

    def update(self, bar_start: int, price: float) -> list:
        lc = math.log(price)
        local = bar_start + new_york_offset(bar_start)
        day = local // 86400
        minute = (local - day * 86400) / 60.0
        if not self.has_day or day != self.day:
            if self.has_day:
                self.closes.append(self.last_price)
                self.prev_close = self.lc_prev
                self.has_prev_close = True
            self.has_day = True
            self.day = day
            self.lc_first = lc
            self.k = 0
        else:
            self.k += 1
            r = lc - self.lc_prev
            if self.n_ret == 0:
                self.var5 = r * r
                self.var_short = r * r
            else:
                self.var5 = (1.0 - self.ALPHA5) * self.var5 + self.ALPHA5 * r * r
                self.var_short = (1.0 - self.ALPHA_SHORT) * self.var_short + self.ALPHA_SHORT * r * r
            self.n_ret += 1

        out = [0.0] * N_CONTEXT
        out[0] = 2.0 * min(max((minute - 570.0) / 5.0 / 77.0, 0.0), 1.0) - 1.0
        if self.n_ret >= 20 and self.var5 > 0.0:
            s5 = math.sqrt(self.var5)
            s_day = s5 * math.sqrt(78.0)
            if self.k > 0:
                out[1] = (lc - self.lc_first) / (s5 * math.sqrt(float(self.k)))
            if self.has_prev_close:
                out[2] = (self.lc_first - self.prev_close) / s_day
            if len(self.ring) >= 78:
                out[3] = (lc - self.ring[-78]) / s_day
            if len(self.ring) >= 390:
                out[4] = (lc - self.ring[-390]) / (s5 * math.sqrt(390.0))
            if len(self.closes) >= 20:
                out[5] = (lc - math.log(self._mean_last(20))) / s_day
            if len(self.closes) >= 50:
                out[6] = (lc - math.log(self._mean_last(50))) / s_day
            if self.var_short > 0.0:
                out[7] = 0.5 * math.log(self.var_short / self.var5)
        out = [min(max(v, -4.0), 4.0) for v in out]
        self.ring.append(lc)
        self.lc_prev = lc
        self.last_price = price
        return out


def context_features(timestamps, prices) -> np.ndarray:
    """Per-ticker context inputs [T, 8] (float64 holding the exact C++ values)."""
    ctx = ContextFeatures()
    p = np.asarray(prices, dtype=np.float32).astype(np.float64)  # C++ parses floats, then widens
    ts = np.asarray(timestamps, dtype=np.int64)
    out = np.zeros((len(p), N_CONTEXT), dtype=np.float64)
    for t in range(len(p)):
        if np.isfinite(p[t]) and p[t] > 0.0:
            out[t] = ctx.update(int(ts[t]), float(p[t]))
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


def model_inputs(classical: np.ndarray, cfg: FeatureConfig, context: np.ndarray = None) -> np.ndarray:
    """Per-tick model input rows [T, input_dim] float32, exactly as the C++ window holds them:
    classical | context (feature_version 3) | quantum lift of the classical five."""
    rows = [classical]
    if cfg.has_context:
        if context is None:
            raise ValueError("feature_version 3 needs the context inputs (context_features)")
        rows.append(context)
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


def write_parity_fixture_v3(out_dir: Path, days: int = 52, bars_per_day: int = 8, seed: int = 11) -> None:
    """feature_version 3 with the quantum lift (row layout classical | context | quantum): session
    timestamps over 52 days (enough for the 50-day trend and the 5-day move), across a DST change."""
    rng = np.random.default_rng(seed)
    cfg = FeatureConfig(feature_version=3, quantum_lift=1, seq_len=10, min_warmup=30)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Weekdays from Monday 2026-02-02; bars at 09:30, 10:10, ... New York time
    stamps = []
    day0 = _days_from_civil(2026, 2, 2)
    d = 0
    while len(stamps) < days * bars_per_day:
        if (d % 7) < 5:
            for b in range(bars_per_day):
                local = (day0 + d) * 86400 + 9 * 3600 + 30 * 60 + b * 40 * 60
                stamps.append(local + 5 * 3600 - (3600 if new_york_offset(local + 5 * 3600) == -4 * 3600 else 0))
        d += 1
    stamps = np.array(stamps[: days * bars_per_day], dtype=np.int64)
    n = len(stamps)
    tickers = ["AAA"]
    raw = {}
    for sym in tickers:
        ofi = np.zeros(n)
        for t in range(1, n):
            ofi[t] = 0.8 * ofi[t - 1] + rng.normal()
        price = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
        delta = np.diff(price, prepend=price[0])
        c = {"raw_price": price.astype(np.float32),
             "raw_spread": np.abs(rng.normal(0.02, 0.005, n)).astype(np.float32),
             "raw_ofi": ofi.astype(np.float32), "raw_delta": delta.astype(np.float32),
             "raw_vol": np.abs(delta).astype(np.float32)}
        c["features"] = model_inputs(classical_features(c["raw_spread"], c["raw_ofi"], c["raw_delta"], c["raw_vol"], cfg), cfg,
                                     context_features(stamps, c["raw_price"]))
        raw[sym] = c
    with open(out_dir / "feature_parity_v3.csv", "w") as f:
        f.write(",".join(CSV_HEADER + [f"f{k}" for k in range(cfg.input_dim)]) + "\n")
        for t in range(n):
            for sym in tickers:
                c = raw[sym]
                vals = [str(int(stamps[t])), sym] + [str(c[k][t]) for k in ["raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol"]]
                vals.append("1")
                vals += [str(v) for v in c["features"][t]]  # float32: shortest text that parses back exactly
                f.write(",".join(vals) + "\n")
    cfg.write(out_dir / "feature_parity_v3_config.txt")
    print(f"[+] Wrote v3 parity fixture to {out_dir}")


if __name__ == "__main__":
    fixtures = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    write_parity_fixture(fixtures)
    write_parity_fixture_v3(fixtures)
