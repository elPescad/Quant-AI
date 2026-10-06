"""Train and export the GRU models used by the engine.

Time splits (no shuffling across time):
    [ fit | inner ]  [ val ]  [ test ]
    fit    gradient steps
    inner  last INNER_FRAC of the training window: early stopping, quantum bandwidth,
           probability calibration. Every model choice is made here, so val stays
           out-of-sample and a val backtest is an honest estimate.
    val    [--val-start, --test-start): reported, never used for fitting or selection
    test   [--test-start, end): reported, never used for fitting or selection
Samples whose 6-bar-ahead label reaches across a split boundary are purged.

Exported (models/):
    baseline_model   GRU on the 5 classical features
    quant_model      GRU on classical + 62 quantum feature-map features
    ensemble_model   average of the two calibrated GRUs' probabilities (quantum config)
    quant_veto_model        quantum GRU, HOLD when the raw GRU disagrees on direction
    quant_veto_short_model  same, but only quantum SELL calls need raw agreement
each with a _config.txt, plus training_report.txt.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from quantum_features import N_CLASSICAL, FeatureConfig, classical_features, model_inputs

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CSV_FILE = DATA_DIR / "market_ticks.csv"

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
INNER_FRAC = 0.2  # Share of the training window held out for early stopping / selection / calibration
BANDWIDTHS = [0.1, 0.25, 0.5, 1.0]
BATCH_SIZE = 256
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3
SEED = 7


class QuantGRU(nn.Module):
    """GRU over the feature sequence; the readout is linear in [GRU state, current feature vector]."""

    def __init__(self, input_dim, hidden_dim=32, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden_dim + input_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(torch.cat([out[:, -1, :], x[:, -1, :]], dim=1))


class LastStepLinear(nn.Module):
    """Logistic regression on the last bar's features (used to pick the quantum bandwidth)."""

    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x[:, -1, :])


class Calibrated(nn.Module):
    """logits / T + b, fitted on the inner split with unweighted NLL.

    Training uses class weights, which distorts the probabilities; the engine turns
    p_buy - p_sell into an expected return and compares it with trading costs, so the
    probabilities themselves need to be right, not just their argmax."""

    def __init__(self, base, temperature, bias, n_inputs):
        super().__init__()
        self.base = base
        self.n_inputs = n_inputs
        self.register_buffer("inv_t", torch.tensor(1.0 / temperature))
        self.register_buffer("bias", bias.clone())

    def forward(self, x):
        return self.base(x[:, :, : self.n_inputs]) * self.inv_t + self.bias


class Ensemble(nn.Module):
    """Mean of the calibrated raw and quantum GRU probabilities, returned as log-probabilities
    (the engine applies softmax, which recovers the averaged probabilities exactly)."""

    def __init__(self, raw, quantum):
        super().__init__()
        self.raw = raw
        self.quantum = quantum

    def forward(self, x):
        p = 0.5 * (torch.softmax(self.raw(x), 1) + torch.softmax(self.quantum(x), 1))
        return torch.log(p.clamp_min(1e-12))


class RawVeto(nn.Module):
    """Quantum GRU trades; the raw GRU only stabilises. When the two disagree on direction
    (opposite signs of p_buy - p_sell) the quantum call is replaced by HOLD. With
    shorts_only, only quantum SELL calls need raw agreement."""

    def __init__(self, quantum, raw, shorts_only):
        super().__init__()
        self.quantum = quantum
        self.raw = raw
        self.shorts_only = shorts_only
        self.register_buffer("hold", torch.tensor([[0.0, 1.0, 0.0]]))

    def forward(self, x):
        pq = torch.softmax(self.quantum(x), 1)
        pr = torch.softmax(self.raw(x), 1)
        eq = pq[:, 2] - pq[:, 0]
        veto = eq * (pr[:, 2] - pr[:, 0]) < 0
        if self.shorts_only:
            veto = veto & (eq < 0)
        p = torch.where(veto.unsqueeze(1), self.hold, pq)
        return torch.log(p.clamp_min(1e-12))


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_ticks():
    df = pl.read_csv(CSV_FILE)
    expected = ["timestamp", "ticker", "raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol", "target"]
    if df.columns[: len(expected)] != expected:
        raise SystemExit(f"[-] {CSV_FILE} has columns {df.columns}; regenerate it with a fetch_*.py script or generate_ticks.py")
    return df.sort("timestamp", maintain_order=True)


def default_boundaries(timeline):
    return int(timeline[int(len(timeline) * TRAIN_FRAC)]), int(timeline[int(len(timeline) * (TRAIN_FRAC + VAL_FRAC))])


def purge_before(timeline, boundary, horizon):
    """Samples stamped before the returned timestamp have labels that end before `boundary`."""
    idx = int(np.searchsorted(timeline, boundary))
    return int(timeline[max(0, idx - horizon)]) if idx < len(timeline) else int(timeline[max(0, len(timeline) - horizon)])


def per_ticker_classical(df, cfg):
    """Classical features per ticker, computed in stream order exactly like the engine."""
    out = {}
    for (sym,), g in df.group_by(["ticker"], maintain_order=True):
        c = classical_features(g["raw_spread"].to_numpy(), g["raw_ofi"].to_numpy(),
                               g["raw_delta"].to_numpy(), g["raw_vol"].to_numpy(), cfg)
        out[sym] = (c, g["target"].to_numpy(), g["timestamp"].to_numpy())
    return out


class WindowSet:
    """All per-tick feature rows plus window end indices; windows are gathered per batch."""

    def __init__(self, classical, cfg):
        rows, ends, targets, stamps = [], [], [], []
        offset = 0
        first = max(cfg.min_warmup, cfg.seq_len) - 1  # Same readiness rule as the C++ pipeline
        for sym, (c, y, ts) in classical.items():
            rows.append(model_inputs(c, cfg))
            valid = np.arange(first, len(c))
            valid = valid[y[valid] >= 0]
            ends.append(valid + offset)
            targets.append(y[valid])
            stamps.append(ts[valid])
            offset += len(c)
        self.features = torch.from_numpy(np.concatenate(rows))
        self.ends = np.concatenate(ends)
        self.targets = np.concatenate(targets)
        self.stamps = np.concatenate(stamps)
        self.seq_len = cfg.seq_len
        self.dim = self.features.shape[1]

    def split(self, lo, hi):
        return np.nonzero((self.stamps >= lo) & (self.stamps < hi))[0]

    def batch(self, idx):
        win = torch.from_numpy(self.ends[idx][:, None] + np.arange(-self.seq_len + 1, 1)[None, :])
        return self.features[win], torch.from_numpy(self.targets[idx]).long()


# --------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------
def class_weights(y):
    counts = np.bincount(y, minlength=3).astype(np.float64)
    return torch.tensor(np.sqrt(len(y) / (3.0 * counts + 1e-5)), dtype=torch.float32)


def predict_logits(model, data, idx):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), 2048):
            out.append(model(data.batch(idx[k:k + 2048])[0]))
    return torch.cat(out)


def evaluate(model, data, idx):
    logits = predict_logits(model, data, idx)
    y = torch.from_numpy(data.targets[idx]).long()
    nll = nn.functional.cross_entropy(logits, y).item()
    pred = logits.argmax(1).numpy()
    yt = y.numpy()
    f1s = []
    for c in range(3):
        tp = np.sum((pred == c) & (yt == c))
        prec = tp / max(1, np.sum(pred == c))
        rec = tp / max(1, np.sum(yt == c))
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    directional = pred != 1
    dir_prec = float(np.mean(pred[directional] == yt[directional])) if directional.any() else 0.0
    return {"nll": nll, "acc": float(np.mean(pred == yt)), "macro_f1": float(np.mean(f1s)),
            "dir_calls": float(np.mean(directional)), "dir_precision": dir_prec}


def train(model, data, fit_idx, stop_idx, label):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    criterion = nn.CrossEntropyLoss(weight=class_weights(data.targets[fit_idx]))
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    y_stop = torch.from_numpy(data.targets[stop_idx]).long()
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(fit_idx)
        for k in range(0, len(perm), BATCH_SIZE):
            xb, yb = data.batch(perm[k:k + BATCH_SIZE])
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        stop_loss = criterion(predict_logits(model, data, stop_idx), y_stop).item()
        if stop_loss < best - 1e-4:
            best, best_state, bad = stop_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    print(f"    {label:<17} {epoch:2d} epochs, best inner loss {best:.4f} ({time.time() - t0:.0f}s)")
    return model


def calibrate(model, data, idx, n_inputs):
    """Fit temperature + per-class bias on held-out logits (unweighted NLL)."""
    logits = predict_logits(model, data, idx)
    y = torch.from_numpy(data.targets[idx]).long()
    log_t = torch.zeros(1, requires_grad=True)
    bias = torch.zeros(3, requires_grad=True)
    opt = torch.optim.LBFGS([log_t, bias], lr=0.5, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp() + bias, y)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp())
    return Calibrated(model, t, bias.detach() - bias.detach().mean(), n_inputs).eval(), t


def export(model, cfg, name):
    model.eval()
    example = torch.zeros(1, cfg.seq_len, cfg.input_dim)
    traced = torch.jit.freeze(torch.jit.trace(model, example))
    traced.save(str(MODEL_DIR / f"{name}.pt"))
    cfg.write(MODEL_DIR / f"{name}_config.txt")
    return traced


def train_and_export(val_ts=None, test_ts=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Loading {CSV_FILE}")
    df = load_ticks()
    timeline = np.unique(df["timestamp"].to_numpy())
    d_val, d_test = default_boundaries(timeline)
    val_ts = d_val if val_ts is None else val_ts
    test_ts = d_test if test_ts is None else test_ts
    if not val_ts <= test_ts:
        raise SystemExit("[-] --val-start must not be after --test-start")

    base_cfg = FeatureConfig(val_start_ts=val_ts, test_start_ts=test_ts)
    horizon = base_cfg.label_horizon
    train_ts = timeline[timeline < val_ts]
    inner_ts = int(train_ts[int(len(train_ts) * (1.0 - INNER_FRAC))])
    print(f"[+] {len(df):,} ticks, {df['ticker'].n_unique()} tickers | inner from {inner_ts}, val from {val_ts}, test from {test_ts}")

    t0 = time.time()
    classical = per_ticker_classical(df, base_cfg)
    print(f"[+] Classical features + RLS AR(1) in {time.time() - t0:.1f}s")

    raw_cfg = FeatureConfig(**{**base_cfg.__dict__, "quantum_lift": 0})
    raw = WindowSet(classical, raw_cfg)
    fit = raw.split(-np.inf, purge_before(timeline, inner_ts, horizon))
    inner = raw.split(inner_ts, purge_before(timeline, val_ts, horizon))
    val = raw.split(val_ts, purge_before(timeline, test_ts, horizon))
    test = raw.split(test_ts, np.inf)
    print(f"[+] Samples: {len(fit)} fit | {len(inner)} inner | {len(val)} val | {len(test)} test")

    print("[+] Training (all choices on the inner split):")
    gru_raw = train(QuantGRU(raw.dim), raw, fit, inner, "gru_raw")

    best_bw, best_loss, qset = None, np.inf, None
    for bw in BANDWIDTHS:
        cand = WindowSet(classical, FeatureConfig(**{**base_cfg.__dict__, "bandwidth": bw}))
        lin = train(LastStepLinear(cand.dim), cand, fit, inner, f"linear_q(bw={bw})")
        loss = evaluate(lin, cand, inner)["nll"]
        if loss < best_loss:
            best_bw, best_loss, qset = bw, loss, cand
    qcfg = FeatureConfig(**{**base_cfg.__dict__, "bandwidth": best_bw})
    print(f"[+] Selected quantum bandwidth {best_bw} (inner split)")
    gru_q = train(QuantGRU(qset.dim), qset, fit, inner, "gru_quantum")

    cal_raw, t_raw = calibrate(gru_raw, raw, inner, N_CLASSICAL)
    cal_q, t_q = calibrate(gru_q, qset, inner, qset.dim)
    ensemble = Ensemble(cal_raw, cal_q).eval()
    veto = RawVeto(cal_q, cal_raw, shorts_only=False).eval()
    veto_short = RawVeto(cal_q, cal_raw, shorts_only=True).eval()
    print(f"[+] Calibration temperatures: raw {t_raw:.2f}, quantum {t_q:.2f}")

    export(cal_raw, raw_cfg, "baseline_model")
    export(cal_q, qcfg, "quant_model")
    traced = export(ensemble, qcfg, "ensemble_model")
    export(veto, qcfg, "quant_veto_model")
    export(veto_short, qcfg, "quant_veto_short_model")
    check = test if len(test) else inner
    xb, _ = qset.batch(check[:512])
    with torch.no_grad():
        drift = (traced(xb) - ensemble(xb)).abs().max().item()

    majority = int(np.bincount(raw.targets[fit], minlength=3).argmax())
    lines = []
    for split_name, idx in [("val", val), ("test", test)]:
        if len(idx) == 0:
            continue
        lines.append(f"{split_name} split ({len(idx)} samples)")
        lines.append(f"{'model':<16}{'accuracy':>10}{'macro_f1':>10}{'nll':>8}{'buy/sell calls':>16}{'call precision':>16}")
        lines.append(f"{'always_' + ['sell', 'hold', 'buy'][majority]:<16}{np.mean(raw.targets[idx] == majority):>10.3f}")
        for name, m, data in [("gru_raw", cal_raw, raw), ("gru_quantum", cal_q, qset), ("ensemble", ensemble, qset),
                               ("q_veto", veto, qset), ("q_veto_short", veto_short, qset)]:
            r = evaluate(m, data, idx)
            lines.append(f"{name:<16}{r['acc']:>10.3f}{r['macro_f1']:>10.3f}{r['nll']:>8.4f}"
                         f"{100 * r['dir_calls']:>15.1f}%{r['dir_precision']:>16.3f}")
        lines.append(f"class mix (sell/hold/buy): {np.round(np.bincount(raw.targets[idx], minlength=3) / len(idx), 3)}")
    lines.append(f"quantum bandwidth {best_bw}, TorchScript export max |diff| {drift:.2e}")
    report = "\n".join(lines)
    (MODEL_DIR / "training_report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\n[+] Exported baseline, quant, ensemble, quant_veto, quant_veto_short models (+ _config.txt) to {MODEL_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=CSV_FILE, help="market ticks CSV")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="where to write models")
    parser.add_argument("--val-start", type=int, help="first val timestamp; training uses data before it (default: 60%%)")
    parser.add_argument("--test-start", type=int, help="first test timestamp (default: 80%%)")
    args = parser.parse_args()
    CSV_FILE, MODEL_DIR = args.data, args.model_dir
    train_and_export(args.val_start, args.test_start)
