"""Train and export the quantum-feature-map GRU (plus baselines for comparison).

Data is split chronologically (no shuffling across time):
    first 60% of bars -> train,  next 20% -> validation,  last 20% -> test
Model choices (quantum bandwidth, early stopping) use validation only; test is touched once.

Models compared on the held-out test split:
    linear_raw     logistic regression on the 5 classical features of the last bar
    linear_quantum logistic regression on the 67-dim quantum-lifted features of the last bar
    gru_raw        GRU over the sequence of 5 classical features
    gru_quantum    GRU over the sequence of quantum-lifted features   <- exported for trading

Outputs (models/):
    quant_model.pt + quant_model_config.txt          gru_quantum (used by the engine by default)
    baseline_model.pt + baseline_model_config.txt    gru_raw, for A/B backtests
    training_report.txt                              metrics table
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from quantum_features import FeatureConfig, classical_features, model_inputs

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CSV_FILE = DATA_DIR / "market_ticks.csv"

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
BANDWIDTHS = [0.1, 0.25, 0.5, 1.0]  # Quantum kernel bandwidth candidates, chosen on validation
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
    """Logistic regression on the last bar's features: tests linear separability directly."""

    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x[:, -1, :])


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_ticks():
    df = pl.read_csv(CSV_FILE)
    expected = ["timestamp", "ticker", "raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol", "target"]
    if df.columns[: len(expected)] != expected:
        raise SystemExit(f"[-] {CSV_FILE} has columns {df.columns}; regenerate it with fetch_real_ticks.py or generate_ticks.py")
    return df.sort("timestamp", maintain_order=True)


def split_timestamps(df):
    ts = np.unique(df["timestamp"].to_numpy())
    return int(ts[int(len(ts) * TRAIN_FRAC)]), int(ts[int(len(ts) * (TRAIN_FRAC + VAL_FRAC))])


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


def evaluate(model, data, idx, criterion):
    model.eval()
    logits = []
    with torch.no_grad():
        for k in range(0, len(idx), 2048):
            xb, _ = data.batch(idx[k:k + 2048])
            logits.append(model(xb))
    logits = torch.cat(logits)
    y = torch.from_numpy(data.targets[idx]).long()
    loss = criterion(logits, y).item()
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
    return {"loss": loss, "acc": float(np.mean(pred == yt)), "macro_f1": float(np.mean(f1s)),
            "dir_calls": float(np.mean(directional)), "dir_precision": dir_prec}


def train(model, data, train_idx, val_idx, label):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    criterion = nn.CrossEntropyLoss(weight=class_weights(data.targets[train_idx]))
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(train_idx)
        for k in range(0, len(perm), BATCH_SIZE):
            xb, yb = data.batch(perm[k:k + BATCH_SIZE])
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val = evaluate(model, data, val_idx, criterion)
        if val["loss"] < best - 1e-4:
            best, best_state, bad = val["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"    {label:<15} {epoch:2d} epochs, best val loss {best:.4f} ({time.time() - t0:.0f}s)")
    return model, criterion


def export(model, cfg, name):
    model.eval()
    example = torch.zeros(1, cfg.seq_len, cfg.input_dim)
    traced = torch.jit.freeze(torch.jit.trace(model, example))
    traced.save(str(MODEL_DIR / f"{name}.pt"))
    cfg.write(MODEL_DIR / f"{name}_config.txt")
    return traced


def train_and_export():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Loading {CSV_FILE}")
    df = load_ticks()
    val_ts, test_ts = split_timestamps(df)
    print(f"[+] {len(df):,} ticks, {df['ticker'].n_unique()} tickers | val from {val_ts}, test from {test_ts}")

    base_cfg = FeatureConfig(val_start_ts=val_ts, test_start_ts=test_ts)
    t0 = time.time()
    classical = per_ticker_classical(df, base_cfg)
    print(f"[+] Classical features + RLS AR(1) in {time.time() - t0:.1f}s")

    results = {}

    # ---- Raw (classical-only) baselines ------------------------------------------------
    raw_cfg = FeatureConfig(**{**base_cfg.__dict__, "quantum_lift": 0})
    raw = WindowSet(classical, raw_cfg)
    tr, va, te = raw.split(-np.inf, val_ts), raw.split(val_ts, test_ts), raw.split(test_ts, np.inf)
    print(f"[+] Samples: {len(tr)} train | {len(va)} val | {len(te)} test")
    print("[+] Training:")
    m, crit = train(LastStepLinear(raw.dim), raw, tr, va, "linear_raw")
    results["linear_raw"] = evaluate(m, raw, te, crit)
    gru_raw, crit = train(QuantGRU(raw.dim), raw, tr, va, "gru_raw")
    results["gru_raw"] = evaluate(gru_raw, raw, te, crit)

    # ---- Quantum feature map: pick the bandwidth on validation -------------------------
    best_bw, best_val, best_set = None, np.inf, None
    for bw in BANDWIDTHS:
        qcfg = FeatureConfig(**{**base_cfg.__dict__, "bandwidth": bw})
        t0 = time.time()
        qset = WindowSet(classical, qcfg)
        sim_s = time.time() - t0
        m, crit = train(LastStepLinear(qset.dim), qset, tr, va, f"linear_q(bw={bw})")
        v = evaluate(m, qset, va, crit)
        print(f"      quantum circuit simulation for {len(qset.features):,} ticks: {sim_s:.1f}s | val loss {v['loss']:.4f}")
        if v["loss"] < best_val:
            best_bw, best_val, best_set, best_lin = bw, v["loss"], qset, (m, crit)
    qcfg = FeatureConfig(**{**base_cfg.__dict__, "bandwidth": best_bw})
    print(f"[+] Selected quantum bandwidth {best_bw} (validation)")
    results["linear_quantum"] = evaluate(best_lin[0], best_set, te, best_lin[1])
    gru_q, crit_q = train(QuantGRU(best_set.dim), best_set, tr, va, "gru_quantum")
    results["gru_quantum"] = evaluate(gru_q, best_set, te, crit_q)

    # ---- Export ---------------------------------------------------------------------------
    traced = export(gru_q, qcfg, "quant_model")
    export(gru_raw, raw_cfg, "baseline_model")
    # The traced module must reproduce the eager model (it is what the C++ engine runs)
    xb, _ = best_set.batch(te[:512])
    with torch.no_grad():
        drift = (traced(xb) - gru_q(xb)).abs().max().item()

    lines = [f"Held-out test split ({len(te)} samples, timestamps >= {test_ts})",
             f"{'model':<16}{'accuracy':>10}{'macro_f1':>10}{'buy/sell calls':>16}{'call precision':>16}"]
    for name in ["linear_raw", "linear_quantum", "gru_raw", "gru_quantum"]:
        r = results[name]
        lines.append(f"{name:<16}{r['acc']:>10.3f}{r['macro_f1']:>10.3f}{100 * r['dir_calls']:>15.1f}%{r['dir_precision']:>16.3f}")
    test_labels = best_set.targets[te]
    lines.append(f"class mix (sell/hold/buy): {np.bincount(test_labels, minlength=3) / len(test_labels)}")
    lines.append(f"quantum bandwidth {best_bw}, TorchScript export max |diff| {drift:.2e}")
    report = "\n".join(lines)
    (MODEL_DIR / "training_report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\n[+] Exported {MODEL_DIR / 'quant_model.pt'} (+ _config.txt) and baseline_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=CSV_FILE, help="market ticks CSV")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="where to write models")
    args = parser.parse_args()
    CSV_FILE, MODEL_DIR = args.data, args.model_dir
    train_and_export()
