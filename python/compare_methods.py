"""Walk-forward selection of the trading configuration, then one look at the test period.

    python python/compare_methods.py --data data/alpaca_ticks.csv

Timeline (by bar timestamp):
    [ ........ pre-test (80%) ........ ][ test (20%) ]
                     [ fold1 ][ fold2 ][ fold3 ]
Fold k: models are trained only on data before the fold (train_and_export.py makes every
model choice on a slice of that training data), then each candidate trades the fold.
A candidate's score is its mean fold Sharpe minus one standard error across folds, so a
configuration that looks great in one window only does not win. The selected candidate,
and every other for reference, then trades the untouched test period once, with models
retrained on all pre-test data.

Candidates: model x QUBO risk aversion gamma (gamma 0 == greedy) [x online learning rate,
with --online-lrs], where model is
    quantum       quantum GRU alone
    ensemble      average of quantum and raw GRU probabilities
    q_veto        quantum GRU; HOLD when the raw GRU points the other way
    q_veto_short  quantum GRU; only its SELL calls need raw agreement
Long and short P&L are reported separately. Finally model_runner (C++ inference, no trading)
scores every model on the last fold and on test: accuracy, signal strength, latency.
Results go to <workdir>/results.csv.
"""

import argparse
import csv
import os
import re
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_FRAC = 0.2        # Matches train_and_export.py's default (60/20/20)
WALK_FORWARD_FRAC = 0.45  # Share of the pre-test period covered by the folds
MODELS = {"quantum": "quant_model", "ensemble": "ensemble_model",
          "q_veto": "quant_veto_model", "q_veto_short": "quant_veto_short_model"}
RUNNER_METRICS = {
    "accuracy": r"Accuracy:\s+([-\d.]+)",
    "macro_f1": r"Macro F1:\s+([-\d.]+)",
    "edge_p90": r"Edge .*p90 ([-\d.]+)",
    "edge_p99": r"Edge .*p99 ([-\d.]+)",
    "gru_us": r"GRU forward ([-\d.]+) us",
}
METRICS = {
    "return_pct": r"Total Net PnL:.*\(([-\d.e+]+)%\)",
    "sharpe": r"Sharpe Ratio \(ann.\):\s+([-\d.e+]+)",
    "max_dd_pct": r"Max Drawdown:\s+([-\d.e+]+)",
    "round_trips": r"Total Round Trips:\s+(\d+)",
    "exposure_pct": r"Avg Gross Exposure:\s+([-\d.e+]+)",
}


def train(data, model_dir, val_start, test_start, log, seeds):
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Training -> {model_dir} (val from {val_start}, test from {test_start})", flush=True)
    with log.open("w") as f:
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "python" / "train_and_export.py"), "--data", str(data),
                            "--model-dir", str(model_dir), "--val-start", str(val_start), "--test-start", str(test_start),
                            "--seeds", str(seeds)],
                           stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f"[-] training failed, see {log}")


def log_name(c):
    """bt_<model>_g<gamma>[_lr<online lr>]: the name sanity_check.py looks for"""
    return f"bt_{c[0]}_g{c[1]:g}" + (f"_lr{c[2]:g}" if c[2] > 0 else "")


def backtest(engine, data, model_dir, feats, gamma, online_lr, start, end, log):
    cmd = [str(engine), "--data", str(data), "--model", str(model_dir / f"{MODELS[feats]}.weights"),
           "--start-ts", str(start), "--trades", str(log.with_suffix(".trades.csv"))]
    if end is not None:
        cmd += ["--end-ts", str(end)]
    cmd += ["--allocator", "greedy"] if gamma == 0 else ["--allocator", "qubo", "--risk-aversion", str(gamma)]
    if online_lr > 0:
        cmd += ["--online-lr", str(online_lr)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    log.write_text(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit(f"[-] {' '.join(cmd)} failed (full log: {log}):\n{out.stderr[:1500]}")
    row = {}
    for key, pat in METRICS.items():
        m = re.search(pat, out.stdout)
        if not m:
            sys.exit(f"[-] could not parse '{key}' from {log}")
        row[key] = float(m.group(1))
    row.update(side_pnl(log.with_suffix(".trades.csv")))
    return row


def side_pnl(trades_csv):
    """Gross P&L ($, before fees) of closed long and short round trips in an engine trade log."""
    out = {"long_pnl": 0.0, "short_pnl": 0.0, "long_trips": 0, "short_trips": 0}
    open_pos = {}
    with trades_csv.open() as f:
        for r in csv.DictReader(f):
            price, units = float(r["fill_price"]), abs(float(r["units_traded"]))
            if r["action"] in ("OPEN_LONG", "OPEN_SHORT"):
                open_pos[r["ticker"]] = (1 if r["action"] == "OPEN_LONG" else -1, price, units)
            elif r["ticker"] in open_pos:
                side, entry, n = open_pos.pop(r["ticker"])
                key = "long" if side > 0 else "short"
                out[f"{key}_pnl"] += side * (price - entry) * n
                out[f"{key}_trips"] += 1
    return out


def model_runner(runner, data, model, period):
    out = subprocess.run([str(runner), "--data", str(data), "--model", str(model), "--period", period],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    vals = {}
    for key, pat in RUNNER_METRICS.items():
        m = re.search(pat, out.stdout)
        vals[key] = float(m.group(1)) if m else float("nan")
    return vals


def run_all(jobs, engine, data, model_dir, candidates, start, end, logdir):
    logdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {c: pool.submit(backtest, engine, data, model_dir, c[0], c[1], c[2], start, end,
                                  logdir / f"{log_name(c)}.log") for c in candidates}
        return {c: f.result() for c, f in futures.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--engine", type=Path, default=PROJECT_ROOT / "build" / "quant_engine")
    p.add_argument("--runner", type=Path, default=PROJECT_ROOT / "build" / "model_runner")
    p.add_argument("--workdir", type=Path, help="default runs/<data file stem>")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seeds", type=int, default=5, help="random starts per GRU, averaged (passed to training)")
    p.add_argument("--gammas", type=float, nargs="+", default=[0.0, 25.0, 100.0], help="QUBO risk aversion grid (0 = greedy)")
    p.add_argument("--online-lrs", type=float, nargs="+", default=[0.0],
                   help="online learning rates to compare, e.g. 0 0.001 0.003 (0 = off; no retraining needed)")
    p.add_argument("--min-trades", type=float, default=5.0, help="minimum average round trips per fold to be selectable")
    p.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel backtests")
    p.add_argument("--skip-train", action="store_true", help="reuse models already in <workdir>; train only missing folds")
    a = p.parse_args()
    if a.folds < 2:
        sys.exit("[-] --folds must be at least 2 (the score needs a spread across folds)")

    data = a.data.resolve()
    engine = a.engine.resolve()
    workdir = (a.workdir or PROJECT_ROOT / "runs" / data.stem).resolve()
    timeline = np.unique(pl.read_csv(data, columns=["timestamp"])["timestamp"].to_numpy())
    test_start = int(timeline[int(len(timeline) * (1 - TEST_FRAC))])
    pre = timeline[timeline < test_start]
    first = int(len(pre) * (1 - WALK_FORWARD_FRAC))
    edges = [int(pre[first + (len(pre) - first) * k // a.folds]) for k in range(a.folds)] + [test_start]
    candidates = [(f, g, lr) for f in MODELS for g in a.gammas for lr in a.online_lrs]
    print(f"[+] {len(timeline)} bars | {a.folds} walk-forward folds from {edges[0]} | test from {test_start}")
    print(f"[+] {len(candidates)} candidates: {', '.join(MODELS)} x gamma {a.gammas} x online lr {a.online_lrs}")

    # Fail now, not after an hour of training, if the engine binary predates a flag we need
    if not engine.exists():
        sys.exit(f"[-] {engine} not found: build it with  cmake -S . -B build && cmake --build build")
    usage = subprocess.run([str(engine), "--help"], capture_output=True, text=True).stdout
    if any(lr > 0 for lr in a.online_lrs) and "--online-lr" not in usage:
        sys.exit(f"[-] {engine} is older than the code (no --online-lr): rebuild it with  cmake --build build")

    def need_training(model_dir):
        return not (a.skip_train and (model_dir / "ensemble_model.weights").exists())

    fold_results = []
    for k in range(a.folds):
        fold_dir = workdir / f"fold{k + 1}"
        if need_training(fold_dir / "models"):
            train(data, fold_dir / "models", edges[k], edges[k + 1], fold_dir / "train.log", a.seeds)
        print(f"    fold {k + 1}: backtesting [{edges[k]}, {edges[k + 1]})", flush=True)
        fold_results.append(run_all(a.jobs, engine, data, fold_dir / "models", candidates, edges[k], edges[k + 1], fold_dir))

    final_dir = workdir / "final"
    if need_training(final_dir / "models"):
        train(data, final_dir / "models", test_start, test_start, final_dir / "train.log", a.seeds)
    print(f"    test: backtesting [{test_start}, end)", flush=True)
    test = run_all(a.jobs, engine, data, final_dir / "models", candidates, test_start, None, final_dir)

    summary = {}
    for c in candidates:
        sharpes = [fr[c]["sharpe"] for fr in fold_results]
        mean = st.mean(sharpes)
        se = st.stdev(sharpes) / len(sharpes) ** 0.5
        trades = st.mean(fr[c]["round_trips"] for fr in fold_results)
        summary[c] = {"sharpes": sharpes, "mean": mean, "se": se, "score": mean - se, "trades": trades,
                      "long": sum(fr[c]["long_pnl"] for fr in fold_results),
                      "short": sum(fr[c]["short_pnl"] for fr in fold_results)}
    selectable = [c for c in candidates if summary[c]["trades"] >= a.min_trades and summary[c]["score"] > 0]
    chosen = max(selectable, key=lambda c: summary[c]["score"]) if selectable else None

    with (workdir / "results.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "gamma", "online_lr", *[f"fold{k + 1}_sharpe" for k in range(a.folds)], "wf_mean", "wf_se", "wf_score",
                    "wf_trades_per_fold", "wf_long_pnl", "wf_short_pnl", *[f"test_{m}" for m in METRICS],
                    "test_long_pnl", "test_short_pnl", "selected"])
        for c in candidates:
            s, t = summary[c], test[c]
            w.writerow([c[0], c[1], c[2], *s["sharpes"], s["mean"], s["se"], s["score"], s["trades"], s["long"], s["short"],
                        *[t[m] for m in METRICS], t["long_pnl"], t["short_pnl"], c == chosen])

    print(f"\nResults for {data.name}: walk-forward ({a.folds} folds) vs held-out test, ranked by WF score = mean - SE")
    print(f"{'':<2}{'model':<14}{'gamma':>6}{'lr':>7}{'WF Sharpe':>16}{'folds>0':>9}{'trips/fold':>11}{'WF long $':>11}{'WF short $':>11}"
          f"{'test Sharpe':>13}{'test ret %':>11}{'trips':>7}{'long $':>9}{'short $':>9}")
    for c in sorted(candidates, key=lambda c: summary[c]["score"], reverse=True):
        s, t = summary[c], test[c]
        print(f"{'*' if c == chosen else '':<2}{c[0]:<14}{c[1]:>6g}{c[2]:>7g}{s['mean']:>9.2f} ± {s['se']:<4.2f}"
              f"{sum(x > 0 for x in s['sharpes']):>6}/{a.folds}{s['trades']:>11.1f}{s['long']:>11.2f}{s['short']:>11.2f}"
              f"{t['sharpe']:>13.2f}{t['return_pct']:>11.2f}{t['round_trips']:>7.0f}{t['long_pnl']:>9.2f}{t['short_pnl']:>9.2f}")
    print("(long $ / short $: gross P&L of closed long / short round trips on $10k; WF columns sum all folds)")

    if chosen is None:
        print("\nNo candidate had a positive walk-forward score with enough trades: no consistent edge, "
              "so the Sharpe-maximising decision is to stay flat.")
    else:
        s, t = summary[chosen], test[chosen]
        if t["round_trips"] == 0:
            verdict = "NO TRADES in test: nothing to judge"
        else:
            inside = abs(t["sharpe"] - s["mean"]) <= 2 * max(s["se"], 1e-9) * a.folds ** 0.5
            verdict = (f"{'POSITIVE' if t['sharpe'] > 0 else 'NEGATIVE'}; "
                       f"{'within' if inside else 'outside'} the fold-to-fold range of ±2 sd")
        print(f"\nSelected: {chosen[0]}, gamma {chosen[1]:g}, online lr {chosen[2]:g} | WF Sharpe {s['mean']:.2f} ± {s['se']:.2f} "
              f"-> test Sharpe {t['sharpe']:.2f} ({verdict})")

    runner = a.runner.resolve()
    if not runner.exists():
        print(f"\n[!] {runner} not found: skipping the model_runner check")
        return
    last_fold = workdir / f"fold{a.folds}" / "models"
    print(f"\nmodel_runner (C++ inference, no trading): fold {a.folds} models on fold {a.folds} | final models on test")
    print(f"{'model':<14}{'accuracy':>18}{'macro F1':>18}{'edge p90':>18}{'edge p99':>18}{'GRU us':>10}")
    for name, file in [("raw", "baseline_model"), *MODELS.items()]:
        f = model_runner(runner, data, last_fold / f"{file}.weights", "val")
        t = model_runner(runner, data, final_dir / "models" / f"{file}.weights", "test")
        if f is None or t is None:
            print(f"{name:<14} model_runner failed")
            continue
        cells = "".join(f"{f[k]:>9.3f} / {t[k]:<6.3f}" for k in ("accuracy", "macro_f1", "edge_p90", "edge_p99"))
        print(f"{name:<14}{cells}{t['gru_us']:>10.0f}")
    print("(each cell: last fold / test; the engine only trades when edge x typical move beats the round-trip cost)")


if __name__ == "__main__":
    main()
