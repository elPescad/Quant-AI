"""Is a compare_methods.py result skill, or market drift and luck? Runs in seconds, no training.

    python python/sanity_check.py --data data/alpaca_ticks.csv            # the selected candidate
    python python/sanity_check.py --data data/alpaca_ticks.csv --model quantum --gamma 25

For each walk-forward fold (and the test period) it compares the strategy with
  * buy & hold: an equal-weight basket of all tickers over the same window
  * all-long: the strategy's own round trips, all taken long (pure market exposure)
  * market-neutral P&L: each round trip's return minus beta x the basket's return over the
    same holding period (beta per ticker from data before the window), times its direction
and tests the market-neutral P&L against random directions (sign-flip permutation).
Positive market-neutral P&L that random directions rarely match is evidence of skill;
P&L that comes from being long while the market rose is not.
Assumes no malformed CSV rows (the engine numbers ticks by row).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BARS_PER_YEAR = 252 * 78
BETA_LOOKBACK_BARS = 2000


def read_config(path):
    return dict(line.split("=", 1) for line in path.read_text().split() if "=" in line)


def round_trips(trades_csv, ts_of_tick):
    """(ticker, direction, entry_ts, exit_ts, notional, gross_return) per closed round trip."""
    out, open_pos = [], {}
    with trades_csv.open() as f:
        for r in csv.DictReader(f):
            tick, price, units = int(r["tick_id"]), float(r["fill_price"]), abs(float(r["units_traded"]))
            if r["action"] in ("OPEN_LONG", "OPEN_SHORT"):
                open_pos[r["ticker"]] = (1 if r["action"] == "OPEN_LONG" else -1, tick, price, units)
            elif r["ticker"] in open_pos:
                d, t0, p0, n = open_pos.pop(r["ticker"])
                out.append((r["ticker"], d, ts_of_tick[t0], ts_of_tick[tick], n * p0, price / p0 - 1.0))
    return out


def sharpe(r):
    r = r[np.isfinite(r)]
    return float(r.mean() / r.std(ddof=1) * np.sqrt(BARS_PER_YEAR)) if len(r) > 2 and r.std() > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--workdir", type=Path, help="compare_methods.py workdir (default runs/<data stem>)")
    p.add_argument("--model", help="candidate model (default: the selected one in results.csv)")
    p.add_argument("--gamma", type=float)
    p.add_argument("--online-lr", type=float, default=0.0)
    p.add_argument("--permutations", type=int, default=20000)
    a = p.parse_args()
    workdir = (a.workdir or PROJECT_ROOT / "runs" / a.data.resolve().stem).resolve()

    results = list(csv.DictReader((workdir / "results.csv").open()))
    if a.model is None:
        chosen = [r for r in results if r["selected"] == "True"]
        if not chosen:
            sys.exit("[-] results.csv has no selected candidate; pass --model and --gamma")
        a.model, a.gamma = chosen[0]["model"], float(chosen[0]["gamma"])
        a.online_lr = float(chosen[0].get("online_lr", 0) or 0)
    row = next((r for r in results if r["model"] == a.model and float(r["gamma"]) == a.gamma
                and float(r.get("online_lr", 0) or 0) == a.online_lr), None)
    if row is None:
        sys.exit(f"[-] {a.model} gamma {a.gamma:g} online lr {a.online_lr:g} is not in {workdir / 'results.csv'}")
    suffix = f"_lr{a.online_lr:g}" if a.online_lr > 0 else ""
    n_folds = sum(1 for k in row if k.startswith("fold") and k.endswith("_sharpe"))

    df = pl.read_csv(a.data, columns=["timestamp", "ticker", "raw_price"])
    ts_of_tick = df["timestamp"].to_numpy()
    prices = df.pivot(on="ticker", index="timestamp", values="raw_price", aggregate_function="last").sort("timestamp")
    stamps = prices["timestamp"].to_numpy()
    tickers = [c for c in prices.columns if c != "timestamp"]
    px = prices.select(tickers).fill_null(strategy="forward").to_numpy().astype(np.float64)
    rets = px[1:] / px[:-1] - 1.0
    basket = np.nanmean(rets, axis=1)
    basket_index = np.concatenate([[1.0], np.cumprod(1.0 + np.nan_to_num(basket))])  # value at each stamp

    windows = []
    for k in range(1, n_folds + 1):
        cfg = read_config(workdir / f"fold{k}" / "models" / "quant_model_config.txt")
        windows.append((f"fold {k}", workdir / f"fold{k}", int(cfg["val_start_ts"]), int(cfg["test_start_ts"]),
                        float(row[f"fold{k}_sharpe"])))
    cfg = read_config(workdir / "final" / "models" / "quant_model_config.txt")
    windows.append(("test", workdir / "final", int(cfg["test_start_ts"]), int(stamps[-1]) + 1, float(row["test_sharpe"])))

    rng = np.random.default_rng(0)
    print(f"Sanity check: {a.model}, gamma {a.gamma:g}, online lr {a.online_lr:g} ({workdir})\n")
    print(f"{'window':<8}{'strategy':>10}{'buy&hold':>10}{'trips':>7}{'gross $':>10}{'all-long $':>12}"
          f"{'mkt-neutral $':>15}{'hit rate':>10}{'p (random dir)':>16}")
    print(f"{'':<8}{'Sharpe':>10}{'Sharpe':>10}")
    pooled = []
    for name, wdir, lo, hi, strat_sharpe in windows:
        in_win = (stamps[1:] >= lo) & (stamps[1:] < hi)
        bh = sharpe(basket[in_win])
        trips = round_trips(wdir / f"bt_{a.model}_g{a.gamma:g}{suffix}.trades.csv", ts_of_tick)
        if not trips:
            print(f"{name:<8}{strat_sharpe:>10.2f}{bh:>10.2f}{0:>7}{'no trades':>12}")
            continue
        # Per-ticker beta to the basket, from bars before the window (no look-ahead)
        before = np.nonzero(stamps[1:] < lo)[0][-BETA_LOOKBACK_BARS:]
        betas = {}
        for j, t in enumerate(tickers):
            x, y = basket[before], rets[before, j]
            ok = np.isfinite(x) & np.isfinite(y)
            betas[t] = float(np.cov(x[ok], y[ok])[0, 1] / np.var(x[ok], ddof=1)) if ok.sum() > 50 else 1.0
        d = np.array([t[1] for t in trips], dtype=np.float64)
        notional = np.array([t[4] for t in trips])
        raw = np.array([t[5] for t in trips])
        mkt = np.array([basket_index[np.searchsorted(stamps, t[3])] / basket_index[np.searchsorted(stamps, t[2])] - 1.0
                        for t in trips])
        beta = np.array([betas[t[0]] for t in trips])
        excess = notional * (raw - beta * mkt)  # $ market-neutral move of each position, taken long
        neutral = d * excess
        if name != "test":
            pooled.append(neutral)
        signs = rng.choice([-1.0, 1.0], size=(a.permutations, len(trips)))
        p_val = float(np.mean((signs * np.abs(excess)).sum(1) >= neutral.sum()))
        print(f"{name:<8}{strat_sharpe:>10.2f}{bh:>10.2f}{len(trips):>7}{(d * notional * raw).sum():>10.2f}"
              f"{(notional * raw).sum():>12.2f}{neutral.sum():>15.2f}{np.mean(neutral > 0):>10.2f}{p_val:>16.3f}")

    if pooled:
        allx = np.concatenate(pooled)
        signs = rng.choice([-1.0, 1.0], size=(a.permutations, len(allx)))
        p_val = float(np.mean((signs * np.abs(allx)).sum(1) >= allx.sum()))
        n_cand = len(results)
        print(f"\nAll walk-forward folds: {len(allx)} round trips, market-neutral P&L ${allx.sum():.2f}, "
              f"hit rate {np.mean(allx > 0):.2f}, p = {p_val:.3f} vs random directions")
        print(f"This candidate was picked from {n_cand}; with that many tries, p < {0.05 / n_cand:.4f} is needed for 5% significance.")
        if p_val < 0.05 / n_cand:
            print("VERDICT: the direction calls earn money beyond market drift more often than chance allows.")
        else:
            print("VERDICT: not distinguishable from luck plus market drift. Do not treat the walk-forward Sharpe as an edge.")
    print("\n(gross $: what the strategy made; all-long $: same trades all taken long; mkt-neutral $: gross with each\n"
          " trade's beta-adjusted basket move removed. Permutation assumes trades are independent, so p is optimistic.)")


if __name__ == "__main__":
    main()
