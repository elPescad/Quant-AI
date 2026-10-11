"""How safe is it? Monte Carlo on a strategy's own daily returns.

    python python/monte_carlo.py runs/alpaca_iex/fold*/bt_ensemble_g100.equity.csv runs/alpaca_iex/final/bt_ensemble_g100.equity.csv
    python python/monte_carlo.py out/account.csv                  # the Alpaca paper account

Inputs: equity curves, either `quant_engine --equity` files (bar_ts,equity; compare_methods.py
writes one per backtest) or the paper broker's account.csv. Several files, e.g. the walk-forward
folds and the test period, are joined as one series of daily returns.

  1. Is the edge real? The same daily returns with their average removed (an otherwise identical
     strategy with no edge), resampled in blocks so calm and turbulent stretches stay together:
     how often does such a strategy show a Sharpe ratio at least this high by luck?
  2. How sure is the Sharpe ratio? Block-bootstrap 90% interval.
  3. What can happen next? 10,000 simulated paths of 1, 3, 6 and 12 months: the chance of a loss,
     a bad case (worst 5%) and the drawdowns. Once if the measured edge persists, and once if it
     is really zero (the cautious view to size the account by).
  4. How long until you would know? Trading days for this Sharpe ratio to be two standard errors
     from zero.

Monte Carlo cannot find an edge or predict prices; it measures how much luck could explain
and how bad the path to the long-run result can get.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regime_study import stationary_bootstrap_idx  # noqa: E402

HORIZONS = {"1 month": 21, "3 months": 63, "6 months": 126, "12 months": 252}


def daily_returns(path, capital):
    """Last equity of each New York trading day -> daily returns on the strategy's capital."""
    rows = list(csv.DictReader(Path(path).open()))
    if not rows:
        return pd.Series(dtype=float)
    if "bar_ts" in rows[0]:  # quant_engine --equity: the simulated $10k account
        t = pd.to_datetime([int(r["bar_ts"]) for r in rows], unit="s", utc=True)
        eq = np.array([float(r["equity"]) for r in rows])
        start = capital
    else:  # account.csv: the whole paper account; the strategy trades `capital` of it
        rows = [r for r in rows if r.get("equity")]
        t = pd.to_datetime([r["time"] for r in rows], utc=True)
        eq = np.array([float(r["equity"]) for r in rows])
        start = None
    s = pd.Series(eq, index=t.tz_convert("America/New_York").date).groupby(level=0).last()
    pnl = s.diff()
    if start is not None:  # first day against the starting capital
        pnl.iloc[0] = s.iloc[0] - start
    return (pnl.dropna() / capital)


def sharpe(r):
    sd = r.std(ddof=1, axis=-1)
    return np.where(sd > 0, r.mean(axis=-1) / np.where(sd > 0, sd, 1) * math.sqrt(252), 0.0)


def max_drawdown(paths):
    eq = np.cumprod(1.0 + paths, axis=-1)
    return (1.0 - eq / np.maximum.accumulate(eq, axis=-1)).max(axis=-1)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("equity", nargs="+", type=Path, help="equity CSVs (quant_engine --equity, or account.csv)")
    p.add_argument("--capital", type=float, default=10000.0, help="capital the strategy trades (default 10000)")
    p.add_argument("--block", type=float, default=5.0, help="mean bootstrap block length in days (default 5)")
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)

    r = pd.concat([daily_returns(f, a.capital) for f in a.equity]).to_numpy()
    n = len(r)
    if n < 10:
        sys.exit(f"[-] {n} trading days: too few for a Monte Carlo (need 10+, ideally 60+)")
    obs = float(sharpe(r))
    eq = np.cumprod(1.0 + r)
    print(f"{n} trading days from {len(a.equity)} file(s) | total {100 * (eq[-1] - 1):+.2f}% | daily mean {1e4 * r.mean():+.2f} bp, "
          f"sd {1e4 * r.std(ddof=1):.1f} bp | Sharpe {obs:.2f} | max drawdown {100 * max_drawdown(r):.2f}% | "
          f"days flat {100 * np.mean(r == 0):.0f}%")

    # 1. Zero-edge Monte Carlo
    r0 = r - r.mean()
    idx = stationary_bootstrap_idx(n, n, a.paths, a.block, rng)
    null = sharpe(r0[idx])
    p_luck = (np.sum(null >= obs) + 1) / (a.paths + 1)
    print(f"\n1. Luck: a strategy with no edge and the same volatility shows Sharpe >= {obs:.2f} in {100 * p_luck:.1f}% of "
          f"{a.paths} simulated histories of {n} days" + ("  -> could easily be luck" if p_luck > 0.05 else "  -> unlikely to be luck alone"))

    # 2. Bootstrap interval of the Sharpe ratio
    boot = sharpe(r[idx])
    lo, hi = np.quantile(boot, [0.05, 0.95])
    print(f"2. Sharpe 90% interval [{lo:.2f}, {hi:.2f}]; the standard error after {n} days is about {math.sqrt(252 / n):.2f}")

    # 3. Forward risk, if the edge persists and if it is zero
    print(f"\n3. Next {', '.join(HORIZONS)} ({a.paths} simulated paths, blocks of ~{a.block:g} days):")
    print(f"   {'':<10}{'scenario':<16}{'P(loss)':>9}{'median':>9}{'bad 5%':>9}{'median DD':>11}{'bad DD 5%':>11}")
    for label, h in HORIZONS.items():
        for scen, src in [("edge persists", r), ("no edge", r0)]:
            paths = src[stationary_bootstrap_idx(n, h, a.paths, a.block, rng)]
            total = np.prod(1.0 + paths, axis=1) - 1.0
            dd = max_drawdown(paths)
            print(f"   {label:<10}{scen:<16}{100 * np.mean(total < 0):>8.0f}%{100 * np.median(total):>+8.2f}%"
                  f"{100 * np.quantile(total, 0.05):>+8.2f}%{100 * np.median(dd):>10.2f}%{100 * np.quantile(dd, 0.95):>10.2f}%")

    # 4. Time to know
    daily = r.mean() / r.std(ddof=1) if r.std() > 0 else 0.0
    print()
    if daily > 0:
        need = (2.0 / daily) ** 2
        print(f"4. At this Sharpe ({obs:.2f}) it takes about {need:.0f} trading days ({need / 252:.1f} years) of results to be "
              f"two standard errors from zero; you have {n}.")
    else:
        print(f"4. The measured Sharpe ({obs:.2f}) is not positive: no amount of time makes this edge significant.")
    print(f"   With {n} days, a strategy with no edge shows |Sharpe| above {2 * math.sqrt(252 / n):.2f} about 5% of the time.")


if __name__ == "__main__":
    main()
