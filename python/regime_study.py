"""Does "trade more in uptrends, less in downtrends" work? Trend and volatility rules against
buy & hold on real prices, with costs, no look-ahead and Monte Carlo tests.

    pip install arch                                 # ships the real data below
    python python/regime_study.py                    # bundled real data:
                                                     #   S&P 500 and NASDAQ daily 1999-2018,
                                                     #   US stock market monthly 1926-2018 (Fama-French)
    python python/regime_study.py --alpaca SPY,QQQ,AAPL,NVDA,MSFT,AMD   # Alpaca daily bars since 2016
    python python/regime_study.py --csv prices.csv   # columns date,close[,symbol]

Rules (decided on day t's close, held over day t+1; cash earns the risk-free rate when known, else 0):
  trend N      fully invested while the close is above its N-period moving average, else cash
  graded       invested in thirds: one third for each of the 50/100/200-day averages the close is above
  vol target   exposure = min(1, 15% / realised volatility of the last 20 days): less when turbulent
  combinations of the two

Monte Carlo:
  * shuffled history: the same returns in random order (no trends, no volatility clusters). A rule's
    improvement over buy & hold that also appears on shuffled data is luck, not trend-following.
  * paired bootstrap: resample blocks of days, rule and buy & hold together, for a confidence
    interval of the Sharpe improvement.
  * risk: simulated one-year and three-month paths from each rule's own returns: chance of a
    loss, bad-case return and drawdown.
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

COST_BP = 2.0  # per unit of turnover (ETF/large-cap spread + fees), charged when exposure changes


# ----------------------------------------------------------------------------------------------
# Rules: exposure in [0, 1] decided at the close of each period
# ----------------------------------------------------------------------------------------------
def sma(x, n):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_vol(r, n, per_year):
    s = pd.Series(r).rolling(n, min_periods=n).std(ddof=1).to_numpy()
    return s * math.sqrt(per_year)


def above(close, n):
    m = sma(close, n)
    return np.where(np.isnan(m), np.nan, (close > m).astype(float))


def rules_for(close, ret, freq):
    """Exposure per rule; NaN while a rule's lookback is warming up."""
    per_year = 252 if freq == "D" else 12
    if freq == "D":
        trend_n, graded_n, vol_n = 200, (50, 100, 200), 20
    else:
        trend_n, graded_n, vol_n = 10, (6, 10, 12), 12
    vt = np.minimum(1.0, 0.15 / rolling_vol(ret, vol_n, per_year))
    trend = above(close, trend_n)
    graded = np.mean([above(close, n) for n in graded_n], axis=0)
    unit = "d" if freq == "D" else "m"
    return {
        "buy & hold": np.ones(len(close)),
        f"trend {trend_n}{unit}": trend,
        f"graded {'/'.join(map(str, graded_n))}{unit}": graded,
        "vol target 15%": vt,
        f"trend {trend_n}{unit} + vol target": trend * vt,
        "graded + vol target": graded * vt,
    }


# ----------------------------------------------------------------------------------------------
# Backtest and metrics
# ----------------------------------------------------------------------------------------------
def strategy_returns(ret, w, cash, cost_bp=COST_BP):
    """Exposure decided at t applies to period t+1; turnover is charged when it changes."""
    w = np.nan_to_num(w, nan=0.0)
    held = np.concatenate([[0.0], w[:-1]])
    turnover = np.abs(np.diff(np.concatenate([[0.0], held])))
    return held * ret + (1.0 - held) * cash - turnover * cost_bp * 1e-4, held


def metrics(r, cash, per_year, held=None):
    ex = r - cash
    eq = np.cumprod(1.0 + r)
    dd = 1.0 - eq / np.maximum.accumulate(eq)
    years = len(r) / per_year
    out = {
        "CAGR %": 100 * (eq[-1] ** (1 / years) - 1),
        "vol %": 100 * r.std(ddof=1) * math.sqrt(per_year),
        "Sharpe": ex.mean() / ex.std(ddof=1) * math.sqrt(per_year) if ex.std() > 0 else 0.0,
        "max DD %": 100 * dd.max(),
    }
    if held is not None:
        out["invested %"] = 100 * held.mean()
        out["switches/yr"] = np.abs(np.diff(held)).sum() / years
    return out


def sharpe(ex, per_year):
    sd = ex.std(ddof=1, axis=-1)
    return np.where(sd > 0, ex.mean(axis=-1) / np.where(sd > 0, sd, 1) * math.sqrt(per_year), 0.0)


# ----------------------------------------------------------------------------------------------
# Monte Carlo
# ----------------------------------------------------------------------------------------------
def stationary_bootstrap_idx(n, length, n_paths, block, rng):
    """Politis-Romano stationary bootstrap: blocks of geometric length (mean `block`)."""
    idx = np.empty((n_paths, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    restart = rng.random((n_paths, length)) < 1.0 / block
    fresh = rng.integers(0, n, (n_paths, length))
    for t in range(1, length):
        idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def shuffled_null(ret, close0, cash, freq, n_perm, rng, s):
    """Sharpe improvement over buy & hold of every rule on randomly reordered histories
    (scored over the same periods `s` as the real history)."""
    per_year = 252 if freq == "D" else 12
    out = {}
    for _ in range(n_perm):
        r = rng.permutation(ret)
        close = close0 * np.cumprod(1.0 + r)
        bh = None
        for name, w in rules_for(close, r, freq).items():
            sr, held = strategy_returns(r, w, cash)
            s_ = metrics(sr[s], cash[s], per_year)["Sharpe"]
            if bh is None:
                bh = s_
            out.setdefault(name, []).append(s_ - bh)
    return {k: np.array(v) for k, v in out.items()}


def risk_paths(r, per_year, block, rng, n_paths=10000):
    """One-year and three-month outcomes from the rule's own returns (stationary bootstrap)."""
    year = per_year
    idx = stationary_bootstrap_idx(len(r), year, n_paths, block, rng)
    paths = np.cumprod(1.0 + r[idx], axis=1)
    dd = (1.0 - paths / np.maximum.accumulate(paths, axis=1)).max(axis=1)
    q = year // 4
    return {
        "P(loss 3m) %": 100 * (paths[:, q - 1] < 1).mean(),
        "P(loss 1y) %": 100 * (paths[:, -1] < 1).mean(),
        "bad 1y (5%) %": 100 * (np.quantile(paths[:, -1], 0.05) - 1),
        "DD 1y median %": 100 * np.median(dd),
        "DD 1y bad (95%) %": 100 * np.quantile(dd, 0.95),
    }


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def load_bundled():
    import arch.data.frenchdata as french
    import arch.data.nasdaq as nasdaq
    import arch.data.sp500 as sp500

    series = []
    for name, mod in [("S&P 500", sp500), ("NASDAQ", nasdaq)]:
        d = mod.load()
        series.append((f"{name} daily", d["Adj Close"].astype(float), None, "D"))
    ff = french.load()
    total = (ff["Mkt-RF"] + ff["RF"]) / 100.0
    idx = pd.period_range("1926-07", periods=len(ff), freq="M").to_timestamp()
    close = pd.Series(np.cumprod(1.0 + total.to_numpy()), index=idx)
    rf = pd.Series(ff["RF"].to_numpy() / 100.0, index=idx)
    series.append(("US market monthly (Fama-French)", close, rf, "M"))
    return series


def load_csv(path):
    df = pd.read_csv(path, parse_dates=["date"])
    groups = df.groupby("symbol") if "symbol" in df else [("prices", df)]
    return [(f"{sym} daily", g.set_index("date")["close"].astype(float).sort_index(), None, "D") for sym, g in groups]


def load_alpaca(symbols):
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(key, secret)
    df = None
    for feed in (DataFeed.SIP, DataFeed.IEX):  # the free plan serves SIP history; IEX if refused
        try:
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=pd.Timestamp("2016-01-01", tz="UTC"),
                                   end=pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=20), adjustment=Adjustment.ALL, feed=feed)
            df = client.get_stock_bars(req).df.reset_index()
            break
        except Exception as e:  # noqa: BLE001 - alpaca-py raises APIError subclasses
            print(f"[!] {feed.value} daily bars refused ({e}); trying the next feed")
    if df is None:
        raise SystemExit("[-] no daily bars from Alpaca (keys? APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
    out = []
    for sym in symbols:
        g = df[df["symbol"] == sym]
        out.append((f"{sym} daily (Alpaca, split/dividend adjusted)", pd.Series(g["close"].to_numpy(float),
                                                                              index=pd.to_datetime(g["timestamp"])), None, "D"))
    return out


# ----------------------------------------------------------------------------------------------
def fmt_table(rows, cols, first="rule", width=24):
    head = f"  {first:<{width}}" + "".join(f"{c:>14}" for c in cols)
    lines = [head]
    for name, m in rows:
        lines.append(f"  {name:<{width}}" + "".join(f"{m[c]:>14.2f}" if isinstance(m[c], float) else f"{m[c]:>14}" for c in cols))
    return "\n".join(lines)


def study(title, close, rf, freq, n_perm, rng, robustness=True):
    per_year = 252 if freq == "D" else 12
    block = 21 if freq == "D" else 6
    close = close.dropna()
    ret = close.pct_change().to_numpy()[1:]
    dates = close.index[1:]
    cash = np.zeros(len(ret)) if rf is None else rf.reindex(dates).fillna(0.0).to_numpy()
    rules = rules_for(close.to_numpy()[1:], ret, freq)
    warm = max(int(np.argmax(~np.isnan(w))) for w in rules.values())  # every rule live from here on
    warm += 1  # first period held under every rule
    s = slice(warm, None)
    print(f"\n=== {title}: {dates[warm].date()} .. {dates[-1].date()} ({len(ret) - warm} periods; cost {COST_BP:g} bp per "
          f"unit of turnover; cash {'= T-bill rate' if rf is not None else 'earns 0'})")

    results = {}
    for name, w in rules.items():
        r, held = strategy_returns(ret, w, cash)
        results[name] = (r[s], held[s])
    cols = ["CAGR %", "vol %", "Sharpe", "max DD %", "invested %", "switches/yr"]
    print(fmt_table([(n, metrics(r, cash[s], per_year, h)) for n, (r, h) in results.items()], cols))

    # Two halves: does it hold in both, or only in the crashes?
    mid = (len(ret) - warm) // 2
    for label, part in [("first half", slice(0, mid)), ("second half", slice(mid, None))]:
        d = dates[s][part]
        row = "  ".join(f"{n.split(' +')[0][:18]}={metrics(r[part], cash[s][part], per_year)['Sharpe']:.2f}"
                        for n, (r, _) in results.items())
        print(f"  Sharpe {label} ({d[0].year}-{d[-1].year}): {row}")

    if robustness:
        ns = [20, 50, 100, 150, 200, 250, 300] if freq == "D" else [3, 6, 8, 10, 12, 15, 18]
        row = []
        for n in ns:
            r, _ = strategy_returns(ret, above(close.to_numpy()[1:], n), cash)
            row.append(f"{n}:{metrics(r[s], cash[s], per_year)['Sharpe']:.2f}")
        print(f"  trend rule Sharpe by moving-average length ({'days' if freq == 'D' else 'months'}): {'  '.join(row)}"
              f"   (buy & hold {metrics(results['buy & hold'][0], cash[s], per_year)['Sharpe']:.2f})")

    # Monte Carlo 1: shuffled histories (no trends, no volatility clustering)
    null = shuffled_null(ret, float(close.iloc[0]), cash, freq, n_perm, rng, s)
    bh_r = results["buy & hold"][0]
    bh_s = metrics(bh_r, cash[s], per_year)["Sharpe"]
    # Monte Carlo 2: paired stationary bootstrap of the Sharpe improvement
    idx = stationary_bootstrap_idx(len(bh_r), len(bh_r), 2000, block, rng)
    bh_boot = sharpe((bh_r - cash[s])[idx], per_year)
    print(f"\n  Monte Carlo ({n_perm} shuffled histories; 2000 paired block-bootstrap resamples, mean block {block}):")
    print(f"  {'rule':<24}{'Sharpe gain':>12}{'shuffled p':>12}{'gain 90% CI':>18}{'P(gain>0)':>11}")
    for name, (r, _) in results.items():
        if name == "buy & hold":
            continue
        gain = metrics(r, cash[s], per_year)["Sharpe"] - bh_s
        p = (np.sum(null[name] >= gain) + 1) / (len(null[name]) + 1)
        boot = sharpe((r - cash[s])[idx], per_year) - bh_boot
        lo, hi = np.quantile(boot, [0.05, 0.95])
        print(f"  {name:<24}{gain:>+12.2f}{p:>12.3f}{f'[{lo:+.2f}, {hi:+.2f}]':>18}{100 * (boot > 0).mean():>10.0f}%")

    # Monte Carlo 3: risk of each rule over the next year
    cols = ["P(loss 3m) %", "P(loss 1y) %", "bad 1y (5%) %", "DD 1y median %", "DD 1y bad (95%) %"]
    print(f"\n  Risk over the next year (10000 simulated paths from each rule's own returns):")
    print(fmt_table([(n, risk_paths(r, per_year, block, rng)) for n, (r, _) in results.items()], cols))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--alpaca", help="comma-separated symbols: Alpaca daily bars since 2016 (needs APCA_API_KEY_ID/SECRET)")
    p.add_argument("--csv", type=Path, help="CSV with date, close[, symbol]")
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    if a.alpaca:
        data = load_alpaca(a.alpaca.split(","))
    elif a.csv:
        data = load_csv(a.csv)
    else:
        data = load_bundled()
    for title, close, rf, freq in data:
        study(title, close, rf, freq, a.permutations, rng)


if __name__ == "__main__":
    main()
