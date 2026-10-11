"""Which prediction horizon and which inputs carry an edge after costs? A fast walk-forward study
on the engine's 5-minute bar CSV, to decide what the GRU should predict, and from what, before
spending hours retraining it.

    python python/signal_lab.py --data data/alpaca_iex.csv
    python python/signal_lab.py --data data/alpaca_sip_5y.csv --horizons 6 12 24 48 close

Horizons are in bars (6 = 30 minutes, the GRU's current label) or "close" (to the session's last
bar). For each one:
  1. Information per input: the daily rank correlation (IC) of each input with the return over
     the horizon, averaged over days, with its t-statistic. |t| above 3 deserves attention;
     below 2 is indistinguishable from noise.
  2. Walk-forward model: ridge regression on all inputs, refit on everything before each fold
     and scored on the fold (out of sample), with a cost-aware rule: trade when the predicted
     move beats k x the round-trip cost. Net basis points per trade, trades per day and the
     Sharpe ratio of the daily P&L, next to always-long for the same horizon (market drift).
  3. Where the edge lives: the same trades split by regime (ticker above / below its 50-day
     average, high / low volatility).
Inputs are price-based, so they mean the same on IEX and SIP bars and you can study (and later
train on) years of SIP history: returns over 1, 6 and 12 bars, today, 1 and 5 days (in units of
recent volatility), the overnight gap, time of day, short/long volatility ratio, the bar's range,
distance from the 20- and 50-day averages, the market's (SPY's) moves and the ticker's move
relative to it. signed_volume (the engine's OFI input) depends on the feed's volume; it is
included for comparison only.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEE = 0.2e-4             # per fill, as the engine
MAX_PROXY_COST = 5e-4    # engine: the high-low range proxy for the spread is capped at 5 bp
BARS_PER_DAY = 78


def ewm_mean(x, span):
    """Causal EWMA per column, skipping NaN (pandas adjust=False, ignore_na=True)."""
    return pd.DataFrame(x).ewm(span=span, adjust=False, ignore_na=True).mean().to_numpy()


def build(path):
    cols = pd.read_csv(path, nrows=0).columns
    want = ["timestamp", "ticker", "raw_price", "raw_spread", "raw_ofi"] + (["quoted_spread"] if "quoted_spread" in cols else [])
    df = pd.read_csv(path, usecols=want)
    wide = {c: df.pivot_table(index="timestamp", columns="ticker", values=c, aggfunc="last") for c in want[2:]}
    tickers = list(wide["raw_price"].columns)
    present = wide["raw_price"].notna().to_numpy()
    C = wide["raw_price"].ffill().to_numpy(np.float64)
    H = wide["raw_spread"].reindex(columns=tickers).ffill().to_numpy(np.float64)
    Q = (wide["quoted_spread"].reindex(columns=tickers).to_numpy(np.float64) if "quoted_spread" in wide
         else np.full_like(C, -1.0))
    OFI = wide["raw_ofi"].reindex(columns=tickers).to_numpy(np.float64)
    ts = wide["raw_price"].index.to_numpy()
    ny = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York")
    date = np.asarray(ny.date)
    _, day = np.unique(date, return_inverse=True)
    first = np.r_[True, day[1:] != day[:-1]]
    last = np.r_[day[1:] != day[:-1], True]
    bar_idx = np.asarray((ny.hour * 60 + ny.minute - 570) // 5)
    return dict(ts=ts, day=day, first=first, last=last, bar_idx=bar_idx, tickers=tickers, present=present,
                C=C, H=H, Q=Q, OFI=OFI)


def shift(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def features(d):
    C, day, first = d["C"], d["day"], d["first"]
    T, N = C.shape
    lc = np.log(C)
    lr = lc - shift(lc, 1)
    intraday = np.where(first[:, None], np.nan, lr)
    var5 = ewm_mean(intraday ** 2, 390)            # ~5 days of 5-minute variance
    var_short = ewm_mean(intraday ** 2, 12)        # ~1 hour
    s5 = np.sqrt(var5)
    s5[s5 <= 0] = np.nan

    def scaled(k):
        return (lc - shift(lc, k)) / (s5 * math.sqrt(k))

    # Session-relative quantities
    first_idx = np.maximum.accumulate(np.where(first, np.arange(T), 0))
    k_today = np.arange(T) - first_idx
    lc_first = lc[first_idx]
    prev_last = np.where(first, np.arange(T) - 1, -1)
    prev_last = np.maximum.accumulate(np.where(first, prev_last, -1))
    lc_prev_close = np.where(prev_last[:, None] >= 0, lc[np.maximum(prev_last, 0)], np.nan)

    # Daily closes and their averages over completed days (no look-ahead)
    n_days = day.max() + 1
    last_row = np.zeros(n_days, dtype=np.int64)
    last_row[day] = np.arange(T)  # last row of each day
    daily_close = pd.DataFrame(C[last_row])
    sma = {n: np.log(daily_close.rolling(n).mean().shift(1).to_numpy())[day] for n in (20, 50)}
    s_daily = s5 * math.sqrt(BARS_PER_DAY)

    f = {
        "r_1bar": lr / s5,
        "r_30m": scaled(6),
        "r_1h": scaled(12),
        "r_today": np.where(k_today[:, None] > 0, (lc - lc_first) / (s5 * np.sqrt(np.maximum(k_today, 1))[:, None]), 0.0),
        "gap": (lc_first - lc_prev_close) / s_daily,
        "r_1d": scaled(BARS_PER_DAY),
        "r_5d": scaled(5 * BARS_PER_DAY),
        "time_of_day": np.repeat((d["bar_idx"] / 77.0)[:, None], N, axis=1),
        "vol_ratio": 0.5 * np.log(var_short / var5),
        "range": np.log(np.maximum(d["H"], 1e-6) / C) - ewm_mean(np.log(np.maximum(d["H"], 1e-6) / C), 390),
        "trend_20d": (lc - sma[20]) / s_daily,
        "trend_50d": (lc - sma[50]) / s_daily,
    }
    # Market context: SPY if present, else the equal-weight basket
    mk = d["tickers"].index("SPY") if "SPY" in d["tickers"] else None
    for name, src in [("mkt_30m", "r_30m"), ("mkt_today", "r_today"), ("mkt_1d", "r_1d")]:
        col = f[src][:, mk] if mk is not None else np.nanmean(f[src], axis=1)
        f[name] = np.repeat(col[:, None], N, axis=1)
    f["rel_30m"] = f["r_30m"] - f["mkt_30m"]
    ofi = d["OFI"]
    m = ewm_mean(ofi, 500)
    sd = np.sqrt(np.maximum(ewm_mean((ofi - m) ** 2, 500), 1e-12))
    f["signed_volume*"] = np.clip((ofi - m) / sd, -4, 4)
    for k in f:
        f[k] = np.clip(np.nan_to_num(f[k], nan=0.0, posinf=0.0, neginf=0.0), -6, 6)
    return f, s5, sma[50]


def targets(d, h):
    """Forward log return from this bar's close: h bars ahead, or to the session's last bar."""
    lc = np.log(d["C"])
    T = len(lc)
    if h == "close":
        last_idx = np.minimum.accumulate(np.where(d["last"], np.arange(T), T)[::-1])[::-1]
        fwd = lc[last_idx] - lc
        fwd[d["last"]] = np.nan
        bars = (last_idx - np.arange(T)).astype(float)
        bars[bars == 0] = np.nan
        return fwd, np.repeat(bars[:, None], lc.shape[1], axis=1)
    fwd = shift(lc, -h) - lc
    return fwd, np.full_like(lc, float(h))


def round_trip_cost(d):
    C, H, Q = d["C"], d["H"], d["Q"]
    half = np.where(Q > 0, Q / (2 * C), 0.5 * np.minimum(H / C, MAX_PROXY_COST))
    return 2 * (half + FEE)


def daily_ic(day, x, y):
    """Mean and t-stat of the per-day Spearman correlation."""
    df = pd.DataFrame({"d": day, "x": x, "y": y})
    g = df.groupby("d")
    rx = g["x"].rank()
    ry = g["y"].rank()
    cx = rx - rx.groupby(df["d"]).transform("mean")
    cy = ry - ry.groupby(df["d"]).transform("mean")
    num = (cx * cy).groupby(df["d"]).sum()
    den = np.sqrt((cx ** 2).groupby(df["d"]).sum() * (cy ** 2).groupby(df["d"]).sum())
    ic = (num / den.replace(0, np.nan)).dropna()
    if len(ic) < 3:
        return float("nan"), float("nan")
    return float(ic.mean()), float(ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic)))


def ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) / sd) @ beta + y.mean(), beta


def stats(day_of_trade, net, days):
    """trades/day, net bp/trade, hit rate, Sharpe of daily P&L (all OOS days, flat days = 0)."""
    if len(net) == 0:
        return 0.0, float("nan"), float("nan"), 0.0
    pnl = pd.Series(net).groupby(day_of_trade).sum().reindex(days, fill_value=0.0).to_numpy()
    sr = pnl.mean() / pnl.std(ddof=1) * math.sqrt(252) if pnl.std() > 0 else 0.0
    return len(net) / len(days), 1e4 * net.mean(), 100 * np.mean(net > 0), sr


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--horizons", nargs="+", default=["1", "6", "12", "24", "48", "close"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--oos-frac", type=float, default=0.6, help="share of days scored out of sample (default 0.6)")
    p.add_argument("--ridge", type=float, default=0.01, help="ridge penalty per sample (default 0.01)")
    p.add_argument("--k", type=float, nargs="+", default=[1.0, 2.0, 3.0], help="trade when |predicted move| > k x round-trip cost")
    a = p.parse_args()

    d = build(a.data)
    f, s5, sma50 = features(d)
    names = list(f)
    cost = round_trip_cost(d)
    T, N = d["C"].shape
    day2 = np.repeat(d["day"][:, None], N, axis=1)
    n_days = d["day"].max() + 1
    warm = d["day"] >= 5  # 5 days for the volatility estimate
    base = d["present"] & np.isfinite(s5) & warm[:, None]
    print(f"[+] {a.data.name}: {T} bars x {N} tickers ({', '.join(d['tickers'])}), {n_days} trading days | "
          f"median round-trip cost {1e4 * np.nanmedian(cost[base]):.2f} bp "
          f"({'measured quotes' if (d['Q'] > 0).mean() > 0.5 else 'high-low proxy, capped'})")
    horizons = [h if h == "close" else int(h) for h in a.horizons]
    label = {1: "5m", 6: "30m", 12: "1h", 24: "2h", 48: "4h", 78: "1d", "close": "close"}

    # 1. Information per input
    print(f"\n1. Information per input: daily rank IC x 100 (t-stat) with the return over each horizon; * = feed-dependent")
    print(f"   {'input':<16}" + "".join(f"{label.get(h, str(h)):>14}" for h in horizons))
    fwd_by_h = {h: targets(d, h) for h in horizons}
    for name in names:
        cells = []
        for h in horizons:
            fwd, _ = fwd_by_h[h]
            ok = base & np.isfinite(fwd)
            ic, t = daily_ic(day2[ok], f[name][ok], fwd[ok])
            mark = "!" if abs(t) >= 3 else " "
            cells.append(f"{100 * ic:>7.2f}({t:>+5.1f}){mark}")
        print(f"   {name:<16}" + "".join(f"{c:>14}" for c in cells))
    print("   (! = |t| >= 3. With 17 inputs x several horizons, a few |t| near 2 are expected by chance alone.)")

    # 2. Walk-forward ridge
    days = np.arange(n_days)
    oos_start = int(n_days * (1 - a.oos_frac))
    edges = [oos_start + (n_days - oos_start) * k // a.folds for k in range(a.folds + 1)]
    X_all = np.stack([f[n] for n in names], axis=-1)
    print(f"\n2. Walk-forward ridge on all inputs: {a.folds} folds over the last {n_days - oos_start} days, each fit only on "
          f"days before it; trade when |predicted move| > k x round-trip cost")
    head = "".join(f"{'k=' + format(k, 'g') + ': trades/d  net bp  hit%  Sharpe':>40}" for k in a.k)
    print(f"   {'horizon':<8}{'OOS IC x100 (t)':>17}{'long-only net bp':>18}{head}")
    best = None
    for h in horizons:
        fwd, bars = fwd_by_h[h]
        ok = base & np.isfinite(fwd)
        scale = s5 * np.sqrt(bars)
        y = np.clip(fwd / scale, -5, 5)
        pred = np.full_like(fwd, np.nan)
        purge = 1 if h == "close" else max(1, math.ceil(h / BARS_PER_DAY) + 1)
        for k in range(a.folds):
            lo, hi = edges[k], edges[k + 1]
            tr = ok & (day2 < lo - purge)
            te = ok & (day2 >= lo) & (day2 < hi)
            if tr.sum() < 1000 or te.sum() == 0:
                continue
            model, beta = ridge(X_all[tr], y[tr], a.ridge)
            pred[te] = model(X_all[te]) * scale[te]
        te = ok & np.isfinite(pred) & (day2 >= oos_start)
        oos_days = days[oos_start:]
        ic, t = daily_ic(day2[te], pred[te], fwd[te])
        long_net = 1e4 * np.mean(fwd[te] - cost[te])
        cells = []
        for k in a.k:
            tr_mask = te & (np.abs(pred) > k * cost)
            net = np.sign(pred[tr_mask]) * fwd[tr_mask] - cost[tr_mask]
            tpd, bp, hit, sr = stats(day2[tr_mask], net, oos_days)
            cells.append(f"{tpd:>17.1f}{bp:>8.2f}{hit:>6.0f}{sr:>9.2f}")
            if best is None or (np.isfinite(sr) and sr > best[0] and tpd > 0.2):
                best = (sr, h, k)
        print(f"   {label.get(h, str(h)):<8}{100 * ic:>9.2f} ({t:>+5.1f}){long_net:>18.2f}" + "".join(cells))
        top = np.argsort(-np.abs(beta))[:4]
        print(f"   {'':<8}  strongest inputs (last fold): " + ", ".join(f"{names[i]} {beta[i]:+.3f}" for i in top))
    print("   (net bp: per trade after the measured spread and fees; Sharpe: of the daily P&L, flat days count as 0;\n"
          "    long-only: always long for that horizon, i.e. the market's drift minus costs)")

    # 3. Where the edge lives
    if best is not None:
        _, h, k = best
        fwd, bars = fwd_by_h[h]
        ok = base & np.isfinite(fwd)
        scale = s5 * np.sqrt(bars)
        y = np.clip(fwd / scale, -5, 5)
        pred = np.full_like(fwd, np.nan)
        purge = 1 if h == "close" else max(1, math.ceil(h / BARS_PER_DAY) + 1)
        for j in range(a.folds):
            lo, hi = edges[j], edges[j + 1]
            tr = ok & (day2 < lo - purge)
            te = ok & (day2 >= lo) & (day2 < hi)
            if tr.sum() >= 1000 and te.sum():
                pred[te] = ridge(X_all[tr], y[tr], a.ridge)[0](X_all[te]) * scale[te]
        traded = ok & np.isfinite(pred) & (np.abs(pred) > k * cost)
        net = np.where(traded, np.sign(pred) * fwd - cost, np.nan)
        up = np.log(d["C"]) > sma50
        known = np.isfinite(sma50)
        hi_vol = s5 > np.nanmedian(s5[base])
        print(f"\n3. Where the edge lives ({label.get(h, h)}, k={k:g}): net bp per trade (trades) by regime")
        for name, mask in [("uptrend (above 50d avg)", known & up), ("downtrend", known & ~up),
                           ("high volatility", hi_vol), ("low volatility", ~hi_vol)]:
            m = traded & mask
            longs = m & (pred > 0)
            shorts = m & (pred < 0)
            fmt = lambda mm: f"{1e4 * np.nanmean(net[mm]):+7.2f} ({int(mm.sum())})" if mm.sum() else "      - (0)"
            print(f"   {name:<26} all {fmt(m):>16}   longs {fmt(longs):>16}   shorts {fmt(shorts):>16}")
        print("   (a rule that only takes the trades where the edge lives must still be tested out of sample)")


if __name__ == "__main__":
    main()
