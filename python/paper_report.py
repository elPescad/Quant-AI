"""Performance of the Alpaca paper account from the engine's account.csv (written by --paper-orders).

    python python/paper_report.py out/account.csv [--capital 10000]

Daily returns use the last equity snapshot of each New York trading day. Returns are on the
strategy's capital (the engine sizes positions for $10k; the rest of the paper account is idle
cash), so the Sharpe ratio is the same either way. The uncertainty of an annualised Sharpe
ratio estimated from N days is about sqrt(252 / N): after two months, roughly +/- 2.
"""

import argparse
import csv
import math
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("account_csv", type=Path)
    p.add_argument("--capital", type=float, default=10000.0, help="capital the engine trades with (default 10000)")
    a = p.parse_args()

    rows = [r for r in csv.DictReader(a.account_csv.open()) if r["equity"]]
    if len(rows) < 2:
        raise SystemExit("[-] not enough snapshots yet")
    df = pd.DataFrame({"time": pd.to_datetime([r["time"] for r in rows], utc=True),
                       "equity": [float(r["equity"]) for r in rows]})
    df["day"] = df["time"].dt.tz_convert("America/New_York").dt.date
    daily = df.groupby("day")["equity"].last()
    pnl = daily.diff().dropna()
    if pnl.empty:
        raise SystemExit("[-] need at least two trading days")
    ret = pnl / a.capital
    n = len(ret)
    sharpe = ret.mean() / ret.std(ddof=1) * math.sqrt(252) if n > 1 and ret.std(ddof=1) > 0 else float("nan")
    curve = pd.concat([pd.Series([a.capital]), a.capital + pnl.cumsum()])  # from the starting capital
    drawdown = (curve / curve.cummax() - 1).min()
    se = math.sqrt(252 / n) if n else float("nan")

    print(f"Paper account: {daily.index[0]} .. {daily.index[-1]} ({n} trading days with a prior day)")
    total = df["equity"].iloc[-1] - df["equity"].iloc[0]  # from the first snapshot, like the engine's status summary
    print(f"  P&L                 ${total:,.2f} ({100 * total / a.capital:+.2f}% of ${a.capital:,.0f})")
    print(f"  daily mean / sd     {100 * ret.mean():+.3f}% / {100 * ret.std(ddof=1):.3f}%")
    print(f"  Sharpe (ann.)       {sharpe:.2f}  (standard error {se:.2f} after {n} days: within +/-{2 * se:.1f} of zero is "
          f"indistinguishable from luck)")
    print(f"  max drawdown        {100 * drawdown:.2f}%")
    print(f"  best / worst day    ${pnl.max():,.2f} / ${pnl.min():,.2f}")
    if n < 60:
        print(f"  [!] {n} days is too few to tell skill from luck; the band above shrinks with the square root of time")


if __name__ == "__main__":
    main()
