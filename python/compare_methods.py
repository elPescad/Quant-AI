"""Train on a tick CSV, then backtest every feature-set x allocator combination.

    python python/compare_methods.py --data data/yahoo_ticks.csv
    python python/compare_methods.py --data data/alpaca_ticks.csv

The method is chosen by validation Sharpe; test (held out, touched once) is reported for all.
Results also go to <workdir>/results.csv.
"""

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METHODS = [("quantum", "quant_model", "qubo"), ("quantum", "quant_model", "greedy"),
           ("raw", "baseline_model", "qubo"), ("raw", "baseline_model", "greedy")]
METRICS = {
    "return_pct": r"Total Net PnL:.*\(([-\d.e+]+)%\)",
    "sharpe": r"Sharpe Ratio \(ann.\):\s+([-\d.e+]+)",
    "max_dd_pct": r"Max Drawdown:\s+([-\d.e+]+)",
    "round_trips": r"Total Round Trips:\s+(\d+)",
    "exposure_pct": r"Avg Gross Exposure:\s+([-\d.e+]+)",
}


def backtest(engine, data, model, allocator, period, workdir):
    tag = f"{period}_{model}_{allocator}"
    cmd = [str(engine), "--data", str(data), "--model", str(workdir / "models" / f"{model}.pt"),
           "--allocator", allocator, "--period", period, "--trades", str(workdir / f"trades_{tag}.csv")]
    out = subprocess.run(cmd, capture_output=True, text=True)
    (workdir / f"bt_{tag}.log").write_text(out.stdout + out.stderr)
    if out.returncode != 0:
        sys.exit(f"[-] {' '.join(cmd)} failed (full log: {workdir / f'bt_{tag}.log'}):\n{out.stderr[:1500]}")
    row = {}
    for key, pat in METRICS.items():
        m = re.search(pat, out.stdout)
        if not m:
            sys.exit(f"[-] could not parse '{key}' from {workdir / f'bt_{tag}.log'}")
        row[key] = float(m.group(1))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--engine", type=Path, default=PROJECT_ROOT / "build" / "quant_engine")
    p.add_argument("--workdir", type=Path, help="default runs/<data file stem>")
    p.add_argument("--skip-train", action="store_true", help="reuse models already in <workdir>/models")
    a = p.parse_args()
    data = a.data.resolve()
    workdir = (a.workdir or PROJECT_ROOT / "runs" / data.stem).resolve()
    (workdir / "models").mkdir(parents=True, exist_ok=True)

    if not a.skip_train:
        print(f"[+] Training on {data} -> {workdir / 'models'}")
        log = workdir / "train.log"
        with log.open("w") as f:
            r = subprocess.run([sys.executable, str(PROJECT_ROOT / "python" / "train_and_export.py"),
                                "--data", str(data), "--model-dir", str(workdir / "models")], stdout=f, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f"[-] training failed, see {log}")
        print((workdir / "models" / "training_report.txt").read_text())

    rows = []
    for feats, model, alloc in METHODS:
        for period in ("val", "test"):
            print(f"    backtest {feats:>7} + {alloc:<6} on {period}")
            rows.append({"method": f"{feats} + {alloc}", "period": period,
                         **backtest(a.engine.resolve(), data, model, alloc, period, workdir)})

    with (workdir / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    val = {r["method"]: r for r in rows if r["period"] == "val"}
    test = {r["method"]: r for r in rows if r["period"] == "test"}
    ranked = sorted(val, key=lambda m: val[m]["sharpe"], reverse=True)
    print(f"\nResults for {data.name} (ranked by validation Sharpe)")
    print(f"{'method':<18}{'val Sharpe':>12}{'test Sharpe':>13}{'test ret %':>12}{'test maxDD %':>14}{'trips':>7}{'expo %':>8}")
    for m in ranked:
        t = test[m]
        print(f"{m:<18}{val[m]['sharpe']:>12.2f}{t['sharpe']:>13.2f}{t['return_pct']:>12.2f}"
              f"{t['max_dd_pct']:>14.2f}{t['round_trips']:>7.0f}{t['exposure_pct']:>8.1f}")
    best = ranked[0]
    verdict = "POSITIVE" if test[best]["sharpe"] > 0 and test[best]["return_pct"] > 0 else "NEGATIVE"
    print(f"\nSelected on validation: {best} -> test Sharpe {test[best]['sharpe']:.2f} ({verdict})")


if __name__ == "__main__":
    main()
