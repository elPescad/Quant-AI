"""Synthetic multi-ticker 5-minute bars in the same RAW schema as fetch_real_ticks.py.

Ground truth built into the data (so the pipeline can be checked end to end):
  * Order-flow imbalance (OFI) is AR(1) and its persistence phi switches between a
    trending regime (0.9) and a choppy regime (0.3).
  * In the trending regime OFI predicts returns with a positive sign (momentum);
    in the choppy regime the sign flips (mean reversion). A model can only exploit
    this if it combines OFI with an estimate of phi - an interaction that is not
    linearly separable in the raw inputs.
  * All tickers share a market factor, so their returns are correlated.

This is a sanity check for the machinery, not evidence about real markets.
"""

import numpy as np
import polars as pl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "market_ticks.csv"

TICKERS = {"SPY": 550.0, "QQQ": 480.0, "AAPL": 225.0, "NVDA": 120.0, "MSFT": 430.0, "AMD": 150.0}
BETAS = {"SPY": 1.0, "QQQ": 1.15, "AAPL": 1.1, "NVDA": 1.6, "MSFT": 1.0, "AMD": 1.5}
LABEL_HORIZON = 6        # Must match fetch_real_ticks.py
FEE_HURDLE_PCT = 0.0004  # Must match fetch_real_ticks.py
START_TS = 1_735_740_000  # Arbitrary epoch start; bars are 300 s apart


def regime_path(num_bars, rng, mean_dwell=600):
    """Alternating persistence regimes with random dwell times. True = trending."""
    trending = np.zeros(num_bars, dtype=bool)
    t, state = 0, bool(rng.integers(2))
    while t < num_bars:
        dwell = int(rng.exponential(mean_dwell)) + 100
        trending[t:t + dwell] = state
        state = not state
        t += dwell
    return trending


def generate_ticker(symbol, start_price, market, num_bars, rng):
    trending = regime_path(num_bars, rng)
    phi = np.where(trending, 0.9, 0.3)

    # Unit-variance AR(1) order flow with regime-dependent persistence
    ofi = np.zeros(num_bars)
    noise = rng.normal(0, 1.0, num_bars)
    for t in range(1, num_bars):
        ofi[t] = phi[t] * ofi[t - 1] + np.sqrt(1 - phi[t] ** 2) * noise[t]

    idio_vol = 0.0007
    alpha = 0.00018  # Return per 1-sigma of lagged OFI
    sign = np.where(trending, 1.0, -0.8)
    returns = BETAS[symbol] * market + idio_vol * rng.normal(0, 1, num_bars)
    returns[1:] += alpha * sign[1:] * ofi[:-1]
    close = start_price * np.exp(np.cumsum(returns))

    price_delta = np.diff(close, prepend=close[0])
    spread = np.clip(close * rng.normal(0.0001, 0.00003, num_bars), 0.01, None)  # ~1 bp quoted spread
    raw_ofi = ofi * 3.0 + 0.5 * rng.normal(0, 1, num_bars)  # Observed with noise, arbitrary scale
    volatility = np.abs(price_delta)

    future_return = np.full(num_bars, np.nan)
    future_return[:-LABEL_HORIZON] = close[LABEL_HORIZON:] / close[:-LABEL_HORIZON] - 1.0
    target = np.ones(num_bars, dtype=np.int64)
    target[future_return > FEE_HURDLE_PCT] = 2
    target[future_return < -FEE_HURDLE_PCT] = 0
    target[np.isnan(future_return)] = -1

    return pl.DataFrame({
        "timestamp": (START_TS + 300 * np.arange(num_bars)).astype(np.int64),
        "ticker": symbol,
        "raw_price": close.astype(np.float32),
        "raw_spread": spread.astype(np.float32),
        "raw_ofi": raw_ofi.astype(np.float32),
        "raw_delta": price_delta.astype(np.float32),
        "raw_vol": volatility.astype(np.float32),
        "target": target,
    })


def generate_market_ticks(bars_per_ticker=5000, seed=42):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.0005, bars_per_ticker)
    frames = [generate_ticker(sym, px, market, bars_per_ticker, rng) for sym, px in TICKERS.items()]

    order = {t: i for i, t in enumerate(TICKERS)}
    df = (
        pl.concat(frames)
        .with_columns(pl.col("ticker").replace_strict(order, return_dtype=pl.Int32).alias("_order"))
        .sort(["timestamp", "_order"])
        .drop("_order")
    )
    df.write_csv(OUTPUT_FILE)
    print(f"[+] Saved {len(df):,} synthetic raw ticks ({bars_per_ticker} bars x {len(TICKERS)} tickers) to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_market_ticks()
