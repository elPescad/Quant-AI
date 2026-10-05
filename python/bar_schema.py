"""OHLCV bars -> the engine's raw tick CSV schema. Shared by every data source so that
Yahoo, Alpaca, etc. produce identical features and labels from identical bars."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

LABEL_HORIZON = 6        # Bars ahead used for the label (30 minutes on 5m bars)
FEE_HURDLE_PCT = 0.0004  # 4 bps return threshold for BUY / SELL labels
COLUMNS = ["timestamp", "ticker", "raw_price", "raw_spread", "raw_ofi", "raw_delta", "raw_vol", "target"]


def epoch_seconds(index):
    """UTC epoch seconds, independent of the index's datetime resolution (ns vs us)."""
    idx = pd.to_datetime(index, utc=True)
    return ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy().astype(np.int64)


def regular_session_mask(index):
    """True for 5-minute bars that start inside 09:30-16:00 New York time."""
    ny = pd.to_datetime(index, utc=True).tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return np.asarray((ny.dayofweek < 5) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60))


def bars_to_ticks(symbol, timestamps, high, low, close, volume):
    close = np.asarray(close, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    price_delta = np.diff(close, prepend=close[0])
    # High-low range of the bar. The engine caps the fill cost it implies (see portfolio.hpp).
    spread = np.clip(high - low, 0.01, 2.0)
    ofi = np.sign(price_delta) * np.log1p(volume)
    volatility = np.abs(price_delta)

    future_return = np.full(len(close), np.nan)
    future_return[:-LABEL_HORIZON] = close[LABEL_HORIZON:] / close[:-LABEL_HORIZON] - 1.0

    target = np.ones(len(close), dtype=np.int64)  # HOLD (1)
    target[future_return > FEE_HURDLE_PCT] = 2     # BUY (2)
    target[future_return < -FEE_HURDLE_PCT] = 0    # SELL (0)
    target[np.isnan(future_return)] = -1           # Unknown: no future bars yet

    return pl.DataFrame({
        "timestamp": np.asarray(timestamps, dtype=np.int64),
        "ticker": symbol,
        "raw_price": close.astype(np.float32),
        "raw_spread": spread.astype(np.float32),
        "raw_ofi": ofi.astype(np.float32),
        "raw_delta": price_delta.astype(np.float32),
        "raw_vol": volatility.astype(np.float32),
        "target": target,
    })


def write_ticks(frames, ticker_order, output_file):
    """Interleave tickers bar by bar so the engine sees the whole cross-section of each bar."""
    if not frames:
        raise SystemExit("[-] No data downloaded (network blocked or tickers unavailable)")
    order = {t: i for i, t in enumerate(ticker_order)}
    combined = (
        pl.concat(frames)
        .with_columns(pl.col("ticker").replace_strict(order, return_dtype=pl.Int32).alias("_order"))
        .sort(["timestamp", "_order"])
        .drop("_order")
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    combined.write_csv(output_file)
    days = len(np.unique(pd.to_datetime(combined["timestamp"].to_numpy(), unit="s", utc=True)
                         .tz_convert("America/New_York").date))
    print(f"[+] Exported {len(combined):,} ticks ({combined['ticker'].n_unique()} tickers, {days} trading days) to: {output_file}")
