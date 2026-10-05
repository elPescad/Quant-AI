import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "market_ticks.csv"

TARGET_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMD"]
LABEL_HORIZON = 6        # Bars ahead used for the label (30 minutes on 5m bars)
FEE_HURDLE_PCT = 0.0004  # 4 bps return threshold for BUY / SELL labels


def generate_raw_tick_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Exporting RAW un-normalized market data for: {TARGET_TICKERS}")

    all_frames = []
    for symbol in TARGET_TICKERS:
        print(f"    Downloading {symbol}...")
        df_raw = yf.download(symbol, period="60d", interval="5m", progress=False)
        if df_raw.empty:
            print(f"    [!] No data for {symbol}, skipping")
            continue

        # UTC epoch seconds, independent of the index's datetime resolution (ns vs us)
        idx = pd.to_datetime(df_raw.index, utc=True)
        timestamps = ((idx - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy().astype(np.int64)
        close = df_raw["Close"].to_numpy().flatten()
        volume = df_raw["Volume"].to_numpy().flatten()
        high = df_raw["High"].to_numpy().flatten()
        low = df_raw["Low"].to_numpy().flatten()

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

        all_frames.append(pl.DataFrame({
            "timestamp": timestamps,
            "ticker": symbol,
            "raw_price": close.astype(np.float32),
            "raw_spread": spread.astype(np.float32),
            "raw_ofi": ofi.astype(np.float32),
            "raw_delta": price_delta.astype(np.float32),
            "raw_vol": volatility.astype(np.float32),
            "target": target,
        }))

    if not all_frames:
        raise SystemExit("[-] No data downloaded (network blocked or tickers unavailable)")

    # Interleave tickers bar by bar so the engine sees the whole cross-section of each bar
    ticker_order = {t: i for i, t in enumerate(TARGET_TICKERS)}
    combined_df = (
        pl.concat(all_frames)
        .with_columns(pl.col("ticker").replace_strict(ticker_order, return_dtype=pl.Int32).alias("_order"))
        .sort(["timestamp", "_order"])
        .drop("_order")
    )
    combined_df.write_csv(OUTPUT_FILE)
    print(f"[+] Successfully exported {len(combined_df):,} raw market ticks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_raw_tick_dataset()
