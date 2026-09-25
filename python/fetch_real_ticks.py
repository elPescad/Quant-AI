import yfinance as yf
import polars as pl
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "market_ticks.csv"

TARGET_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMD"]
FEE_HURDLE_PCT = 0.0004  # 8 bps return threshold for target labeling

def generate_raw_tick_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Exporting RAW un-normalized market data for: {TARGET_TICKERS}")

    all_frames = []

    for symbol in TARGET_TICKERS:
        print(f"    Downloading {symbol}...")
        df_raw = yf.download(symbol, period="60d", interval="5m", progress=False)

        if df_raw.empty:
            continue

        close = df_raw["Close"].to_numpy().flatten()
        volume = df_raw["Volume"].to_numpy().flatten()
        high = df_raw["High"].to_numpy().flatten()
        low = df_raw["Low"].to_numpy().flatten()

        price_delta = np.diff(close, prepend=close[0])
        spread = np.clip(high - low, 0.01, 2.0)
        
        ofi = np.sign(price_delta) * np.log1p(volume)
        volatility = np.abs(price_delta)

        # Target Labeling based on real relative future return
        future_return = (np.roll(close, -6) - close) / close
        future_return[-6:] = 0.0

        target = np.ones(len(close), dtype=int) # HOLD (1)
        target[future_return > FEE_HURDLE_PCT] = 2   # BUY (2)
        target[future_return < -FEE_HURDLE_PCT] = 0  # SELL (0)

        df_sym = pl.DataFrame({
            "ticker": symbol,
            "raw_price": close.astype(np.float32),
            "raw_spread": spread.astype(np.float32),
            "raw_ofi": ofi.astype(np.float32),
            "raw_delta": price_delta.astype(np.float32),
            "raw_vol": volatility.astype(np.float32),
            "target": target.astype(np.int64)
        })

        all_frames.append(df_sym)

    combined_df = pl.concat(all_frames)
    combined_df.write_csv(OUTPUT_FILE)
    print(f"[+] Successfully exported {len(combined_df):,} raw market ticks to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_raw_tick_dataset()