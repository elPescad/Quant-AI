import numpy as np
import polars as pl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "market_ticks.csv"

# Synthetic tickers and their starting prices
TICKERS = {"SPY": 550.0, "QQQ": 480.0, "AAPL": 225.0, "NVDA": 120.0, "MSFT": 430.0, "AMD": 150.0}
LABEL_HORIZON = 6        # Must match fetch_real_ticks.py
FEE_HURDLE_PCT = 0.0004  # Must match fetch_real_ticks.py


def generate_ticker(symbol, start_price, num_ticks, rng):
    # 1. Order Flow Imbalance via AR(1) process (Momentum/Persistence)
    phi = 0.85
    ofi = np.zeros(num_ticks)
    noise = rng.normal(0, 1.0, num_ticks)
    for t in range(1, num_ticks):
        ofi[t] = phi * ofi[t - 1] + noise[t]

    # 2. Price path: returns partially driven by lagged OFI (the alpha the model should learn)
    base_vol = 0.0008  # ~8 bps per bar
    returns = 0.00025 * ofi / ofi.std() + rng.normal(0, base_vol, num_ticks)
    returns = np.roll(returns, 1)
    returns[0] = 0.0
    close = start_price * np.exp(np.cumsum(returns))

    # 3. Microstructure features in the same RAW schema as fetch_real_ticks.py
    price_delta = np.diff(close, prepend=close[0])
    spread = np.clip(close * rng.normal(0.0001, 0.00003, num_ticks), 0.01, None)  # ~1 bp quoted spread
    raw_ofi = ofi
    volatility = np.abs(price_delta)

    future_return = (np.roll(close, -LABEL_HORIZON) - close) / close
    future_return[-LABEL_HORIZON:] = 0.0
    target = np.ones(num_ticks, dtype=np.int64)  # HOLD (1)
    target[future_return > FEE_HURDLE_PCT] = 2    # BUY (2)
    target[future_return < -FEE_HURDLE_PCT] = 0   # SELL (0)

    return pl.DataFrame({
        "ticker": symbol,
        "raw_price": close.astype(np.float32),
        "raw_spread": spread.astype(np.float32),
        "raw_ofi": raw_ofi.astype(np.float32),
        "raw_delta": price_delta.astype(np.float32),
        "raw_vol": volatility.astype(np.float32),
        "target": target,
    })


def generate_market_ticks(ticks_per_ticker=5000, seed=42):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = [generate_ticker(sym, px, ticks_per_ticker, rng) for sym, px in TICKERS.items()]
    df = pl.concat(frames)
    df.write_csv(OUTPUT_FILE)
    print(f"[+] Saved {len(df):,} synthetic raw ticks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_market_ticks()
