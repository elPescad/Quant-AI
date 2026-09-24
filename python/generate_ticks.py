import os
import numpy as np
import polars as pl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "market_ticks.csv"

def generate_market_ticks(num_ticks=500000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Generating {num_ticks:,} microstructural ticks with persistent alpha...")

    np.random.seed(42)

    # 1. Generate Order Flow Imbalance via AR(1) process (Momentum/Persistence)
    ofi = np.zeros(num_ticks, dtype=np.float32)
    phi = 0.85 # Autoregressive momentum coefficient
    noise_ofi = np.random.normal(0, 0.2, num_ticks)

    for t in range(1, num_ticks):
        ofi[t] = phi * ofi[t-1] + noise_ofi[t]

    ofi = np.clip(ofi, -1.0, 1.0)

    # 2. Derive Microstructure Features
    spread = np.clip(np.abs(np.random.normal(0.02, 0.005, num_ticks)), 0.005, 0.10).astype(np.float32)
    volatility = (np.abs(ofi) * 0.05 + np.random.normal(0.01, 0.002, num_ticks)).astype(np.float32)

    # 3. Microstructure Alpha Engine: Future delta is directly driven by current OFI & Spread compression
    future_delta = (0.06 * ofi) - (0.02 * spread) + np.random.normal(0, 0.008, num_ticks)

    df = pl.DataFrame({
        "spread": spread,
        "ofi": ofi,
        "price_delta_5": future_delta.astype(np.float32),
        "volatility_20": volatility
    })

    # 4. Feature Engineering: Smooth OFI & Z-Score Normalization
    df = df.with_columns([
        pl.col("ofi").rolling_mean(window_size=5, min_samples=1).alias("ofi_sma_5")
    ])

    df = df.with_columns([
        ((pl.col("spread") - pl.col("spread").mean()) / pl.col("spread").std()).alias("norm_spread"),
        ((pl.col("ofi_sma_5") - pl.col("ofi_sma_5").mean()) / pl.col("ofi_sma_5").std()).alias("norm_ofi"),
        ((pl.col("price_delta_5") - pl.col("price_delta_5").mean()) / pl.col("price_delta_5").std()).alias("norm_delta"),
        ((pl.col("volatility_20") - pl.col("volatility_20").mean()) / pl.col("volatility_20").std()).alias("norm_vol")
    ])

    # 5. Class Labeling: Equal distribution across quantiles
    p33 = np.percentile(future_delta, 33)
    p66 = np.percentile(future_delta, 66)

    target = np.ones(num_ticks, dtype=int) # HOLD (1)
    target[future_delta > p66] = 2         # BUY (2)
    target[future_delta < p33] = 0         # SELL (0)

    final_df = pl.DataFrame({
        "spread": df["norm_spread"],
        "order_imbalance": df["norm_ofi"],
        "price_delta_5": df["norm_delta"],
        "volatility_20": df["norm_vol"],
        "target": target.astype(np.int64)
    })

    final_df.write_csv(OUTPUT_FILE)
    print(f"[+] Saved microstructural dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_market_ticks()