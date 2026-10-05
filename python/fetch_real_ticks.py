"""Download 5-minute bars from Yahoo Finance (last 60 days, the most Yahoo serves at 5m)."""

import argparse
from pathlib import Path

import yfinance as yf

from bar_schema import bars_to_ticks, epoch_seconds, regular_session_mask, write_ticks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "data" / "market_ticks.csv"

TARGET_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMD"]


def generate_raw_tick_dataset(output_file=OUTPUT_FILE):
    print(f"[+] Exporting RAW un-normalized Yahoo market data for: {TARGET_TICKERS}")
    frames = []
    for symbol in TARGET_TICKERS:
        print(f"    Downloading {symbol}...")
        df = yf.download(symbol, period="60d", interval="5m", progress=False)
        if df.empty:
            print(f"    [!] No data for {symbol}, skipping")
            continue
        df = df[regular_session_mask(df.index)]
        frames.append(bars_to_ticks(symbol, epoch_seconds(df.index), df["High"].to_numpy().ravel(),
                                    df["Low"].to_numpy().ravel(), df["Close"].to_numpy().ravel(),
                                    df["Volume"].to_numpy().ravel()))
    write_ticks(frames, TARGET_TICKERS, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    generate_raw_tick_dataset(parser.parse_args().output)
