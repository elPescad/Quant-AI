"""Download 5-minute bars from Alpaca (alpaca-py) into the engine's tick CSV schema.

Needs a (free) Alpaca account:
    export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
"""

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from bar_schema import bars_to_ticks, epoch_seconds, regular_session_mask, write_ticks
from fetch_real_ticks import OUTPUT_FILE, TARGET_TICKERS


def env_value(*names):
    """First set variable among names, stripped of the whitespace/quotes copy-paste tends to add."""
    for name in names:
        value = os.environ.get(name, "").strip().strip("'\"").strip()
        if value:
            return name, value
    return None, None


def credentials():
    key_var, key = env_value("APCA_API_KEY_ID", "ALPACA_API_KEY")
    secret_var, secret = env_value("APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("[-] Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (Alpaca dashboard -> API keys)")
    if "..." in key or "..." in secret:
        raise SystemExit(f"[-] ${key_var} / ${secret_var} still hold the placeholder from the README; "
                         "replace them with your own keys from the Alpaca paper trading dashboard")
    print(f"[+] Using key {key[:2]}...({len(key)} chars) from ${key_var}, secret ({len(secret)} chars) from ${secret_var}")
    return key, secret


def auth_failure(e):
    raise SystemExit(
        f"[-] Alpaca rejected the API keys (HTTP {e.status_code}). Check that:\n"
        "    1. Key ID and Secret are the pair shown together on the Trading API dashboard\n"
        "       (regenerating keys invalidates the old pair; the secret is only shown once)\n"
        "    2. they are Trading API keys (paper keys start with PK, live with AK);\n"
        "       Broker API keys do not work with data.alpaca.markets\n"
        "    3. they are exported in this shell: export APCA_API_KEY_ID=PK... APCA_API_SECRET_KEY=...")


def download(client, days, feed):
    # The free plan serves SIP history only up to 15 minutes ago
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    request = StockBarsRequest(symbol_or_symbols=TARGET_TICKERS, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=end - timedelta(days=days), end=end, adjustment=Adjustment.ALL, feed=feed)
    return client.get_stock_bars(request).df


def generate_raw_tick_dataset(days, feed, output_file):
    client = StockHistoricalDataClient(*credentials())
    print(f"[+] Downloading {days}d of 5-minute {feed.value.upper()} bars from Alpaca for: {TARGET_TICKERS}")
    try:
        bars = download(client, days, feed)
    except APIError as e:
        if e.status_code == 401:
            auth_failure(e)
        if feed != DataFeed.SIP or e.status_code != 403:
            raise
        # 403 = authenticated, but the plan does not include this SIP window
        print(f"    [!] SIP feed refused ({e}); falling back to IEX (IEX-only volume)")
        bars = download(client, days, DataFeed.IEX)

    frames = []
    for symbol in TARGET_TICKERS:
        if symbol not in bars.index.get_level_values("symbol"):
            print(f"    [!] No data for {symbol}, skipping")
            continue
        df = bars.xs(symbol, level="symbol").sort_index()
        df = df[regular_session_mask(df.index)]
        frames.append(bars_to_ticks(symbol, epoch_seconds(df.index), df["high"], df["low"], df["close"], df["volume"]))
    write_ticks(frames, TARGET_TICKERS, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=180, help="calendar days of history (default 180)")
    parser.add_argument("--feed", choices=["sip", "iex"], default="sip")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()
    generate_raw_tick_dataset(args.days, DataFeed(args.feed), args.output)
