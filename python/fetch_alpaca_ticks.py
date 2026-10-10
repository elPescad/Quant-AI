"""Download 5-minute bars from Alpaca (alpaca-py) into the engine's tick CSV schema.

Needs a (free) Alpaca account:
    export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...

Besides the bars it measures real bid-ask spreads for trading costs (quoted_spread column):
--quote-samples times per trading day, all tickers' quotes over a 1-second window; each bar
gets the median spread of the nearest sample. Spreads of these large caps are very stable,
so sampling costs ~10 minutes of API calls instead of downloading every quote.
"""

import argparse
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import numpy as np
import pandas as pd

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
        "    3. they are exported in this shell: export APCA_API_KEY_ID=PK... APCA_API_SECRET_KEY=...\n"
        "    Run with --check-keys to see which Alpaca service accepts these keys.")


def check_keys():
    """Try the keys on each Alpaca service; which one accepts them says what kind of keys they are."""
    key, secret = credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    probes = [
        ("market data (this script)", "https://data.alpaca.markets/v2/stocks/bars/latest?symbols=SPY&feed=iex", headers, None),
        ("paper trading account", "https://paper-api.alpaca.markets/v2/account", headers, None),
        ("live trading account", "https://api.alpaca.markets/v2/account", headers, None),
        ("Broker API sandbox", "https://broker-api.sandbox.alpaca.markets/v1/accounts?limit=1", None, (key, secret)),
    ]
    for name, url, hdrs, auth in probes:
        try:
            code = requests.get(url, headers=hdrs, auth=auth, timeout=15).status_code
            verdict = {200: "ACCEPTED", 401: "rejected (unknown key/secret)", 403: "authenticated, but forbidden"}.get(code, "")
        except requests.RequestException as e:
            code, verdict = "-", f"no connection ({type(e).__name__})"
        print(f"    {name:<27} HTTP {code}  {verdict}")


def download(client, days, feed):
    # The free plan serves SIP history only up to 15 minutes ago
    end = datetime.now(timezone.utc) - timedelta(minutes=16)
    request = StockBarsRequest(symbol_or_symbols=TARGET_TICKERS, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                               start=end - timedelta(days=days), end=end, adjustment=Adjustment.ALL, feed=feed)
    return client.get_stock_bars(request).df


QUOTE_WINDOW = timedelta(seconds=1)
MIN_REQUEST_INTERVAL = 0.32  # s; the free plan allows 200 market-data requests per minute


def sample_times(bar_index, per_day):
    """per_day evenly spaced instants inside each day's session (first bar + 5 min .. last bar + 4 min)."""
    ny = pd.DatetimeIndex(bar_index).tz_convert("America/New_York")
    times = []
    for _, day in pd.Series(ny, index=ny).groupby(ny.date):
        lo, hi = day.min() + pd.Timedelta(minutes=5), day.max() + pd.Timedelta(minutes=4)
        times += list(pd.date_range(lo, hi, periods=per_day)) if per_day > 1 else [lo]
    return [t.tz_convert("UTC") for t in times]


def sample_spreads(key, secret, times, feed):
    """Median bid-ask spread ($) per ticker in a QUOTE_WINDOW starting at each time."""
    client = StockHistoricalDataClient(key, secret, raw_data=True)
    print(f"[+] Sampling {feed.value.upper()} quotes at {len(times)} times (~{len(times) * MIN_REQUEST_INTERVAL / 60:.0f} min)")
    rows, last = [], 0.0
    for i, t in enumerate(times):
        time.sleep(max(0.0, MIN_REQUEST_INTERVAL - (time.monotonic() - last)))
        last = time.monotonic()
        request = StockQuotesRequest(symbol_or_symbols=TARGET_TICKERS, start=t.to_pydatetime(),
                                     end=(t + QUOTE_WINDOW).to_pydatetime(), feed=feed)
        try:
            quotes = client.get_stock_quotes(request)
        except APIError as e:
            if e.status_code == 401:
                auth_failure(e)
            if e.status_code != 403 or feed != DataFeed.SIP:
                raise
            print("    [!] SIP quotes refused; using IEX quotes (IEX's own book, wider than the NBBO)")
            done = _samples_frame(rows)
            return pd.concat([done, sample_spreads(key, secret, times[i:], DataFeed.IEX)], ignore_index=True)
        for sym, qs in quotes.items():
            spreads = [q["ap"] - q["bp"] for q in qs if q.get("bp", 0) > 0 and q.get("ap", 0) > q["bp"]]
            if spreads:
                rows.append((sym, int(t.timestamp()), float(np.median(spreads))))
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(times)} samples")
    return _samples_frame(rows)


def _samples_frame(rows):
    return pd.DataFrame(rows, columns=["symbol", "ts", "spread"]).astype({"ts": "int64", "spread": "float64"})


def attach_spreads(symbol, ts, samples):
    """Spread of the nearest same-session sample for each bar; the ticker's median where none is close."""
    mine = samples[samples["symbol"] == symbol].sort_values("ts")
    if mine.empty:
        return None
    bars = pd.DataFrame({"ts": ts, "order": np.arange(len(ts))}).sort_values("ts")
    merged = pd.merge_asof(bars, mine[["ts", "spread"]], on="ts", direction="nearest", tolerance=45 * 60)
    merged["spread"] = merged["spread"].fillna(mine["spread"].median())
    return merged.sort_values("order")["spread"].to_numpy()


def generate_raw_tick_dataset(days, feed, output_file, quote_samples=13):
    key, secret = credentials()
    client = StockHistoricalDataClient(key, secret)
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

    per_symbol = {}
    for symbol in TARGET_TICKERS:
        if symbol not in bars.index.get_level_values("symbol"):
            print(f"    [!] No data for {symbol}, skipping")
            continue
        df = bars.xs(symbol, level="symbol").sort_index()
        per_symbol[symbol] = df[regular_session_mask(df.index)]

    samples = None
    if quote_samples > 0 and per_symbol:
        all_bars = pd.DatetimeIndex(sorted(set().union(*(set(df.index) for df in per_symbol.values()))))
        samples = sample_spreads(key, secret, sample_times(all_bars, quote_samples), feed)

    frames = []
    print(f"    {'ticker':<6}{'quoted spread':>15}{'quoted bp':>11}{'high-low bp':>13}  (medians)")
    for symbol, df in per_symbol.items():
        ts = epoch_seconds(df.index)
        quoted = attach_spreads(symbol, ts, samples) if samples is not None else None
        hl_bp = np.median((df["high"] - df["low"]) / df["close"]) * 1e4
        if quoted is not None:
            print(f"    {symbol:<6}{'$' + format(np.median(quoted), '.4f'):>15}{np.median(quoted / df['close']) * 1e4:>11.2f}{hl_bp:>13.2f}")
        frames.append(bars_to_ticks(symbol, ts, df["high"], df["low"], df["close"], df["volume"], quoted))
    write_ticks(frames, TARGET_TICKERS, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=180, help="calendar days of history (default 180)")
    parser.add_argument("--feed", choices=["sip", "iex"], default="sip")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--quote-samples", type=int, default=13,
                        help="spread samples per trading day for trading costs (0 = no quotes, default 13)")
    parser.add_argument("--check-keys", action="store_true", help="only test the keys against each Alpaca service")
    args = parser.parse_args()
    if args.check_keys:
        check_keys()
    else:
        generate_raw_tick_dataset(args.days, DataFeed(args.feed), args.output, args.quote_samples)
