# Quant-AI

Low-latency C++ paper-trading engine with a quantum-inspired pipeline, all running on a
classical CPU. Python only trains; the engine is a single static C++ binary with no Python,
no libtorch and no shared-library dependencies (2.5 MB, ~10 MB peak RSS on a backtest).

```
ticks (CSV/stdin) ──► SPSC ring buffer ──► per-ticker features ──► simulated quantum ──► GRU ──────► allocation ──────► portfolio
 (producer thread)                         EWMA z-scores           feature map          (native     greedy, or QUBO    risk exits,
                                           + RLS AR(1) phi         5 qubits → 62        C++, AVX2)  by simulated       accounting
                                                                   Pauli expectations               annealing
```

| Component | File | What it does |
|---|---|---|
| AR(1) estimator | `src/ar_estimator.hpp` | Recursive least squares with forgetting factor for `x_t = c + φ x_{t-1} + e_t` on order-flow imbalance. Exact exponentially weighted OLS fit, with standard error and half-life. |
| Quantum simulator | `src/quantum_sim.hpp` | Statevector simulator with H, X, RZ, RY, CNOT and RZZ (= CNOT·RZ·CNOT) gates. ZZ feature map (Havlíček et al. 2019) encodes the 5 features into a 32-amplitude state and reads out every Pauli Z- and X-string expectation. |
| GRU inference | `src/native_model.hpp` | Loads the `.weights` files written by `python/native_model.py` and runs the GRUs (and their ensemble / veto combinations) as vectorised loops. Matches PyTorch to ~1e-7 (parity test); ~18x faster than the TorchScript path it replaced. |
| Feature pipeline | `src/feature_pipeline.hpp` | Shared by engine, model runner and tests. `python/quantum_features.py` is the training-side mirror (parity-tested). |
| QUBO + annealing | `src/qubo.hpp` | QUBO model, simulated annealing (incremental fields, restarts, warm start, 1/2-flip polish), exact Gray-code brute force for verification. |
| Allocator | `src/qubo_allocator.hpp` | Long/short/flat per ticker as binaries; mean-variance objective with EWMA covariance, turnover costs, and a slack-encoded max-positions constraint. |
| Live feed | `src/live_feed.hpp`, `src/live_bars.hpp`, `src/alpaca_client.hpp` | Alpaca market clock, REST history and WebSocket stream on libcurl; builds 5-minute bars from 1-minute bars with the same feature code as training (bit-identical to `bar_schema.py`). |
| Portfolio | `src/portfolio.hpp` | Fills at mid ± capped half spread, fees, gross-leverage cap, short proceeds excluded from buying power, stop-loss / take-profit / trailing exits. |

## Workflow

```bash
source ~/quant-ml-engine/venv/bin/activate

# 1. Data: 5-minute regular-session bars, same features/labels for every source (python/bar_schema.py)
python python/fetch_alpaca_ticks.py      # Alpaca (alpaca-py): needs APCA_API_KEY_ID / APCA_API_SECRET_KEY;
                                         #   --days 180 (default), --feed sip|iex, --output PATH; also samples
                                         #   real bid-ask quotes (--quote-samples 13/day) for trading costs
python python/fetch_real_ticks.py        # Yahoo: last 60 days only
python python/generate_ticks.py          # synthetic data with a known signal

# Pick the trading configuration by walk-forward validation (3 folds before the test period):
# candidates = {quantum, ensemble, q_veto, q_veto_short} x QUBO gamma {0 (= greedy), 25, 100}
# (q_veto: quantum GRU trades, raw GRU can only veto; _short: only SELL calls need agreement).
# Each fold retrains on data before it; score = mean fold Sharpe - 1 SE. Then one look at test,
# long/short P&L split, and a model_runner pass (C++ accuracy, signal strength, latency).
python python/compare_methods.py --data data/alpaca_ticks.csv   # add --online-lrs 0 0.001 0.003 to test online learning
# Then check the result is skill and not market drift or luck (seconds, no training):
# vs buy & hold, beta-adjusted market-neutral P&L, and a random-direction permutation test
python python/sanity_check.py --data data/alpaca_ticks.csv

# 2. Train. Splits by time: 60% train / 20% validation / 20% test. Early stopping, quantum
#    bandwidth and probability calibration use the last 20% of the training window, so the
#    validation period stays out-of-sample. Labels crossing a split boundary are purged.
#    Each GRU is trained from --seeds 5 random starts and their probabilities are averaged.
#    Writes baseline_model (raw), quant_model (quantum), ensemble_model (average),
#    quant_veto_model and quant_veto_short_model (raw vetoes quantum) as <name>.weights,
#    each with <name>_config.txt
python python/train_and_export.py

# 3. Build (Release by default; needs only a C++20 compiler and CMake) and run
cmake -S . -B build -G Ninja && cmake --build build
cd build
./engine_tests ../tests/fixtures         # unit + C++/PyTorch parity tests
./model_runner                           # classification metrics on the test split
./quant_engine                           # backtest on the held-out test split
```

`-DQUANT_ARCH=x86-64-v3` builds a portable AVX2 binary instead of one tuned for the build
machine; `-DQUANT_STATIC=ON` links it fully statically. After changing the GRU code or the
model classes, regenerate the parity fixture with `python python/native_model.py --fixture tests/fixtures`.

### Docker (Debian slim)

```bash
docker build -t quant-engine .           # compiles, runs engine_tests, keeps only the binaries
docker run --rm --network=none --cpus=2 --memory=1g --user "$(id -u):$(id -g)" \
  -v "$PWD/models:/app/models:ro" -v "$PWD/data:/app/data:ro" -v "$PWD/out:/app/out" \
  quant-engine                           # = quant_engine on /app/data/market_ticks.csv
```

If the build fails with `failed to add the host (veth...) <=> sandbox (veth...) pair interfaces:
operation not supported`, Docker cannot create its virtual network interfaces on this host
(on Arch: the kernel was upgraded and the running kernel's modules are gone until a reboot).
Reboot, or build with `docker build --network=host -t quant-engine .`.

The image is ~40 MB compressed (Debian trixie slim, libcurl for the live feed and our two
binaries). The build targets x86-64-v3 so an image built on one machine runs on any current
x86 cloud VM.

The model and its `_config.txt` must come from the same training run; the engine refuses
to start without the config. Re-run `train_and_export.py` after changing the feature code.

### Live paper trading (Alpaca)

```bash
export APCA_API_KEY_ID=... APCA_API_SECRET_KEY=...
./quant_engine --live-check                   # clock, REST and stream connections (works when the market is closed)
./quant_engine --live --paper-orders --model ../models/ensemble_model.weights --trades out/live_trades.csv
# or in Docker, running until stopped (restarts by itself after a crash or VM reboot):
docker run -d --name quant-live --restart unless-stopped --cpus=2 --memory=1g --user "$(id -u):$(id -g)" \
  -e APCA_API_KEY_ID -e APCA_API_SECRET_KEY -v "$PWD/models:/app/models:ro" -v "$PWD/out:/app/out" \
  quant-engine --live --paper-orders --model /app/models/ensemble_model.weights --trades /app/out/live_trades.csv \
  --online-lr 0.001 --online-state /app/out/online_state.weights      # only if walk-forward chose it
docker logs -f quant-live                     # one status line per 5-minute bar
docker stop -t 30 quant-live                  # flattens (if the market is open) and prints the report
python python/paper_report.py out/account.csv # Sharpe, drawdown and P&L of the paper account
```

How it runs: it asks Alpaca's market clock whether the market is open (holidays and early
closes included) and sleeps until the next open when it is not. At start-up it fetches
`--warmup-days` (30) of 5-minute history so the features are warm, then streams 1-minute
bars and quotes over one WebSocket and builds each 5-minute bar, acting on it as soon as it
is complete. Data is pushed by Alpaca, so nothing polls and no bar is processed twice; after
a dropped connection it reconnects and fills any missed bars from REST. Idle it uses ~0.1% of
a core and ~21 MB of memory.

Paper orders (`--paper-orders`): after every bar a separate thread makes the paper account's
positions in the configured tickers equal the engine's (whole shares, sized for $10k of the
account): it reads the positions, cancels open orders and sends market orders for the
differences; a long is closed before a short is opened (Alpaca rejects flips). The account is
the source of truth, so restarts and partial fills converge on the next bar. It only trades
while the market is open, only touches its own tickers, and refuses any trading URL other than
`https://paper-api.alpaca.markets`. Every order is logged in `orders.csv`, the account equity
and positions after each bar in `account.csv` (next to `--trades`). The engine's own simulated
trades stay in `--trades`; differences between the two are fill timing and price.

Online learning (`--online-lr`, off by default): six bars after each prediction its label is
known (same rule as the training data); the model then takes one small gradient step on its
output layer and calibration, pulled back towards the trained weights (`--online-anchor`), so
it keeps adapting to recent days without drifting far on noise. The GRUs themselves stay fixed;
full retraining stays offline. `--online-state` saves what was learned (hourly and at exit) and
restores it on restart, only for the same model file. Choose the rate by walk-forward
(`compare_methods.py --online-lrs 0 0.001 0.003`): on synthetic data 0.001 helped and 0.01+
hurt, because large steps chase noise.

Feeds: Alpaca's free plan streams real-time data from IEX only (`--feed iex`, default); IEX
volume is a few percent of the whole market, so for live use train on IEX bars too
(`python python/fetch_alpaca_ticks.py --feed iex --output data/alpaca_iex.csv`). A paid plan
gives the consolidated SIP feed (`--feed sip`), matching the default training data. Live mode
needs libcurl with WebSocket support (curl 8.11+, e.g. Arch or Debian trixie); CMake builds
without it otherwise.

### `quant_engine` options

| Flag | Default | |
|---|---|---|
| `--period test\|val\|all` | `test` | Trade only the held-out test period (earlier data still warms up the features). `all` is in-sample. |
| `--allocator greedy\|qubo` | `greedy` | `greedy` = top-K by net expected return; it equals the QUBO optimum at `gamma = 0`, the walk-forward choice. |
| `--risk-aversion X` | `100` | QUBO covariance penalty γ (tune with `--period val`). |
| `--max-positions K` | `4` | Each position is `1/K` of equity. |
| `--fee-bps X` | `0.2` | Fees per fill. Alpaca charges no commission; this covers the regulatory fees on sales. |
| `--verify-qubo` | off | Solve every bar's QUBO by brute force too and report how often SA found the optimum. |
| `--model`, `--config`, `--data`, `--trades` | | Paths. Default model `../models/ensemble_model.weights`; `--data -` reads ticks from stdin (for a live feed). |

## Data format

```
timestamp,ticker,raw_price,raw_spread,raw_ofi,raw_delta,raw_vol,target,quoted_spread
```

Rows that share a `timestamp` form one bar; the engine rebalances once per bar. `target`
is the 6-bar-ahead label (0 sell, 1 hold, 2 buy, -1 unknown for the last bars).
`raw_spread` is the bar's high-low range (a model feature). `quoted_spread` is the measured
bid-ask spread in $ used for fill costs; it is optional (`-1` or absent when the source has
no quotes, e.g. Yahoo or synthetic data, and then the engine falls back to the high-low
range capped at 5 bp, which overstates costs for liquid stocks).

## Caveats

* The quantum part is a classical simulation of a 5-qubit circuit. It is a fixed nonlinear
  feature map, not a source of quantum speed-up.
* Fills happen at the bar's closing mid ± half the quoted spread. With Alpaca data the spread
  is measured from sampled quotes; queue position, latency and market impact are not modelled.
* Results on synthetic data only show that the machinery works; validate on real data
  before drawing conclusions.
