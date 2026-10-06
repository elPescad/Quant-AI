# Quant-AI

Low-latency C++ paper-trading engine with a quantum-inspired pipeline, all running on a
classical CPU:

```
CSV ticks ──► SPSC ring buffer ──► per-ticker features ──► simulated quantum ──► GRU ──► QUBO portfolio ──► portfolio
 (producer thread)                  EWMA z-scores           feature map          (Torch-   selection by       risk exits,
                                    + RLS AR(1) phi         5 qubits → 62        Script)   simulated          accounting
                                                            Pauli expectations              annealing
```

| Component | File | What it does |
|---|---|---|
| AR(1) estimator | `src/ar_estimator.hpp` | Recursive least squares with forgetting factor for `x_t = c + φ x_{t-1} + e_t` on order-flow imbalance. Exact exponentially weighted OLS fit, with standard error and half-life. |
| Quantum simulator | `src/quantum_sim.hpp` | Statevector simulator with H, X, RZ, RY, CNOT and RZZ (= CNOT·RZ·CNOT) gates. ZZ feature map (Havlíček et al. 2019) encodes the 5 features into a 32-amplitude state and reads out every Pauli Z- and X-string expectation. |
| Feature pipeline | `src/feature_pipeline.hpp` | Shared by engine, model runner and tests. `python/quantum_features.py` is the training-side mirror (parity-tested). |
| QUBO + annealing | `src/qubo.hpp` | QUBO model, simulated annealing (incremental fields, restarts, warm start, 1/2-flip polish), exact Gray-code brute force for verification. |
| Allocator | `src/qubo_allocator.hpp` | Long/short/flat per ticker as binaries; mean-variance objective with EWMA covariance, turnover costs, and a slack-encoded max-positions constraint. |
| Portfolio | `src/portfolio.hpp` | Fills at mid ± capped half spread, fees, gross-leverage cap, short proceeds excluded from buying power, stop-loss / take-profit / trailing exits. |

## Workflow

```bash
source ~/quant-ml-engine/venv/bin/activate

# 1. Data: 5-minute regular-session bars, same features/labels for every source (python/bar_schema.py)
python python/fetch_alpaca_ticks.py      # Alpaca (alpaca-py): needs APCA_API_KEY_ID / APCA_API_SECRET_KEY;
                                         #   --days 180 (default), --feed sip|iex, --output PATH
python python/fetch_real_ticks.py        # Yahoo: last 60 days only
python python/generate_ticks.py          # synthetic data with a known signal

# Pick the trading configuration by walk-forward validation (3 folds before the test period):
# candidates = {quantum, ensemble, q_veto, q_veto_short} x QUBO gamma {0 (= greedy), 25, 100}
# (q_veto: quantum GRU trades, raw GRU can only veto; _short: only SELL calls need agreement).
# Each fold retrains on data before it; score = mean fold Sharpe - 1 SE. Then one look at test,
# long/short P&L split, and a model_runner pass (C++ accuracy, signal strength, latency).
python python/compare_methods.py --data data/alpaca_ticks.csv

# 2. Train. Splits by time: 60% train / 20% validation / 20% test. Early stopping, quantum
#    bandwidth and probability calibration use the last 20% of the training window, so the
#    validation period stays out-of-sample. Labels crossing a split boundary are purged.
#    Writes baseline_model.pt (raw), quant_model.pt (quantum), ensemble_model.pt (average),
#    quant_veto_model.pt and quant_veto_short_model.pt (raw vetoes quantum), each with _config.txt
python python/train_and_export.py

# 3. Build (Release by default) and run
cmake -S . -B build -G Ninja && cmake --build build
cd build
./engine_tests ../tests/fixtures         # unit + parity tests, no torch needed
./model_runner                           # classification metrics on the test split
./quant_engine                           # backtest on the held-out test split
```

The model and its `_config.txt` must come from the same training run; the engine refuses
to start without the config. Re-run `train_and_export.py` after changing the feature code.

### `quant_engine` options

| Flag | Default | |
|---|---|---|
| `--period test\|val\|all` | `test` | Trade only the held-out test period (earlier data still warms up the features). `all` is in-sample. |
| `--allocator qubo\|greedy` | `qubo` | `greedy` = top-K by net expected return; it equals the QUBO optimum at `gamma = 0`. |
| `--risk-aversion X` | `100` | QUBO covariance penalty γ (tune with `--period val`). |
| `--max-positions K` | `4` | Each position is `1/K` of equity. |
| `--verify-qubo` | off | Solve every bar's QUBO by brute force too and report how often SA found the optimum. |
| `--model`, `--config`, `--data`, `--trades` | | Paths. `--model ../models/baseline_model.pt` runs the raw-feature baseline. |

## Data format

```
timestamp,ticker,raw_price,raw_spread,raw_ofi,raw_delta,raw_vol,target
```

Rows that share a `timestamp` form one bar; the engine rebalances once per bar. `target`
is the 6-bar-ahead label (0 sell, 1 hold, 2 buy, -1 unknown for the last bars).

## Caveats

* The quantum part is a classical simulation of a 5-qubit circuit. It is a fixed nonlinear
  feature map, not a source of quantum speed-up.
* `raw_spread` (Yahoo and Alpaca) is the bar's high-low range, not a quoted spread; fills assume
  at most a 5 bp spread.
* Results on synthetic data only show that the machinery works; validate on real data
  before drawing conclusions.
