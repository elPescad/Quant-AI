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

# 1. Data (real 5-minute bars from Yahoo, or synthetic data with a known signal)
python python/fetch_real_ticks.py        # or: python python/generate_ticks.py

# 2. Train. Splits by time: 60% train / 20% validation / 20% test.
#    Writes models/quant_model.pt + quant_model_config.txt (quantum features)
#    and models/baseline_model.pt + baseline_model_config.txt (raw features, for A/B)
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
* `raw_spread` from Yahoo is the bar's high-low range, not a quoted spread; fills assume
  at most a 5 bp spread.
* Results on synthetic data only show that the machinery works; validate on real data
  before drawing conclusions.
