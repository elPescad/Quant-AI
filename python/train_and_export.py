import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CSV_FILE = DATA_DIR / "market_ticks.csv"
MODEL_FILE = MODEL_DIR / "quant_model.pt"

SEQ_LEN = 10
EWMA_SPAN = 500  # ~2 hours of memory for dynamic normalization

class QuantGRU(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_classes=3):
        super(QuantGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :]
        return self.fc(last_step_out)

def train_and_export():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Loading raw dataset from {CSV_FILE}...")

    df = pl.read_csv(CSV_FILE)

    # 1. DYNAMIC EWMA PER-TICKER NORMALIZATION 
    cols_to_norm = ["raw_spread", "raw_ofi", "raw_delta", "raw_vol"]
    
    # We use ewm_mean and ewm_std to prevent 60-day variance compression
    df_norm = df.with_columns([
        ((pl.col(c) - pl.col(c).ewm_mean(span=EWMA_SPAN, ignore_nulls=True).over("ticker")) / 
         (pl.col(c).ewm_std(span=EWMA_SPAN, ignore_nulls=True).over("ticker").fill_null(1.0) + 1e-6)).alias(c)
        for c in cols_to_norm
    ])
    df_norm = df_norm.fill_nan(0.0).fill_null(0.0)

    # 2. CREATE ISOLATED SLIDING WINDOWS PER TICKER
    X_list, y_list = [], []
    for ticker, group in df_norm.group_by("ticker"):
        X_g = group.select(cols_to_norm).to_numpy().astype(np.float32)
        y_g = group.select("target").to_numpy().flatten().astype(np.int64)
        
        if len(X_g) >= SEQ_LEN:
            shape = (X_g.shape[0] - SEQ_LEN + 1, SEQ_LEN, X_g.shape[1])
            strides = (X_g.strides[0], X_g.strides[0], X_g.strides[1])
            X_win = np.lib.stride_tricks.as_strided(X_g, shape=shape, strides=strides)
            
            X_list.append(X_win)
            y_list.append(y_g[SEQ_LEN - 1:])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    print(f"[+] Generated {X.shape[0]} sequences. Feature non-stationarity fixed via EWMA.")

    # 3. CLASS WEIGHTING
    class_counts = np.bincount(y, minlength=3)
    weights = len(y) / (3.0 * class_counts)
    weights = np.nan_to_num(weights, posinf=1.0)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    model = QuantGRU(input_dim=4, hidden_dim=16, num_classes=3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print("[+] Training TorchScript GRU Model on dynamic features...")
    model.train()
    for epoch in range(120):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"[+] Training complete. Final Loss: {loss.item():.4f}")

    example_input = torch.rand(1, SEQ_LEN, 4, dtype=torch.float32)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save(MODEL_FILE)
    print(f"[+] TorchScript model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_and_export()