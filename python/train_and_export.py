from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CSV_FILE = DATA_DIR / "market_ticks.csv"
MODEL_FILE = MODEL_DIR / "quant_model.pt"

SEQ_LEN = 30
EWMA_SPAN = 500
BATCH_SIZE = 256
EPOCHS = 35
LEARNING_RATE = 0.001


class QuantGRU(nn.Module):

  def __init__(self, input_dim=4, hidden_dim=32, num_classes=3):
    super().__init__()
    self.gru = nn.GRU(
        input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2
    )
    self.fc = nn.Sequential(
        nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, num_classes)
    )

  def forward(self, x):
    out, _ = self.gru(x)
    return self.fc(out[:, -1, :])


def train_and_export():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] Loading raw dataset from {CSV_FILE}...")
    df = pl.read_csv(CSV_FILE)

    cols_to_norm = ["raw_spread", "raw_ofi", "raw_delta", "raw_vol"]
    df_norm = df.with_columns([
        (
        (pl.col(c) - pl.col(c).ewm_mean(span=EWMA_SPAN, adjust=False, ignore_nulls=True).over("ticker"))
        / (pl.col(c).ewm_std(span=EWMA_SPAN, adjust=False, bias=True, ignore_nulls=True).over("ticker").fill_null(1.0) + 1e-6)
    ).alias(c)
    for c in cols_to_norm
    ]).fill_nan(0.0).fill_null(0.0)

    X_list, y_list = [], []
    for _, group in df_norm.group_by("ticker"):
        X_g = group.select(cols_to_norm).to_numpy().astype(np.float32)
    y_g = group.select("target").to_numpy().flatten().astype(np.int64)

    if len(X_g) >= SEQ_LEN:
        shape = (X_g.shape[0] - SEQ_LEN + 1, SEQ_LEN, X_g.shape[1])
        strides = (X_g.strides[0], X_g.strides[0], X_g.strides[1])
        X_win = np.lib.stride_tricks.as_strided(X_g, shape=shape, strides=strides)

        X_list.append(X_win)
        y_list.append(y_g[SEQ_LEN - 1 :])

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    class_counts = np.bincount(y_train, minlength=3)
    weights = np.sqrt(len(y_train) / (3.0 * class_counts + 1e-5))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = QuantGRU(input_dim=4, hidden_dim=32, num_classes=3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
    )

    print(
        f"[+] Dataset Split: {len(X_train)} Train | {len(X_val)} Validation"
        " Sequences"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
    train_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += batch_x.size(0) * loss.item()

    train_loss /= len(X_train)

    model.eval()
    with torch.no_grad():
        val_outputs = model(torch.tensor(X_val, dtype=torch.float32))
        val_loss = criterion(
            val_outputs, torch.tensor(y_val, dtype=torch.long)
        ).item()
        preds = torch.argmax(val_outputs, dim=1).numpy()
        val_acc = (preds == y_val).mean() * 100.0

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val"
            f" Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

    # Optimized TorchScript Export Pass
    model.eval()
    example_input = torch.rand(1, SEQ_LEN, 4, dtype=torch.float32)
    traced_module = torch.jit.trace(model, example_input)
    frozen_module = torch.jit.freeze(traced_module)
    optimized_module = torch.jit.optimize_for_inference(frozen_module)
    optimized_module.save(MODEL_FILE)
      # --- Offline signal-quality check (does NOT touch the exported .pt) ---
    model.eval()
    with torch.no_grad():
        val_probs = torch.softmax(
            model(torch.tensor(X_val, dtype=torch.float32)), dim=1
        ).numpy()
    preds = val_probs.argmax(axis=1)

    non_hold_mask = preds != 1
    n_non_hold = non_hold_mask.sum()

    print(f"\n[CHECK] Total val sequences:      {len(preds)}")
    print(f"[CHECK] Model called BUY/SELL on:  {n_non_hold} ({100*n_non_hold/len(preds):.1f}%)")

    if n_non_hold > 0:
        precision = (preds[non_hold_mask] == y_val[non_hold_mask]).mean()
        print(f"[CHECK] Precision on those calls:  {precision:.3f}  (chance = 0.333)")
    else:
        print("[CHECK] Model never calls BUY/SELL on val set — no signal to measure.")

    # Same thing broken out by class, so you can see if BUY or SELL specifically has edge
    for cls, name in [(0, "SELL"), (2, "BUY")]:
        cls_mask = preds == cls
        if cls_mask.sum() > 0:
            cls_prec = (preds[cls_mask] == y_val[cls_mask]).mean()
            print(f"[CHECK]   {name}: {cls_mask.sum()} calls, precision {cls_prec:.3f}")
        else:
            print(f"[CHECK]   {name}: 0 calls")
    print(f"[+] Optimized model exported to {MODEL_FILE}")


if __name__ == "__main__":
  train_and_export()