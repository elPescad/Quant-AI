import os
import torch
import torch.nn as nn
import torch.optim as optim
import polars as pl
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_FILE = PROJECT_ROOT / "data" / "market_ticks.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "quant_model.pt"

class QuantMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super(QuantMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def train_and_export():
    if not DATA_FILE.exists():
        print(f"[-] Data file not found at: {DATA_FILE}")
        return

    print(f"[+] Loading dataset from: {DATA_FILE}")
    df = pl.read_csv(DATA_FILE)
    
    X = torch.tensor(df.select(["spread", "order_imbalance", "price_delta_5", "volatility_20"]).to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["target"].to_numpy(), dtype=torch.long)

    model = QuantMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    print("[+] Training PyTorch model on Persistent Alpha Data...")
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}/10 - Loss: {total_loss / len(dataloader):.4f}")

    model.eval()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    example_input = torch.zeros(1, 4, dtype=torch.float32)
    traced_model = torch.jit.trace(model, example_input)
    frozen_model = torch.jit.freeze(traced_model)
    optimized_model = torch.jit.optimize_for_inference(frozen_model)

    optimized_model.save(str(MODEL_FILE))
    print(f"[+] TorchScript model saved to: {MODEL_FILE}")

if __name__ == "__main__":
    train_and_export()