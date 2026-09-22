import polars as pl
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 1. Generate synthetic high-frequency order book tick data
np.random.seed(42)
n_ticks = 50_000

bid_prices = 100.0 + np.cumsum(np.random.randn(n_ticks) * 0.02)
ask_prices = bid_prices + np.random.uniform(0.01, 0.05, size=n_ticks)
bid_volumes = np.random.randint(10, 500, size=n_ticks)
ask_volumes = np.random.randint(10, 500, size=n_ticks)

df = pl.DataFrame({
    "bid_price": bid_prices,
    "ask_price": ask_prices,
    "bid_vol": bid_volumes,
    "ask_vol": ask_volumes,
})

# 2. Vectorized Feature Engineering with Polars
df_features = df.with_columns([
    ((pl.col("bid_price") + pl.col("ask_price")) / 2.0).alias("mid_price"),
    (pl.col("ask_price") - pl.col("bid_price")).alias("spread"),
    ((pl.col("bid_vol") - pl.col("ask_vol")) / (pl.col("bid_vol") + pl.col("ask_vol"))).alias("order_imbalance"),
]).with_columns([
    (pl.col("mid_price").diff(5)).alias("price_delta_5"),
    (pl.col("mid_price").rolling_std(window_size=20)).alias("volatility_20")
]).drop_nulls()

# Define target: +1 if mid-price rises in 5 ticks, -1 if it falls, 0 otherwise
df_features = df_features.with_columns(
    pl.when(pl.col("mid_price").shift(-5) > pl.col("mid_price") + 0.01).then(1)
      .when(pl.col("mid_price").shift(-5) < pl.col("mid_price") - 0.01).then(-1)
      .otherwise(0).alias("target")
).drop_nulls()

feature_cols = ["spread", "order_imbalance", "price_delta_5", "volatility_20"]
X_np = df_features.select(feature_cols).to_numpy()
y_np = df_features.select("target").to_numpy().flatten() + 1  # Shift [-1, 0, 1] to class indices [0, 1, 2]

# 3. Define Neural Network
class QuantMLP(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = QuantMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

X_tensor = torch.tensor(X_np, dtype=torch.float32)
y_tensor = torch.tensor(y_np, dtype=torch.long)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/10 - Loss: {loss.item():.4f}")

# 4. Serialize Model Graph to TorchScript (.pt)
model.eval()
traced_script_module = torch.jit.trace(model, torch.randn(1, 4))

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
output_path = models_dir / "quant_model.pt"

traced_script_module.save(output_path)
print(f"\nModel successfully exported to {output_path.resolve()}")