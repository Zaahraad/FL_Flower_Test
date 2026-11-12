
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Project root / data/
DATA_DIR = Path("/home/zahra/fl-school-2026/quickstart-pytorch/Data")

SEQ_LEN = 168
HORIZON = 24
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
BATCH_SIZE = 64


class Net(nn.Module):
    """LSTM forecasting model: 168 hourly inputs -> 24 predictions."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, HORIZON)

    def forward(self, x):
        # x: [batch, 168, 1]
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])  # [batch, 24]


class ForecastDataset(Dataset):
    """Create input windows and future 24-hour targets."""

    def __init__(
        self,
        values: np.ndarray,
        target_start: int,
        target_end: int,
    ):
        self.values = values
        self.indices = list(
            range(
                max(SEQ_LEN, target_start),
                target_end - HORIZON + 1,
            )
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.values[i - SEQ_LEN:i]
        y = self.values[i:i + HORIZON]

        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(y, dtype=torch.float32),
        )


def get_building_files():
    files = sorted(DATA_DIR.glob("*.csv"))
    print(f"Data directory: {DATA_DIR}")
    print(f"Found {len(files)} CSV files:")
    for file in files:
        print(f"  {file.name}")
    return files


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = BATCH_SIZE,
):
    """Load one building and split it chronologically."""
    files = get_building_files()

    if len(files) != num_partitions:
        raise ValueError(
            f"Found {len(files)} CSV files in {DATA_DIR}, "
            f"but Flower configured {num_partitions} clients. "
            "Use one CSV per client and match the counts."
        )

    if not 0 <= partition_id < len(files):
        raise ValueError(f"Invalid partition ID: {partition_id}")

    path = files[partition_id]
    df = pd.read_csv(path)

    required = {"datetime", "consumption"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path.name} must contain columns {sorted(required)}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"], errors="raise"
    )
    df["consumption"] = pd.to_numeric(
        df["consumption"], errors="raise"
    )

    df = (
        df.sort_values("datetime")
        .drop_duplicates("datetime", keep="last")
        .set_index("datetime")
    )

    # Require complete hourly data rather than silently inventing
    # observations. Handle missing hours during preprocessing.
    if df.index.has_duplicates:
        raise ValueError(f"Duplicate datetimes in {path.name}")

    df = df.asfreq("h")

    if df["consumption"].isna().any():
        raise ValueError(
            f"{path.name} has missing hourly consumption values. "
            "Clean or impute them before training."
        )

    raw = df["consumption"].to_numpy(dtype=np.float32)

    if not np.isfinite(raw).all():
        raise ValueError(f"Non-finite values in {path.name}")

    n = len(raw)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    if train_end <= SEQ_LEN + HORIZON:
        raise ValueError(
            f"{path.name} has too few training observations."
        )
    if val_end - train_end < HORIZON:
        raise ValueError(
            f"{path.name} has too few validation observations."
        )
    if n - val_end < HORIZON:
        raise ValueError(
            f"{path.name} has too few test observations."
        )

    # Fit the scaler using training observations only.
    mean = float(raw[:train_end].mean())
    std = float(raw[:train_end].std())
    if std < 1e-8:
        std = 1.0

    values = (raw - mean) / std

    train_ds = ForecastDataset(values, SEQ_LEN, train_end)
    val_ds = ForecastDataset(values, train_end, val_end)
    test_ds = ForecastDataset(values, val_end, n)
    # Kept so predictions can be mapped back to real units and timestamps.
    test_ds.mean, test_ds.std, test_ds.times = mean, std, df.index

    if not len(train_ds) or not len(val_ds) or not len(test_ds):
        raise ValueError(
            f"An empty train/validation/test window was created "
            f"for {path.name}."
        )

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )
    testloader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return trainloader, valloader, testloader


def train(
    model,
    trainloader,
    epochs: int,
    lr: float,
    device,
):
    """Train on local data using MSE regression loss."""
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = 0.0
    total_values = 0

    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            predictions = model(x)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_values += y.numel()

    return total_loss / max(total_values, 1)


def evaluate(model, loader, device):
    """Return MSE, MAE and RMSE in standardized units."""
    model.to(device)
    model.eval()

    squared_error = 0.0
    absolute_error = 0.0
    count = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x)

            squared_error += (
                (predictions - y).square().sum().item()
            )
            absolute_error += (
                (predictions - y).abs().sum().item()
            )
            count += y.numel()

    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    mse = squared_error / count
    mae = absolute_error / count
    rmse = mse ** 0.5

    return {"mse": mse, "mae": mae, "rmse": rmse}


def evaluate_all_test_buildings(model, num_partitions, device):
    """Final test evaluation across all buildings, after FL finishes."""
    results = []

    for partition_id in range(num_partitions):
        _, _, testloader = load_data(
            partition_id, num_partitions, BATCH_SIZE
        )
        metrics = evaluate(model, testloader, device)
        results.append(metrics)

        print(
            f"Building {partition_id} test: "
            f"MSE={metrics['mse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}"
        )

    # Equal building weighting, so each building contributes equally.
    return {
        key: float(np.mean([row[key] for row in results]))
        for key in ("mse", "mae", "rmse")
    }
