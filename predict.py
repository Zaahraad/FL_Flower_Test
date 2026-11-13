import sys
import tomllib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from FL_EnergyForecasting.task import Net, load_data, get_building_files

# Use the same settings as the Flower run (seq-len, horizon, paths).
with open(Path(__file__).parent / "pyproject.toml", "rb") as f:
    config = tomllib.load(f)["tool"]["flwr"]["app"]["config"]
SEQ_LEN, HORIZON = config["seq-len"], config["horizon"]
DATA_DIR, OUTPUT_DIR = config["data-dir"], Path(config["output-dir"])

# Usage: python predict.py [model_path] [stride]
# stride = plot every Nth test sequence (24 -> one plot per day, 1 -> every sequence)
model_path = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_DIR / "final_model.pt"
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 24

model = Net(horizon=HORIZON)
model.load_state_dict(torch.load(model_path))
model.eval()

num_buildings = len(get_building_files(DATA_DIR))
rows = []
for b in range(num_buildings):
    _, _, testloader = load_data(b, num_buildings, DATA_DIR, SEQ_LEN, HORIZON)
    ds = testloader.dataset
    with torch.no_grad():
        # The model outputs all HORIZON hours in one forward pass (direct, not recursive).
        preds = torch.cat([model(x) for x, _ in testloader]).numpy() * ds.std + ds.mean
    real = ds.values * ds.std + ds.mean

    out_dir = Path("plots") / f"building_{b}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for k, i in enumerate(ds.indices):
        for h in range(HORIZON):
            rows.append({
                "building": b,
                "sequence": k,
                "forecast_made_at": ds.times[i - 1],
                "datetime": ds.times[i + h],
                "horizon": h + 1,
                "actual": real[i + h],
                "predicted": preds[k, h],
            })

        if k % stride:
            continue
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(ds.times[i - SEQ_LEN:i], real[i - SEQ_LEN:i], color="gray", label=f"history ({SEQ_LEN} h input)")
        ax.plot(ds.times[i:i + HORIZON], real[i:i + HORIZON], color="tab:blue", label=f"actual next {HORIZON} h")
        ax.plot(ds.times[i:i + HORIZON], preds[k], color="tab:red", linestyle="--", label=f"predicted next {HORIZON} h")
        ax.axvline(ds.times[i], color="black", linewidth=0.8)
        ax.set_title(f"Building {b} - test sequence {k} - forecast from {ds.times[i]}")
        ax.set_ylabel("consumption (kWh)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"seq_{k:05d}.png")
        plt.close(fig)
    print(f"Building {b}: plots saved to {out_dir}/")

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
print(f"Saved {OUTPUT_DIR / 'test_predictions.csv'}")
