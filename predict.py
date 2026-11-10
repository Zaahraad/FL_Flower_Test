import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pytorchexample.task import HORIZON, SEQ_LEN, Net, load_data, get_building_files

# Usage: python predict.py [model_path] [stride]
# stride = plot every Nth test sequence (24 -> one plot per day, 1 -> every sequence)
model_path = sys.argv[1] if len(sys.argv) > 1 else "final_model.pt"
stride = int(sys.argv[2]) if len(sys.argv) > 2 else 24

model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()

num_buildings = len(get_building_files())
rows = []
for b in range(num_buildings):
    _, _, testloader = load_data(b, num_buildings)
    ds = testloader.dataset
    with torch.no_grad():
        # The model outputs all 24 hours in one forward pass (direct, not recursive).
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
        ax.plot(ds.times[i - SEQ_LEN:i], real[i - SEQ_LEN:i], color="gray", label="history (168 h input)")
        ax.plot(ds.times[i:i + HORIZON], real[i:i + HORIZON], color="tab:blue", label="actual next 24 h")
        ax.plot(ds.times[i:i + HORIZON], preds[k], color="tab:red", linestyle="--", label="predicted next 24 h")
        ax.axvline(ds.times[i], color="black", linewidth=0.8)
        ax.set_title(f"Building {b} - test sequence {k} - forecast from {ds.times[i]}")
        ax.set_ylabel("consumption")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"seq_{k:05d}.png")
        plt.close(fig)
    print(f"Building {b}: plots saved to {out_dir}/")

df = pd.DataFrame(rows)
df.to_csv("test_predictions.csv", index=False)
print("Saved test_predictions.csv")
