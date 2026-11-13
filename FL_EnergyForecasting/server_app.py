
from pathlib import Path

import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from FL_EnergyForecasting.task import (
    Net,
    evaluate_all_test_buildings,
)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated energy forecasting with FedAvg."""

    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(
        context.run_config["fraction-evaluate"]
    )
    lr = float(context.run_config["learning-rate"])
    seq_len = int(context.run_config["seq-len"])
    horizon = int(context.run_config["horizon"])
    data_dir = context.run_config["data-dir"]
    output_dir = Path(context.run_config["output-dir"])

    # This example assumes the same model architecture at all clients.
    global_model = Net(horizon=horizon)
    initial_arrays = ArrayRecord(global_model.state_dict())

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Restore the final aggregated model.
    global_model.load_state_dict(
        result.arrays.to_torch_state_dict()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run the held-out test evaluation only after FL has finished.
    num_partitions = int(context.run_config["num-buildings"])
    test_metrics = evaluate_all_test_buildings(
        model=global_model,
        num_partitions=num_partitions,
        data_dir=data_dir,
        seq_len=seq_len,
        horizon=horizon,
        device=device,
    )

    print("\nFinal test results (mean across buildings)")
    print(f"MSE:  {test_metrics['mse']:.6f}")
    print(f"MAE:  {test_metrics['mae']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final_model.pt"
    torch.save(global_model.state_dict(), model_path)
    print(f"Saved final global model to {model_path}")
