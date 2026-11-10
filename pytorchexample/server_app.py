
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
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

    # This example assumes the same model architecture at all clients.
    global_model = Net()
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
        device=device,
    )

    print("\nFinal test results (mean across buildings)")
    print(f"MSE:  {test_metrics['mse']:.6f}")
    print(f"MAE:  {test_metrics['mae']:.6f}")
    print(f"RMSE: {test_metrics['rmse']:.6f}")

    torch.save(global_model.state_dict(), "final_model.pt")
    print("Saved final global model to final_model.pt")
