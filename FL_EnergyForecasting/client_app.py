
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from FL_EnergyForecasting.task import Net, load_data, train, evaluate


app = ClientApp()


def get_data(context):
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])

    trainloader, valloader, _ = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        batch_size=batch_size,
    )
    return partition_id, trainloader, valloader


@app.train()
def train_client(msg: Message, context: Context) -> Message:
    """Train the global model on one building's local training data."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    partition_id, trainloader, _ = get_data(context)

    model = Net()
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    lr = float(msg.content["config"]["lr"])
    epochs = int(context.run_config["local-epochs"])

    train_loss = train(
        model=model,
        trainloader=trainloader,
        epochs=epochs,
        lr=lr,
        device=device,
    )

    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": float(train_loss),
                    "num-examples": len(trainloader.dataset),
                }
            ),
        }
    )

    print(
        f"Client {partition_id}: "
        f"train_loss={train_loss:.4f}"
    )

    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate the received model on local validation data."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    partition_id, _, valloader = get_data(context)

    model = Net()
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    metrics = evaluate(model, valloader, device)

    print(
        f"Client {partition_id} validation: "
        f"MSE={metrics['mse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}"
    )

    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    **metrics,
                    "num-examples": len(valloader.dataset),
                }
            )
        }
    )

    return Message(content=content, reply_to=msg)
