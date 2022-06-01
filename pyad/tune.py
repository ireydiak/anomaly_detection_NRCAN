import torch
import os
from ray import tune as ray_tune
from ray.tune.suggest.optuna import OptunaSearch
from pyad.model.base import BaseModel
from pyad.trainer.base import BaseTrainer


def tune(model_cls: BaseModel, trainer_cls: BaseTrainer, cfg: dict, n_epochs: int, device: str, checkpoint_dir=None):
    net = model_cls(**cfg)
    net.to(device)

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    data_dir = os.path.abspath("./data")
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


# # 1. Wrap a PyTorch model in an objective function.
# def objective(config):
#     train_loader, test_loader = load_data()  # Load some data
#     model = ConvNet().to("cpu")  # Create a PyTorch conv net
#     optimizer = torch.optim.SGD(  # Tune the optimizer
#         model.parameters(), lr=config["lr"], momentum=config["momentum"]
#     )
#
#     while True:
#         train(model, optimizer, train_loader)  # Train the model
#         acc = test(model, test_loader)  # Compute test accuracy
#         tune.report(aupr=acc)  # Report to Tune
#
#
# # 2. Define a search space and initialize the search algorithm.
# search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
# algo = OptunaSearch()
#
# # 3. Start a Tune run that maximizes mean accuracy and stops after 5 iterations.
# analysis = ray_tune.run(
#     objective,
#     metric="mean_accuracy",
#     mode="max",
#     search_alg=algo,
#     stop={"training_iteration": 5},
#     config=search_space,
# )
# print("Best config is:", analysis.best_config)

def setup(model_cls, trainer_cls):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    return device


if __name__ == "__main__":
    tune(device=device)
