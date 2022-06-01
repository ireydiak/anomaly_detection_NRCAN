from ray import tune as ray_tune
from torch.utils.data import DataLoader
from pyad.model.base import BaseModel
from pyad.trainer.base import BaseTrainer


class Tuner:
    def __init__(self, model: BaseModel, trainer: BaseTrainer, cfg: dict):
        self.model = model
        self.trainer = trainer
        self.cfg = cfg

    def tune(self, dataset: DataLoader):
        self.trainer.train(dataset)
        self.model.train(mode=True)

        self.before_training(dataset)
        assert self.model.training, "Model not in training mode. Aborting"
        print("Started tuning")
        for epoch in range(10):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(dataset, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, _ = data
                inputs, labels = inputs.to(self.device).float(), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                loss = self.train_iter(inputs)

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0