# -*- coding:utf-8 -*-
import warnings
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm
from src.datamanager import DataManager


class AETrainTestManager:
    """
    Class used to train and test model given model and query strategy
    """

    def __init__(self, model,
                 dm: DataManager,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True):
        """

        Parameters
        ----------
        model: model to train
        querier: query_strategy object for active learning
        loss_fn: the loss function used
        optimizer_factory: A callable to create the optimizer. see optimizer function
        below for more details
        use_cuda: to Use the gpu to train the model
        """
        device_name = 'cuda:0' if use_cuda else 'cpu'
        if use_cuda and not torch.cuda.is_available():
            warnings.warn("CUDA is not available. Suppress this warning by passing "
                          "use_cuda=False to {}()."
                          .format(self.__class__.__name__), RuntimeWarning)
            device_name = 'cpu'

        self.device = torch.device(device_name)
        self.dm = dm
        self.loss_fn = loss_fn
        self.model = model
        self.optimizer = optimizer_factory(self.model)
        self.model = self.model.to(self.device)
        self.use_cuda = use_cuda
        self.metric_values = {}

    def training_iteration(self, num_epochs) -> dict:
        """
        Train the model for num_epochs times on given data

        Parameters
        ----------
        num_epochs: number of times to train the model

        Returns
        -------
        A dictionary containing the metrics:
            - train loss
            - train accuracy
            - val loss
            - val accuracy
        """
        # Initialize metrics container
        metrics = {'train_loss': [],
                   'train_accuracy': [],
                   'val_loss': [],
                   'val_accuracy': []}

        # Create pytorch's train data_loader
        train_loader = self.dm.get_train_set()
        # train num_epochs times
        for epoch in range(num_epochs):
            print("Epoch: {} of {}".format(epoch + 1, num_epochs))
            train_loss = 0.0

            with tqdm(range(len(train_loader))) as t:
                train_losses = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, _ = data[0].float().to(self.device), data[1].float().to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)

                    # computes loss using loss function loss_fn
                    loss = self.loss_fn(train_outputs, train_inputs)
                    loss = loss.mean(axis=1)
                    loss = loss.mean()
                    # Use autograd to compute the backward pass.
                    loss.backward()

                    # updates the weights using gradient descent
                    self.optimizer.step()

                    # Save losses for plotting purposes
                    train_losses.append(loss.item())

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

            # evaluate the model on validation data after each epoch
            mean_train_loss = np.mean(train_losses)
            mean_val_loss, mean_val_accuracy = self.evaluate_on_validation_set()
            metrics['train_loss'].append(mean_train_loss)
            metrics['val_loss'].append(mean_val_loss)

        return metrics

    def train(self, num_epochs, save_metrics=True, save_path='./', free_up_mem=True):
        """
        Train the model until reaching complete_data_ratio of labeled instances
        """

        # self.model.reset_weights()

        # Initialize metrics container
        metrics = self.training_iteration(num_epochs)

        # if save_metrics:
        #     np.savez_compressed(f'{save_path}{self.dm.__class__.__name__}_{self.model.__class__.__name__}',
        #                         global_train_loss=np.array(self.metric_values['global_train_loss']),
        #                         global_train_accuracy=np.array(self.metric_values['global_train_accuracy']),
        #                         global_val_loss=np.array(self.metric_values['global_val_loss']),
        #                         global_val_accuracy=np.array(self.metric_values['global_val_accuracy']),
        #                         global_test_loss=np.array(self.metric_values['global_test_loss']),
        #                         global_test_accuracy=np.array(self.metric_values['global_test_accuracy']),
        #                         number_of_data=np.array(self.metric_values['number_of_data']),
        #                         )

        # GPUtil.showUtilization()
        # if free_up_mem:
        #     self.querier.free_up_mem()
        #     self.model.to('cpu')
        #     del self.model, self.optimizer, self.loss_fn
        #     gc.collect()
        #
        # if self.use_cuda:
        #     torch.cuda.empty_cache()
        # GPUtil.showUtilization()
        print('Finished learning process')
        return metrics

    def evaluate_on_validation_set(self):
        """
        function that evaluate the model on the validation set every epoch
        """
        # switch to eval mode so that layers like batchnorm's layers nor dropout's layers
        # works in eval mode instead of training mode
        self.model.eval()

        # Get validation data
        val_loader = self.dm.get_validation_set()
        validation_loss = 0.0
        validation_losses = []

        with torch.no_grad():
            for j, val_data in enumerate(val_loader, 0):
                # transfer tensors to the selected device
                val_inputs, _ = val_data[0].to(self.device), val_data[1].to(self.device)

                # forward pass
                val_outputs = self.model(val_inputs)

                # compute loss function
                loss = self.loss_fn(val_inputs, val_outputs)
                loss = loss.mean(axis=1)
                loss = loss.mean()

                validation_losses.append(loss.item())
                # validation_accuracies.append(accuracy(val_outputs, val_labels))
                validation_loss += loss.item()

        mean_val_loss = np.mean(validation_losses)
        # mean_val_accuracy = np.mean(validation_accuracies)

        # displays metrics
        print('Validation loss %.3f' % (validation_loss / len(val_loader)))

        # switch back to train mode
        self.model.train()

        return mean_val_loss, mean_val_loss

    def evaluate_on_test_set(self):
        """
        function that evaluate the model on the test set every iteration of the
        active learning process
        """
        losses = []
        codings = []
        codings_label = []
        losses_item = []
        test_loader = self.dm.get_test_set()
        with torch.no_grad():
            for data in test_loader:
                test_inputs, label_inputs = data[0].to(self.device), data[1].to(self.device)

                test_outputs = self.model.encode(test_inputs)
                codings.append(test_outputs)
                codings_label.append(label_inputs)
                test_outputs = self.model.decode(test_outputs)

                # Compute loss function
                loss = self.loss_fn(test_outputs, test_inputs)
                loss = loss.mean(axis=1)

                losses_item.append(loss)

                loss = loss.mean()

                losses.append(loss.item())

        codings = torch.cat(codings).cpu().numpy()
        codings_label = torch.cat(codings_label).cpu().numpy()
        losses_item = torch.cat(losses_item).cpu().numpy()
        return codings, codings_label, np.mean(losses), losses_item
