# -*- coding:utf-8 -*-
import gc
import warnings
from typing import Callable, Type

import numpy as np
import torch
from tqdm import tqdm
from datamanager.DataManager import DataManager


class AETrainTestManager(object):
    """
    Class used to train and test model given model and query strategy
    """

    def __init__(self, model,
                 dm: DataManager,
                 loss_fn: torch.nn.Module,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True):
        """
        Args:
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

    def training_iteration(self, num_epochs):
        """
        Train the model for num_epochs times on given data
        Args:
            num_epochs: number of times to train the model
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
                train_accuracies = []
                for i, data in enumerate(train_loader, 0):
                    # transfer tensors to selected device
                    train_inputs, _ = data[0].to(self.device), data[1].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward pass
                    train_outputs = self.model(train_inputs)

                    # TODO try to compare manual and builtin function for the MSE

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
                    # train_accuracies.append(accuracy(train_outputs, train_labels))

                    # print metrics along progress bar
                    train_loss += loss.item()
                    t.set_postfix(loss='{:05.3f}'.format(train_loss / (i + 1)))
                    t.update()

            # evaluate the model on validation data after each epoch
            mean_train_loss = np.mean(train_losses)
            # mean_train_accuracy = np.mean(train_accuracies)
            mean_val_loss, mean_val_accuracy = self.evaluate_on_validation_set()
            metrics['train_loss'].append(mean_train_loss)
            # metrics['train_accuracy'].append(mean_train_accuracy)
            metrics['val_loss'].append(mean_val_loss)
            # metrics['val_accuracy'].append(mean_val_accuracy)

        # self.metric_values['global_train_loss'].append(np.mean(metrics['train_loss']))
        # # self.metric_values['global_train_accuracy'].append(np.mean(metrics['train_accuracy']))
        # self.metric_values['global_val_loss'].append(np.mean(metrics['val_loss']))
        # # self.metric_values['global_val_accuracy'].append(np.mean(metrics['val_accuracy']))
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
        validation_accuracies = []

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
        accuracies = []
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
                # loss = loss.mean()

                losses.append(loss.item())
                # accuracies.append(accuracy(test_outputs, test_labels))

        # self.metric_values['global_test_loss'].append(np.mean(losses))
        # self.metric_values['global_test_accuracy'].append(np.mean(accuracies))
        codings = torch.cat(codings).cpu().numpy()
        codings_label = torch.cat(codings_label).cpu().numpy()
        losses_item = torch.cat(losses_item).cpu().numpy()
        return codings, codings_label, np.mean(losses), losses_item


def accuracy(outputs, labels):
    """
    Computes the accuracy of the model
    Args:
        outputs: outputs predicted by the model
        labels: real outputs of the data
    Returns:
        Accuracy of the model
    """
    predicted = outputs.argmax(dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def optimizer_setup(optimizer_class: Type[torch.optim.Optimizer], **hyperparameters) -> \
        Callable[[torch.nn.Module], torch.optim.Optimizer]:
    """
    Creates a factory method that can instanciate optimizer_class with the given
    hyperparameters.

    Why this? torch.optim.Optimizer takes the model's parameters as an argument.
    Thus we cannot pass an Optimizer to the CNNBase constructor.

    Args:
        optimizer_class: optimizer used to train the model
        **hyperparameters: hyperparameters for the model
        Returns:
            function to setup the optimizer
    """

    def f(model):
        return optimizer_class(model.parameters(), **hyperparameters)

    return f
