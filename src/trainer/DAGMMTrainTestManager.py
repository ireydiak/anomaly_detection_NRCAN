# -*- coding:utf-8 -*-
import gc
import warnings
from typing import Callable, Type

import numpy as np
import torch
from tqdm import tqdm
from datamanager.DataManager import DataManager
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score


class DAGMMTrainTestManager(object):
    """
    Class used to train and test model given model and query strategy
    """

    def __init__(self, model,
                 dm: DataManager,
                 optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
                 use_cuda=True):
        """
        Args:
            model: model to train
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
                    code, x_hat, cosim, z_error, gamma = self.model(train_inputs)
                    phi, mu, cov_mat = self.model.compute_params(z_error, gamma)
                    energy_result, pen_cov_mat = self.model.estimate_sample_energy(z_error, phi, mu, cov_mat,
                                                                                   device=self.device)

                    loss = self.model.compute_loss(train_inputs, x_hat, energy_result, pen_cov_mat)

                    # # Use autograd to compute the backward pass.
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
            metrics['train_loss'].append(mean_train_loss)
            # TODO
            # Add other metrics to plot

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

    def evaluate_on_test_set(self, energy_threshold=80):
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
        N = 0
        gamma_sum = 0
        mu_sum = 0
        cov_mat_sum = 0

        # Change the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # Create pytorch's train data_loader
            train_loader = self.dm.get_train_set()

            for i, data in enumerate(train_loader, 0):
                # transfer tensors to selected device
                train_inputs, _ = data[0].to(self.device), data[1].to(self.device)

                # forward pass
                code, x_hat, cosim, z, gamma = self.model(train_inputs)
                phi, mu, cov_mat = self.model.compute_params(z, gamma)

                batch_gamma_sum = gamma.sum(axis=0)

                gamma_sum += batch_gamma_sum
                mu_sum += mu * batch_gamma_sum.unsqueeze(-1)  # keep sums of the numerator only
                cov_mat_sum += cov_mat * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)  # keep sums of the numerator only

                N += train_inputs.shape[0]

            train_phi = gamma_sum / N
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_mat_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

            print("Train N:", N)
            print("phi :\n", train_phi)
            print("mu :\n", train_mu)
            print("cov :\n", train_cov)

            # Calculate energy using estimated parameters

            train_energy = []
            train_labels = []
            train_z = []

            for i, data in enumerate(train_loader, 0):
                # transfer tensors to selected device
                train_inputs, train_inputs_labels = data[0].to(self.device), data[1]

                # forward pass
                code, x_hat, cosim, z, gamma = self.model(train_inputs)
                sample_energy, pen_cov_mat = self.model.estimate_sample_energy(z,
                                                                               train_phi,
                                                                               train_mu,
                                                                               train_cov,
                                                                               average_it=False,
                                                                               device=self.device)

                train_energy.append(sample_energy.cpu().numpy())
                train_z.append(z.cpu().numpy())
                train_labels.append(train_inputs_labels.numpy())

            train_energy = np.concatenate(train_energy, axis=0)
            train_z = np.concatenate(train_z, axis=0)
            train_labels = np.concatenate(train_labels, axis=0)

            test_energy = []
            test_labels = []
            test_z = []

            for data in test_loader:
                test_inputs, label_inputs = data[0].to(self.device), data[1]

                # forward pass
                code, x_hat, cosim, z, gamma = self.model(test_inputs)
                sample_energy, pen_cov_mat = self.model.estimate_sample_energy(z,
                                                                               train_phi,
                                                                               train_mu,
                                                                               train_cov,
                                                                               average_it=False,
                                                                               device=self.device)
                test_energy.append(sample_energy.cpu().numpy())
                test_z.append(z.cpu().numpy())
                test_labels.append(label_inputs.numpy())

            test_energy = np.concatenate(test_energy, axis=0)
            test_z = np.concatenate(test_z, axis=0)
            test_labels = np.concatenate(test_labels, axis=0)

            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            combined_labels = np.concatenate([train_labels, test_labels], axis=0)

            thresh = np.percentile(combined_energy, energy_threshold)
            print("Threshold :", thresh)

            # Prediction using the threshold value
            pred = (test_energy > thresh).astype(int)
            gt = test_labels.astype(int)

            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = prf(gt, pred, average='binary')

            print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy,
                                                                                                        precision,
                                                                                                        recall,
                                                                                                        f_score))
            # switch back to train mode
            self.model.train()
            return accuracy, precision, recall, f_score, test_z, test_labels, combined_energy


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
