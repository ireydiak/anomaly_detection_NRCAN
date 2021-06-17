#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
PhD Project
Authors: D'Jeff Kanda
"""

import argparse
import torch.optim as optim
import torch.nn as nn
import copy

from datamanager.GenericDataset import GenericDataset
from datamanager.NSLKDDDataset import NSLKDDDataset
from utils.utils import check_dir, optimizer_setup
# from TrainTestManager import TrainTestManager, optimizer_setup

from model.AutoEncoder import AutoEncoder as AE
from model.DAGMM import DAGMM
from datamanager.KDDDataset import KDDDataset
from datamanager.DataManager import DataManager
from trainer.AETrainTestManager import AETrainTestManager
from trainer.DAGMMTrainTestManager import DAGMMTrainTestManager
from viz.viz import plot_losses, plot_2D_latent, plot_1D_latent_vs_loss, plot_3D_latent, plot_energy_percentile


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 main.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 main.py --model AE [hyper_parameters]'
                                           '\n python3 main.py --model AE --predict',
                                     description="Description...."
                                     )
    parser.add_argument('--model', type=str, default="DAGMM",
                        choices=["AE", "DAGMM", "MLAD"])
    parser.add_argument('--dataset', type=str, default="nslkdd", choices=["kdd", "nslkdd", "hsherbrooke"])
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD", "RMSProp"],
                        help="The optimizer to use for training the model")
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--p-threshold', type=float, default=80,
                       help='percentile threshold for the energy ')
    parser.add_argument('--lambda-energy', type=float, default=0.1,
                        help='loss energy factor')
    parser.add_argument('--lambda-p', type=float, default=0.005,
                        help='loss related to the inverse of diagonal element of the covariance matrix ')
    parser.add_argument('--save_path', type=str, default="./", help='The path where the output will be stored,'
                                                                    'model weights as well as the figures of '
                                                                    'experiments')
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    val_set = args.validation
    learning_rate = args.lr
    lambda_1 = args.lambda_energy
    lambda_2 = args.lambda_p
    p_threshold = args.p_threshold

    # Loading the data
    if args.dataset == 'kdd':
        dataset = KDDDataset(path='../data/kddcup_data_10_percent_corrected.csv')
    elif args.dataset == 'nslkdd':
        dataset = NSLKDDDataset(path='../data/NSL-KDD/KDDTrain.txt')
    else:
        raise Exception("This dataset is not available for experiment at present")

    # normal_data = dataset.get_data_index_by_label(label=0)

    # split data in train and test sets
    train_set, test_set = dataset.one_class_split_train_test(test_perc=0.5, label=0)
    # test_set = NSLKDDDataset(path='../data/NSL-KDD/KDDTest.txt')
    dm = DataManager(train_set, test_set, batch_size=batch_size, validation=0.000001)

    # safely create save path
    check_dir(args.save_path)

    if args.optimizer == 'SGD':
        optimizer_factory = optimizer_setup(optim.SGD, lr=learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    model = DAGMM(dataset.get_shape()[1],
                  [60, 30, 10, 1],
                  fa='tanh',
                  gmm_layers=[10, 2]
                  )

    model_trainer = DAGMMTrainTestManager(model=model,
                                          dm=dm,
                                          optimizer_factory=optimizer_factory,
                                          )

    metrics = model_trainer.train(num_epochs)
    _, _, _, _, test_z, test_label, energy = model_trainer.evaluate_on_test_set(p_threshold)

    plot_3D_latent(test_z, test_label)
    plot_energy_percentile(energy)

    # print(dataset.get_shape())
    # model = AE(dataset.get_shape()[1], [60, 30, 10, 2], fa='tanh')
    #
    # model_trainer = AETrainTestManager(model=model,
    #                                    dm=dm,
    #                                    loss_fn=nn.MSELoss(reduction='none'),
    #                                    optimizer_factory=optimizer_factory,
    #                                    )
    #
    # metrics = model_trainer.train(200)
    # codes, labels, loss, losses_items = model_trainer.evaluate_on_test_set()
    # # train_loss
    # plot_losses(metrics['train_loss'], metrics['val_loss'])
    # plot_2D_latent(codes, labels)
    # # plot_1D_latent_vs_loss(codes, labels, losses_items)
    #
    # print('Test one')
    # codes, labels, loss, losses_items = model_trainer.evaluate_on_test_set()
    # # train_loss
    # plot_losses(metrics['train_loss'], metrics['val_loss'])
    # plot_2D_latent(codes, labels)
    # # plot_1D_latent_vs_loss(codes, labels, losses_items)

    # Test DAGMM