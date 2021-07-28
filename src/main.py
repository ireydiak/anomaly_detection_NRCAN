#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
PhD Project
Authors:
    - D'Jeff Kanda
    - Jean-Charles Verdier
"""

import argparse

from torch import nn
from trainer import SOMDAGMMTrainer
import torch.optim as optim
from utils.utils import check_dir, optimizer_setup
from model import DAGMM, MemAutoEncoder as MemAE, SOMDAGMM
from datamanager import DataManager, KDD10Dataset, NSLKDDDataset, IDS2018Dataset, USBIDSDataset
from trainer import DAGMMTrainTestManager, MemAETrainer
from viz.viz import plot_3D_latent, plot_energy_percentile
from datetime import datetime as dt
import torch


vizualizable_models = ["AE", "DAGMM", "SOM-DAGMM"]

def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(
        usage='\n python3 main.py -m [model] -d [dataset-path] --dataset [dataset] [hyper_parameters]'
    )
    parser.add_argument('-m', '--model', type=str, default="DAGMM", choices=["AE", "DAGMM", "SOM-DAGMM", "MLAD", "MemAE"])
    parser.add_argument('-L', '--latent-dim', type=int, default=1)
    parser.add_argument('-d', '--dataset-path', type=str, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, default="kdd10", choices=["kdd10", "nslkdd", "ids2018", "usbids"])
    parser.add_argument('--batch-size', type=int, default=1024, help='The size of the training batch')
    parser.add_argument('--optimizer', type=str, default="Adam", choices=["Adam", "SGD", "RMSProp"],
                        help="The optimizer to use for training the model")
    parser.add_argument('-e', '--num-epochs', type=int, default=200, help='The number of epochs')
    parser.add_argument('-o', '--output-file', type=str, default=None, help='Where the results will be stored')
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
    parser.add_argument('--pct', type=float, default=1.0, help='Percentage of original data to keep')
    parser.add_argument('--rho', type=float, default=0.0, help='Anomaly ratio within training set')
    parser.add_argument('--save_path', type=str, default="./", help='The path where the output will be stored,'
                                                                    'model weights as well as the figures of '
                                                                    'experiments')
    parser.add_argument('--mem-dim', help='Memory Dimension', type=int, default=2000)
    parser.add_argument('--shrink-thres', type=float, default=0.0025)
    parser.add_argument('--vizualization', type=bool, default=False)

    return parser.parse_args()


def store_results(results: dict, params: dict, model_name: str, dataset: str, path: str, output_path: str=None):
    output_path = output_path or f'../results/{model_name}_results.txt'
    with open(output_path, 'a') as f:
        hdr = "Experiments on {}\n".format(dt.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({path.split("/")[-1].split(".")[0]})\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")


def resolve_optimizer(optimizer_str: str):
    # Defaults to 'Adam'
    optimizer_factory = optimizer_setup(optim.Adam, lr=args.lr)
    if optimizer_str == 'SGD':
        optimizer_factory = optimizer_setup(optim.SGD, lr=args.lr, momentum=0.9)
    return optimizer_factory


def resolve_trainer(trainer_str: str, optimizer_factory, **kwargs):
    model, trainer = None, None
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if trainer_str == 'DAGMM':
        model = DAGMM(dataset.get_shape()[1])
        trainer = DAGMMTrainTestManager(
            model=model, dm=dm, optimizer_factory=optimizer_factory
        )
    elif trainer_str == 'SOM-DAGMM':
        dagmm = DAGMM(
            dataset.get_shape()[1],
            gmm_layers=[(5, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]
        )
        # TODO
        # set these values according to the used dataset
        grid_length = 40  # int(5 * np.sqrt(len(dataset)))
        som_args = {
            "x": grid_length,
            "y": grid_length,
            "lr": 0.6,
            "neighborhood_function": "bubble",
            'n_epoch': 3000
        }
        model = SOMDAGMM(dataset.get_shape()[1], dagmm, som_args=som_args)
        trainer = SOMDAGMMTrainer(
            model=model, dm=dm, optimizer_factory=optimizer_factory
        )
        som_train_data = dataset.split_train_test()[0]
        data = [som_train_data[i][0] for i in range(len(som_train_data))]
        trainer.train_som(data)
    elif trainer_str == 'MemAE':
        print(f'training on {device}')
        model = MemAE(
            dataset.get_shape()[1], kwargs.get('latent_dim'), kwargs.get('mem_dim'), kwargs.get('shrink_thres'), device
        ).to(device)
        trainer = MemAETrainer(
            model=model, dm=dm, optimizer_factory=optimizer_factory, device=device
        )

    return model, trainer


if __name__ == "__main__":
    args = argument_parser()

    val_set = args.validation
    lambda_1 = args.lambda_energy
    lambda_2 = args.lambda_p


    # Dynamically load the Dataset instance
    clsname = globals()[f'{args.dataset.upper()}Dataset']
    dataset = clsname(args.dataset_path, args.pct)

    batch_size = len(dataset) if args.batch_size < 0 else args.batch_size

    # split data in train and test sets
    # we train only on the majority class
    train_set, test_set = dataset.one_class_split_train_test(test_perc=0.5, label=dataset.majority_cls_label)
    dm = DataManager(train_set, test_set, batch_size=batch_size, validation=0.1)

    # safely create save path
    check_dir(args.save_path)

    optimizer = resolve_optimizer(args.optimizer)

    model, model_trainer = resolve_trainer(
        args.model, optimizer, latent_dim=args.latent_dim, mem_dim=args.mem_dim, shrink_thres=args.shrink_thres
    )

    if model and model_trainer:
        metrics = model_trainer.train(args.num_epochs)
        print('Finished learning process')
        print('Evaluating model on test set')
        # We test with the minority samples as the positive class
        results, test_z, test_label, energy = model_trainer.evaluate_on_test_set(pos_label=dataset.majority_cls_label,
                                                                                 energy_threshold=args.p_threshold)

        params = dict({"BatchSize": batch_size, "Epochs": args.num_epochs, "\u03C1": args.rho}, **model.get_params())
        store_results(results, params, args.model, args.dataset, args.dataset_path, args.output_file)
        if args.vizualization and model in vizualizable_models:
            plot_3D_latent(test_z, test_label)
            plot_energy_percentile(energy)
    else:
        print(f'Error: Could not train {args.dataset} on model {args.model}')
