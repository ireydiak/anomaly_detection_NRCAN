#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
University of Sherbrooke
PhD Project
Authors:
    - D'Jeff Kanda
    - Jean-Charles Verdier
    - Maxime Alvarez
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
from recforest import RecForest
from torch import nn

from model.DUAD import DUAD
from model.density import DSEBM
from model.adversarial import ALAD
from model.BaseModel import BaseModel
from model.transformers import NeuTraAD
from model.one_class import DeepSVDD
from trainer.NeuTraADTrainer import NeuTraADTrainer
from trainer.DSEBMTrainer import DSEBMTrainer
from trainer.AETrainer import AETrainer
from trainer.DUADTrainer import DUADTrainer
from utils.metrics import score_recall_precision, score_recall_precision_w_thresold
from trainer import SOMDAGMMTrainer
import torch.optim as optim
from utils.utils import check_dir, optimizer_setup, get_X_from_loader, average_results
from model import AutoEncoder as AE
from datamanager import ArrhythmiaDataset, DataManager, KDD10Dataset, NSLKDDDataset, IDS2018Dataset
from model import DAGMM, MemAutoEncoder as MemAE, DeepSVDD #TODO what about the original DeepSVDD?
from model.adversarial import ALAD
from datamanager import ArrhythmiaDataset, DataManager, KDD10Dataset, NSLKDDDataset, IDS2018Dataset, USBIDSDataset, \
    ThyroidDataset
from trainer import ALADTrainer, DAGMMTrainer, MemAETrainer, DeepSVDDTrainer
from viz.viz import plot_3D_latent, plot_energy_percentile
from datetime import datetime as dt
from sklearn import metrics
from sklearn.svm import OneClassSVM

import torch
import os

vizualizable_models = ["AE", "DAGMM", "SOM-DAGMM"]
SKLEARN_MODEL = ['RECFOREST', 'OC-SVM']


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(
        usage='\n python3 main.py -m [model] -d [dataset-path] --dataset [dataset] [hyper_parameters]'
    )
    parser.add_argument('-m', '--model', type=str, default="DAGMM",
                        choices=["AE", "ALAD", "DAGMM", "SOM-DAGMM", "MLAD", "MemAE", "DUAD", 'OC-SVM', 'RECFOREST',
                                 "DSEBM", "DeepSVDD", "NeurTraAD"])

    parser.add_argument('-rt', '--run-type', type=str, default="train",
                        choices=["train", "test"])

    parser.add_argument('--n-runs', help='number of runs of the experiment', type=int, default=1)
    parser.add_argument('-lat', '--latent-dim', type=int, default=1)

    parser.add_argument('-d', '--dataset-path', type=str, help='Path to the dataset')
    parser.add_argument('--dataset', type=str, default="kdd10",
                        choices=["Arrhythmia", "KDD10", "NSLKDD", "IDS2018", "USBIDS", "Thyroid"])
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

    parser.add_argument('--n-som', help='number of SOM component', type=int, default=1)

    # =======================OC-SVM==========================
    parser.add_argument('--nu', type=float, default=0.5, help="The 'margin' for the SVM. Specifies the"
                                                                "anomaly ratio in training data.")

    # =======================DUAD=========================
    parser.add_argument('--r', type=int, default=10, help='Number of epoch required to re-evaluate the selection')
    parser.add_argument('--p_s', type=float, default=35, help='Variance threshold of initial selection')
    parser.add_argument('--p_0', type=float, default=30, help='Variance threshold of re-evaluation selection')
    parser.add_argument('--num-cluster', type=int, default=20, help='Number of clusters')

    # =======================DAGMM==========================
    parser.add_argument('--reg-covar', type=float, default=1e-6, help="Non - negative regularization added to the "
                                                                      "diagonal of covariance. Allows to assure that "
                                                                      "the covariance matrices are all positive.")

    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default="./", help='The path where output models are stored')

    return parser.parse_args()


def store_results(results: dict, params: dict, model_name: str, dataset: str, path: str):
    output_dir = f'../results/{dataset}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    with open(output_dir + '/' + f'{model_name}_results.txt', 'a') as f:
        hdr = "Experiments on {}\n".format(dt.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({path.split("/")[-1].split(".")[0]})\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")


def store_models(models: List[BaseModel], model_name: str, dataset: str, path: str):
    output_dir = f'../models/{dataset}/{model_name}/{dt.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('Saving models')
    for i, model in enumerate(models):
        model.save(f"{output_dir}/model_{i}")
    print('Models saved')


def resolve_optimizer(optimizer_str: str):
    # Defaults to 'Adam'
    optimizer_factory = optimizer_setup(optim.Adam, lr=args.lr)
    if optimizer_str == 'SGD':
        optimizer_factory = optimizer_setup(optim.SGD, lr=args.lr, momentum=0.9)
    return optimizer_factory


def resolve_trainer(trainer_str: str, optimizer_factory, **kwargs):
    model, trainer = None, None
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    D = dataset.get_shape()[1]
    L = kwargs.get("latent_dim", D // 2)
    reg_covar = kwargs.get("reg_covar", 1e-12)
    if trainer_str == 'DAGMM' or trainer_str == 'SOM-DAGMM' or trainer_str == 'AE':
        if dataset.name == 'Arrhythmia' or (dataset.name == 'Thyroid' and trainer_str != 'DAGMM'):
            enc_layers = [(D, 10, nn.Tanh()), (10, L, None)]
            dec_layers = [(L, 10, nn.Tanh()), (10, D, None)]
            gmm_layers = [(L + 2, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
        elif dataset.name == 'Thyroid' and trainer_str == 'DAGMM':
            enc_layers = [(D, 12, nn.Tanh()), (12, 4, nn.Tanh()), (4, L, None)]
            dec_layers = [(L, 4, nn.Tanh()), (4, 12, nn.Tanh()), (12, D, None)]
            gmm_layers = [(L + 2, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
        else:
            enc_layers = [(D, 60, nn.Tanh()), (60, 30, nn.Tanh()), (30, 10, nn.Tanh()), (10, L, None)]
            dec_layers = [(L, 10, nn.Tanh()), (10, 30, nn.Tanh()), (30, 60, nn.Tanh()), (60, D, None)]
            gmm_layers = [(L + 2, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]

        if trainer_str == 'DAGMM':
            model = DAGMM(D, ae_layers=(enc_layers, dec_layers), gmm_layers=gmm_layers, reg_covar=reg_covar)
            trainer = DAGMMTrainTestManager(
                model=model, dm=dm, optimizer_factory=optimizer_factory,
            )
        elif trainer_str == 'AE':
            model = AE(enc_layers, dec_layers)
            trainer = AETrainer(
                model=model, dm=dm, optimizer_factory=optimizer_factory
            )
        else:
            gmm_input = kwargs.get('n_som', 1) * 2 + kwargs.get('latent_dim', 1) + 2
            if dataset.name == 'Arrhythmia':
                gmm_layers = [(gmm_input, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
            else:
                gmm_layers = [(gmm_input, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]

            dagmm = DAGMM(
                dataset.get_shape()[1],
                ae_layers=(enc_layers, dec_layers),
                gmm_layers=gmm_layers, reg_covar=reg_covar
            )
            # TODO
            # set these values according to the used dataset
            grid_length = int(np.sqrt(5 * np.sqrt(len(dataset)))) // 2
            grid_length = 32 if grid_length > 32 else grid_length
            som_args = {
                "x": grid_length,
                "y": grid_length,
                "lr": 0.6,
                "neighborhood_function": "bubble",
                'n_epoch': 8000,
                'n_som': kwargs.get('n_som')
            }
            model = SOMDAGMM(dataset.get_shape()[1], dagmm, som_args=som_args)
            trainer = SOMDAGMMTrainer(
                model=model, dm=dm, optimizer_factory=optimizer_factory
            )
            som_train_data = dataset.split_train_test()[0]
            data = [som_train_data[i][0] for i in range(len(som_train_data))]
            trainer.train_som(data)
    elif trainer_str == 'MemAE':
        print(f"training on {device}")
        if dataset.name == 'Arrhythmia':
            enc_layers = [
                nn.Linear(D, 10), nn.Tanh(),
                nn.Linear(10, L)
            ]
            dec_layers = [
                nn.Linear(L, 10), nn.Tanh(),
                nn.Linear(10, D)
            ]
        elif dataset.name == 'Thyroid':
            enc_layers = [
                nn.Linear(D, 12),
                nn.Tanh(),
                nn.Linear(12, 4),
                nn.Tanh(),
                nn.Linear(4, L)
            ]
            dec_layers = [
                nn.Linear(L, 4),
                nn.Tanh(),
                nn.Linear(4, 12),
                nn.Tanh(),
                nn.Linear(12, D)
            ]
        else:
            enc_layers = [
                nn.Linear(D, 60), nn.Tanh(),
                nn.Linear(60, 30), nn.Tanh(),
                nn.Linear(30, 10), nn.Tanh(),
                nn.Linear(10, L)
            ]
            dec_layers = [
                nn.Linear(L, 10), nn.Tanh(),
                nn.Linear(10, 30), nn.Tanh(),
                nn.Linear(30, 60), nn.Tanh(),
                nn.Linear(60, D)
            ]
        model = MemAE(
            kwargs.get('mem_dim'), enc_layers, dec_layers, kwargs.get('shrink_thres'), device
        ).to(device)
        alpha = kwargs.get("alpha", 2e-4)
        trainer = MemAETrainer(
            alpha=alpha, model=model, device=device, lr=kwargs.get("learning_rate"), batch_size=batch_size
        )
    elif trainer_str == "DUAD":
        model = DUAD(D, 10)
        trainer = DUADTrainer(model=model, dm=dm, optimizer_factory=optimizer_factory, device=device,
                              p=kwargs.get('p_s'), p_0=kwargs.get('p_0'), r=kwargs.get('r'),
                              num_cluster=kwargs.get('num_cluster'))
    elif trainer_str == 'ALAD':
        # bsize = kwargs.get('batch_size', None)
        lr = kwargs.get('learning_rate', None)
        assert batch_size and lr
        model = ALAD(D, L, device=device).to(device)
        trainer = ALADTrainer(
            model=model,
            dm=dm,
            device=device,
            learning_rate=lr,
            L=L
        )
    elif trainer_str == 'DeepSVDD':
        model = DeepSVDD(D)
        trainer = DeepSVDDTrainer(
            model,
            optimizer_factory=optimizer_factory,
            dm=dm,
            R=kwargs.get('R'),
            c=kwargs.get('c'),
            device=device
        )

    elif trainer_str == 'DSEBM':
        # bsize = kwargs.get('batch_size', None)
        lr = kwargs.get('learning_rate', None)

        assert batch_size and lr
        model = DSEBM(D, dataset=dataset.name).to(device)
        trainer = DSEBMTrainer(
            model=model,
            dm=dm,
            device=device,
            batch=batch_size, dim=D, learning_rate=lr,
        )
    elif trainer_str == 'NeurTraAD':
        # bsize = kwargs.get('batch_size', None)
        lr = kwargs.get('learning_rate', None)

        assert batch_size and lr

        # Load a pretrained model in case it should be used

        model = NeuTraAD(D, device=device, temperature=0.07, dataset=dataset.name).to(device)
        trainer = NeuTraADTrainer(
            model=model,
            dm=dm,
            device=device,
            optimizer_factory=optimizer_factory,
            L=L, learning_rate=lr,
        )

    return model, trainer


if __name__ == "__main__":
    args = argument_parser()

    val_set = args.validation
    lambda_1 = args.lambda_energy
    lambda_2 = args.lambda_p
    L = args.latent_dim
    n_runs = args.n_runs

    n_som = args.n_som
    p_s = args.p_s
    p_0 = args.p_0
    r = args.r
    num_cluster = args.num_cluster

    # Dynamically load the Dataset instance
    clsname = globals()[f'{args.dataset}Dataset']
    dataset = clsname(args.dataset_path, args.pct)
    batch_size = len(dataset) if args.batch_size < 0 else args.batch_size

    # split data in train and test sets
    # we train only on the majority class

    train_set, test_set = dataset.one_class_split_train_test_inject(test_perc=0.50, inject_perc=args.rho)
    #dm = DataManager(train_set, test_set, batch_size=batch_size, validation=1e-3)

    # safely create save path
    check_dir(args.save_path)

    # For models based on sklearn like APIs
    if args.model in SKLEARN_MODEL:
        X_train, _ = get_X_from_loader(dm.get_train_set())
        X_test, y_test = get_X_from_loader(dm.get_test_set())
        print(f'Starting training: {args.model}')

        all_results = defaultdict(list)
        for r in range(n_runs):
            print(f"Run number {r}/{n_runs}")

            # Create the model with the appropriate parameters.
            if args.model == 'RECFOREST':
                model = RecForest(n_jobs=-1, random_state=-1)
            elif args.model == 'OC-SVM':
                print(f"Using nu = {args.nu}.")
                model = OneClassSVM(kernel='rbf', gamma='scale', shrinking=False, verbose=True, nu=args.nu)
            else:
                print(f"'{args.model}' is not a supported sklearn model.")
                exit(1)

            model.fit(X_train)
            print('Finished learning process')

            anomaly_score_train = []
            anomaly_score_test = []

            # prediction for the training set
            for i, X_i in enumerate(dm.get_train_set(), 0):
                # OC-SVM predicts -1 for outliers (and 1 for inliers), however we want outliers to be 1.
                # So we negate the predictions.
                if args.model == 'OC-SVM':
                    anomaly_score_train.append(-model.predict(X_i[0].numpy()))
                else:
                    anomaly_score_train.append(model.predict(X_i[0].numpy()))
            anomaly_score_train = np.concatenate(anomaly_score_train)

            # prediction for the test set
            y_test = []
            for i, X_i in enumerate(dm.get_test_set(), 0):
                # OC-SVM predicts -1 for outliers (and 1 for inliers), however we want outliers to be 1.
                # So we negate the predictions.
                if args.model == 'OC-SVM':
                    anomaly_score_test.append(-model.predict(X_i[0].numpy()))
                else:
                    anomaly_score_test.append(model.predict(X_i[0].numpy()))
                y_test.append(X_i[1].numpy())

            anomaly_score_test = np.concatenate(anomaly_score_test)
            y_test = np.concatenate(y_test)

            anomaly_score = np.concatenate([anomaly_score_train, anomaly_score_test])
            # dump metrics with different thresholds
            score_recall_precision(anomaly_score, anomaly_score_test, y_test)
            results = score_recall_precision_w_thresold(anomaly_score, anomaly_score_test, y_test, pos_label=1,
                                                        threshold=args.p_threshold)
            for k, v in results.items():
                all_results[k].append(v)

        params = dict({"BatchSize": batch_size, "Epochs": args.num_epochs, "rho": args.rho,
                       'threshold': args.p_threshold})

        print('Averaging results')
        final_results = average_results(all_results)
        store_results(final_results, params, args.model, args.dataset, args.dataset_path)
    else:

        optimizer = resolve_optimizer(args.optimizer)

        model, model_trainer = resolve_trainer(
            args.model, optimizer, latent_dim=L, mem_dim=args.mem_dim, shrink_thres=args.shrink_thres,
            n_som=n_som,
            r=r,
            p_s=p_s,
            p_0=p_0,
            num_cluster=num_cluster,
            reg_covar=args.reg_covar,
            learning_rate=args.lr,
        )

        if model and model_trainer:
            # Training and evaluation on different runs
            all_results = defaultdict(list)
            all_models = []
            thresh = 1. - dataset.anomaly_ratio

            if args.test_mode:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                for model_file_name in os.listdir(args.model_path):
                    model = BaseModel.load(f"{args.model_path}/{model_file_name}")
                    model = model.to(device)
                    model_trainer.model = model
                    print('Evaluating the model on test set')
                    # We test with the minority samples as the positive class
                    results, test_z, test_label, energy = model_trainer.evaluate_on_test_set(energy_threshold=thresh)
                    for k, v in results.items():
                        all_results[k].append(v)
            else:
                for r in range(n_runs):
                    print(f"Run number {r}/{n_runs}")
                    metrics = model_trainer.train(train_set)
                    print('Finished learning process')
                    print('Evaluating model on test set')
                    # We test with the minority samples as the positive class
                    results, test_z, test_label, energy = model_trainer.test(test_set)
                    for k, v in results.items():
                        all_results[k].append(v)
                    all_models.append(deepcopy(model))
                    model.reset()

            # Calculate Means and Stds of metrics
            print('Averaging results')
            final_results = average_results(all_results)

            params = dict({"BatchSize": batch_size, "Epochs": args.num_epochs, "rho": args.rho,
                           'threshold': args.p_threshold}, **model.get_params())
            # Store the average of results
            store_results(final_results, params, args.model, args.dataset, args.dataset_path)

            # Persist models
            if not args.test_mode:
                store_models(all_models, args.model, args.dataset, args.dataset_path)

            if args.vizualization and args.model in vizualizable_models:
                plot_3D_latent(test_z, test_label)
                plot_energy_percentile(energy)
        else:
            print(f'Error: Could not train {args.dataset} on model {args.model}')

