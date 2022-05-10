import numpy as np
import torch
import os
from collections import defaultdict
from datetime import datetime as dt

from torch.utils.data import DataLoader
from model.adversarial import ALAD

from model.base import BaseModel
from model.density import DSEBM
from model.DUAD import DUAD
from model.one_class import DeepSVDD, DROCC
from model.transformers import NeuTraLAD
from model.reconstruction import AutoEncoder as AE, DAGMM, MemAutoEncoder as MemAE, SOMDAGMM
from model.shallow import RecForest, OCSVM, LOF
from trainer.adversarial import ALADTrainer
from trainer.density import DSEBMTrainer
from trainer.one_class import DeepSVDDTrainer, EdgeMLDROCCTrainer
from trainer.reconstruction import AutoEncoderTrainer as AETrainer, DAGMMTrainer, MemAETrainer, SOMDAGMMTrainer
from trainer.shallow import OCSVMTrainer, RecForestTrainer, LOFTrainer
from trainer.transformers import NeuTraLADTrainer
from trainer.DUADTrainer import DUADTrainer
from utils import metrics
from utils.utils import average_results
from datamanager.DataManager import DataManager
from datamanager.dataset import *

available_models = [
    "AE",
    "ALAD",
    "DAGMM",
    "DeepSVDD",
    "DSEBM",
    "DROCC",
    "DUAD",
    "LOF",
    "MemAE",
    "NeuTraLAD",
    "OC-SVM",
    "RecForest",
    "SOM-DAGMM",
]
available_datasets = [
    "Arrhythmia",
    "KDD10",
    "MalMem2022",
    "NSLKDD",
    "IDS2018",
    "USBIDS",
    "Thyroid"
]


def store_results(results: dict, params: dict, model_name: str, dataset: str, dataset_path: str,
                  results_path: str = None):
    output_dir = results_path or f"../results/{dataset}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fname = output_dir + '/' + f'{model_name}_results.txt'
    with open(fname, 'a') as f:
        hdr = "Experiments on {}\n".format(dt.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({dataset_path.split("/")[-1].split(".")[0]})\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
    return fname


def store_model(model, model_name: str, dataset: str, models_path: str = None):
    output_dir = models_path or f'../models/{dataset}/{model_name}/{dt.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(f"{output_dir}/{model_name}.pt")


model_trainer_map = {
    # Deep Models
    "ALAD": (ALAD, ALADTrainer),
    "AE": (AE, AETrainer),
    "DAGMM": (DAGMM, DAGMMTrainer),
    "DSEBM": (DSEBM, DSEBMTrainer),
    "DROCC": (DROCC, EdgeMLDROCCTrainer),
    "DUAD": (DUAD, DUADTrainer),
    "MemAE": (MemAE, MemAETrainer),
    "DeepSVDD": (DeepSVDD, DeepSVDDTrainer),
    "SOM-DAGMM": (SOMDAGMM, SOMDAGMMTrainer),
    "NeuTraLAD": (NeuTraLAD, NeuTraLADTrainer),
    # Shallow Models
    "OC-SVM": (OCSVM, OCSVMTrainer),
    "LOF": (LOF, LOFTrainer),
    "RecForest": (RecForest, RecForestTrainer)
}


def resolve_model_trainer(
        model_name: str,
        dataset: AbstractDataset,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        device: str,
        duad_r,
        duad_p_s,
        duad_p_0,
        duad_num_cluster,
        ae_latent_dim,
        datamanager: DataManager = None,
):
    if model_name == "DUAD":
        model = DUAD(
            r=duad_r,
            p0=duad_p_0,
            ps=duad_p_s,
            dataset_name=dataset.name,
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            device=device,
            ae_latent_dim=ae_latent_dim
        )
        trainer = DUADTrainer(
            model=model,
            dm=datamanager,
            device=device,
            n_epochs=n_epochs,
            duad_p_s=duad_p_s,
            duad_p_0=duad_p_0,
            duad_r=duad_r,
            duad_num_cluster=duad_num_cluster,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        model_trainer_tuple = model_trainer_map.get(model_name, None)
        assert model_trainer_tuple, "Model %s not found" % model_name
        model, trainer = model_trainer_tuple

        model = model(
            dataset_name=dataset.name,
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            device=device,
            ae_latent_dim=ae_latent_dim
        )
        trainer = trainer(
            model=model,
            lr=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=device,
            weight_decay=weight_decay
        )

    return model, trainer


def train_model(
        model: BaseModel,
        model_trainer,

        dataset_name: str,
        n_runs: int,
        device: str,
        model_path: str,
        test_mode: bool,
        dataset,
        batch_size,
        seed,
        contamination_rate,
        holdout,
        drop_lastbatch,
        validation_ratio

):
    # Training and evaluation on different runs
    all_results = defaultdict(list)

    train_ldr, test_ldr, val_ldr = dataset.loaders(batch_size=batch_size,
                                                   seed=seed,
                                                   contamination_rate=contamination_rate,
                                                   validation_ratio=validation_ratio,
                                                   holdout=holdout, drop_last_batch=drop_lastbatch)
    if seed:
        torch.manual_seed(seed)

    if test_mode:
        for model_file_name in os.listdir(model_path):
            model = BaseModel.load(f"{model_path}/{model_file_name}")
            model = model.to(device)
            model_trainer.model = model
            print("Evaluating the model on test set")
            # We test with the minority samples as the positive class
            # y_train_true, train_scores = model_trainer.test(train_ldr)
            y_test_true, test_scores = model_trainer.test(test_ldr)
            # y_true = np.concatenate((y_train_true, y_test_true), axis=0)
            # scores = np.concatenate((train_scores, test_scores), axis=0)
            print("Evaluating model")
            results = metrics.estimate_optimal_threshold(test_scores, y_test_true)
            for k, v in results.items():
                all_results[k].append(v)
    else:
        for i in range(n_runs):
            print(f"Run {i + 1} of {n_runs}")
            if model.name == "DUAD":
                # DataManager for DUAD only
                # split data in train and test sets
                train_set, test_set, val_set = dataset.split_train_test(test_pct=0.50,
                                                               contamination_rate=contamination_rate,
                                                               holdout=holdout)
                dm = DataManager(train_set, test_set, batch_size=batch_size, drop_last=drop_lastbatch)
                # we train only on the majority class
                model_trainer.setDataManager(dm)
                model_trainer.train()
            else:
                _ = model_trainer.train(train_ldr, val_ldr)
            print("Completed learning process")
            print("Evaluating model on test set")
            # We test with the minority samples as the positive class
            if model.name == "DUAD":
                test_scores, y_test_true = model_trainer.evaluate_on_test_set()
            else:
                # y_train_true, train_scores = model_trainer.test(train_ldr)
                y_test_true, test_scores = model_trainer.test(test_ldr)
                # y_true = np.concatenate((y_train_true, y_test_true), axis=0)
                # scores = np.concatenate((train_scores, test_scores), axis=0)
            results = metrics.estimate_optimal_threshold(test_scores, y_test_true)
            print(results)
            for k, v in results.items():
                all_results[k].append(v)
            store_model(model, model.name, dataset_name, None)
            model.reset()

            if i < n_runs - 1:
                train_ldr, test_ldr, val_ldr = dataset.loaders(batch_size=batch_size,
                                                               seed=seed,
                                                               contamination_rate=contamination_rate,
                                                               validation_ratio=validation_ratio,
                                                               holdout=holdout, drop_last_batch=drop_lastbatch)

    # Compute mean and standard deviation of the performance metrics
    print("Averaging results ...")
    return average_results(all_results)


def train(
        model_name: str,
        dataset_name: str,
        dataset_path: str,
        batch_size: int,
        pct: float,
        corruption_ratio: float,
        n_runs: int,
        n_epochs: int,
        learning_rate: float,
        weight_decay: float,
        results_path: str,
        models_path: str,
        test_mode: bool,
        seed: int,
        duad_r,
        duad_p_s,
        duad_p_0,
        duad_num_cluster,
        ae_latent_dim,
        holdout=0.0,
        contamination_r=0.0,
        drop_lastbatch=False,
        validation_ratio=.2,
):
    # Dynamically load the Dataset instance
    clsname = globals()[f'{dataset_name}Dataset']
    dataset = clsname(path=dataset_path, pct=pct)
    anomaly_thresh = 1 - dataset.anomaly_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check path
    for p in [results_path, models_path]:
        if p:
            assert os.path.exists(p), "Path %s does not exist" % p

    model, model_trainer = resolve_model_trainer(
        model_name=model_name,
        dataset=dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device=device,
        duad_r=duad_r,
        duad_p_s=duad_p_s,
        duad_p_0=duad_p_0,
        duad_num_cluster=duad_num_cluster,
        datamanager=None,
        ae_latent_dim=ae_latent_dim
    )
    res = train_model(
        model=model,
        model_trainer=model_trainer,
        dataset_name=dataset_name,
        n_runs=n_runs,
        device=device,
        model_path=models_path,
        test_mode=test_mode,
        dataset=dataset,
        batch_size=batch_size,
        seed=seed,
        contamination_rate=contamination_r,
        holdout=holdout,
        drop_lastbatch=drop_lastbatch,
        validation_ratio=validation_ratio,

    )
    print(res)
    params = dict(
        {"BatchSize": batch_size, "Epochs": n_epochs, "CorruptionRatio": corruption_ratio,
         "HoldoutRatio": holdout,
         "Threshold": anomaly_thresh},
        **model.get_params()
    )
    # Store the average of results
    fname = store_results(res, params, model_name, dataset.name, dataset_path, results_path)
    print(f"Results stored in {fname}")
