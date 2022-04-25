import numpy as np
import torch
import os
from collections import defaultdict
from datetime import datetime as dt

from torch.utils.data import DataLoader
from src.model.adversarial import ALAD

from src.model.base import BaseModel
from src.model.density import DSEBM
from src.model.DUAD import DUAD
from src.model.one_class import DeepSVDD, DROCC
from src.model.transformers import NeuTraLAD
from src.model.reconstruction import AutoEncoder as AE, DAGMM, MemAutoEncoder as MemAE, SOMDAGMM
from src.model.shallow import RecForest, OCSVM, LOF
from src.trainer.adversarial import ALADTrainer
from src.trainer.density import DSEBMTrainer
from src.trainer.one_class import DeepSVDDTrainer, EdgeMLDROCCTrainer
from src.trainer.reconstruction import AutoEncoderTrainer as AETrainer, DAGMMTrainer, MemAETrainer, SOMDAGMMTrainer
from src.trainer.shallow import OCSVMTrainer, RecForestTrainer, LOFTrainer
from src.trainer.transformers import NeuTraLADTrainer
from src.trainer.DUADTrainer import DUADTrainer
from src.utils import metrics
from src.utils.utils import average_results
from src.datamanager.DataManager import DataManager
from src.datamanager.dataset import *

available_models = [
    AE,
    # "ALAD",
    DAGMM,
    DeepSVDD,
    # "DSEBM",
    DROCC,
    # "DUAD",
    # "LOF",
    MemAE,
    # "NeuTraLAD",
    # "OC-SVM",
    # "RecForest",
    SOMDAGMM,
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

model_trainer_map = {
    # Deep Models
    # "ALAD": (ALAD, ALADTrainer),
    "AE": (AE, AETrainer),
    "DAGMM": (DAGMM, DAGMMTrainer),
    # "DSEBM": (DSEBM, DSEBMTrainer),
    "DROCC": (DROCC, EdgeMLDROCCTrainer),
    # "DUAD": (DUAD, DUADTrainer),
    "MemAE": (MemAE, MemAETrainer),
    "DeepSVDD": (DeepSVDD, DeepSVDDTrainer),
    "SOMDAGMM": (SOMDAGMM, SOMDAGMMTrainer),
    # "NeuTraLAD": (NeuTraLAD, NeuTraLADTrainer),
    # Shallow Models
    # "OC-SVM": (OCSVM, OCSVMTrainer),
    # "LOF": (LOF, LOFTrainer),
    # "RecForest": (RecForest, RecForestTrainer)
}


def store_results(
    results: dict,
    params: dict,
    model_name: str,
    dataset: str,
    dataset_path: str,
    results_path: str = None
):
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
    output_dir = models_path or f'../models/{dataset}/{model_name}/{dt.now().strftime("%d_%m_%Y_%H_%M_%S")}"'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(f"{output_dir}/{model_name}.pt")


def resolve_model_trainer(
        model_name: str,
        model_params: dict,
        dataset: AbstractDataset,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        device: str,
        # duad_r,
        # duad_p_s,
        # duad_p_0,
        # duad_num_cluster,
        # datamanager: DataManager = None,
):
    # TODO: dead code
    # if model_name == "DUAD":
    #     model = DUAD(
    #         dataset.in_features,
    #         10,
    #         dataset_name=dataset.name,
    #         in_features=dataset.in_features,
    #         n_instances=dataset.n_instances,
    #         device=device
    #     )
    #     trainer = DUADTrainer(
    #         model=model,
    #         dm=datamanager,
    #         device=device,
    #         n_epochs=n_epochs,
    #         duad_p_s=duad_p_s,
    #         duad_p_0=duad_p_0,
    #         duad_r=duad_r,
    #         duad_num_cluster=duad_num_cluster,
    #         lr=learning_rate,
    #         weight_decay=weight_decay
    #     )
    model_trainer_tuple = model_trainer_map.get(model_name, None)
    assert model_trainer_tuple, "Model %s not found" % model_name
    model, trainer = model_trainer_tuple

    model = model(
        dataset_name=dataset.name,
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        device=device,
        **model_params,
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
        train_ldr: DataLoader,
        test_ldr: DataLoader,
        dataset_name: str,
        n_runs: int,
        thresh: float,
        device: str,
        model_path: str,
        test_mode: bool
):
    # Training and evaluation on different runs
    all_results = defaultdict(list)
    print("Training model {} for {} epochs on device {}".format(model.name, model_trainer.n_epochs, device))
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
                model_trainer.train()
            else:
                _ = model_trainer.train(train_ldr)
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
            store_model(model, model.name, dataset_name, model_path)
            model.reset()

    # Compute mean and standard deviation of the performance metrics
    print("Averaging results ...")
    return average_results(all_results)


def train(
        model_name: str,
        model_params: dict,
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
        # duad_r,
        # duad_p_s,
        # duad_p_0,
        # duad_num_cluster,
):
    # Dynamically load the Dataset instance
    clsname = globals()[f'{dataset_name}Dataset']
    dataset = clsname(path=dataset_path, pct=pct)
    anomaly_thresh = 1 - dataset.anomaly_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # split data in train and test sets
    # we train only on the majority class
    if model_name == "DUAD":
        # DataManager for DUAD only
        train_set, test_set = dataset.split_train_test(test_pct=0.50)
        dm = DataManager(train_set, test_set, batch_size=batch_size)
        train_ldr = None,
        test_ldr = None
    else:
        train_ldr, test_ldr = dataset.loaders(batch_size=batch_size, seed=seed)
        dm = None

    # check path
    for p in [results_path, models_path]:
        if p:
            assert os.path.exists(p), "Path %s does not exist" % p

    model, model_trainer = resolve_model_trainer(
        model_name=model_name,
        model_params=model_params,
        dataset=dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device=device,
        # duad_r=duad_r,
        # duad_p_s=duad_p_s,
        # duad_p_0=duad_p_0,
        # duad_num_cluster=duad_num_cluster,
        # datamanager=dm,
    )
    res = train_model(
        model=model,
        model_trainer=model_trainer,
        train_ldr=train_ldr,
        test_ldr=test_ldr,
        dataset_name=dataset_name,
        n_runs=n_runs,
        device=device,
        thresh=anomaly_thresh,
        model_path=models_path,
        test_mode=test_mode
    )
    print(res)
    params = dict(
        {"BatchSize": batch_size, "Epochs": n_epochs, "CorruptionRatio": corruption_ratio,
         "Threshold": anomaly_thresh},
        **model.get_params()
    )
    # Store the average of results
    fname = store_results(res, params, model_name, dataset.name, dataset_path, results_path)
    print(f"Results stored in {fname}")
