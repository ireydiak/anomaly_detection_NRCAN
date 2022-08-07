import pickle
import re

import os
from collections import defaultdict
from datetime import datetime as dt

from pyad.legacy.model.DUAD import DUAD
from pyad.legacy.model.reconstruction import SOMDAGMM
from pyad.legacy.trainer.reconstruction import SOMDAGMMTrainer
from pyad.legacy.trainer.DUADTrainer import DUADTrainer
from pyad.utils import metrics
from pyad.utils.utils import average_results, ids_misclf_per_label
from pyad.legacy.datamanager.DataManager import DataManager
from pyad.legacy.datamanager.dataset import *

available_models = [
    DUAD,
    SOMDAGMM,
]

datasets_map = {
    "Arrhythmia": ArrhythmiaDataset,
    "KDD10": KDD10Dataset,
    "MalMem2022": MalMem2022Dataset,
    "NSLKDD": NSLKDDDataset,
    "IDS2018": IDS2018Dataset,
    "IDS2017": IDS2017Dataset,
    "USBIDS": IDS2018Dataset,
    "Thyroid": ThyroidDataset
}

model_trainer_map = {
    "DUAD": (DUAD, DUADTrainer),
    "SOMDAGMM": (SOMDAGMM, SOMDAGMMTrainer),
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
    output_dir = models_path or f'../models/{dataset}/{model_name}/{dt.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(f"{output_dir}/{model_name}.pt")


def resolve_dataset(name: str, path: str, normal_size=1):
    dataset_cls = datasets_map.get(name, None)
    if not dataset_cls:
        raise Exception("unknown dataset %s" % name)
    return dataset_cls(
        path=path,
        normal_size=normal_size
    )


def resolve_model_trainer(
        model_name: str,
        model_params: dict,
        dataset: AbstractDataset,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        device: str,
        datamanager=None
):
    model_trainer_tuple = model_trainer_map.get(model_name, None)
    assert model_trainer_tuple, "Model %s not found" % model_name
    model_cls, trainer_cls = model_trainer_tuple

    model = model_cls(
        dataset_name=dataset.name,
        device=device,
        **model_params,
    )
    trainer = trainer_cls(
        model=model,
        lr=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device,
        weight_decay=weight_decay,
        dm=datamanager
    )

    return model, trainer


def train_model(
        model_name: str,
        model_params: dict,
        n_epochs: int,
        weight_decay: float,
        learning_rate: float,
        device: str,
        model_path: str,
        test_mode: bool,
        dataset,
        batch_size,
        seed,
):
    model, model_trainer = resolve_model_trainer(
        model_name=model_name,
        model_params=model_params,
        dataset=dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        device=device
    )

    train_ldr, test_ldr, val_ldr = dataset.loaders(
        batch_size=batch_size,
        seed=seed,
    )

    if model.name == "DUAD":
        # DataManager for DUAD only
        # split data in train and test sets
        train_set, test_set, val_set = dataset.split_train_test(
            test_pct=0.50,
        )
        dm = DataManager(train_set, test_set, batch_size=batch_size)
        # we train only on the majority class
        model_trainer.setDataManager(dm)
        model_trainer.train(dataset=train_set)
    else:
        model_trainer.train(train_ldr)
    print("Completed learning process")
    print("Evaluating model on test set")
    # We test with the minority samples as the positive class
    if model.name == "DUAD":
        scores, y_true, labels = model_trainer.evaluate_on_test_set()
    else:
        scores, y_true, labels = model_trainer.test(test_ldr)

    results, y_pred = metrics.estimate_optimal_threshold(scores, y_true)
    if len(np.unique(labels)) > 2:
        misclf_df = ids_misclf_per_label(y_pred, y_true, labels)
        misclf_df = misclf_df.sort_values("Misclassified ratio", ascending=False)
        for i, row in misclf_df.iterrows():
            results[i] = row["Accuracy"]

    return results


def merge_cfg(priority, defaults):
    cfg = defaultdict()
    for key, val in defaults:
        cfg[key] = priority.get(
            key, val
        )
    return cfg


def train_from_cfg(datasets: dict, models: dict, base_path: str):
    # simple reporting system
    reporter = defaultdict()
    # for every dataset ...
    for dataset_name, dataset_cfg in datasets.items():
        reporter[dataset_name] = defaultdict()
        dataset_cls = datasets_map[dataset_name]
        dataset_cfg = merge_cfg(dataset_cfg, dataset_cls.get_default_cfg())
        dataset = dataset_cls(path=dataset_cfg["path"], normal_size=dataset_cfg["normal_size"])
        # for every model ...
        for model_name, model_params in models.items():
            reporter[dataset_name][model_name] = {
                "Precision": [], "Recall": [], "F1-Score": [], "AUROC": [], "AUPR": [], "Thresh": [],
                "params": model_params["INIT_PARAMS"]
            }
            # create base folder structure
            results_path = os.path.join(base_path, dataset_name.lower(), model_name.lower())
            os.makedirs(results_path, exist_ok=True)
            # repeat experiments `n_runs` times
            for run_i in range(dataset_cfg["n_runs"]):
                # create folder structure for checkpoints
                ckpt_path = os.path.join(results_path, "run_{}".format(run_i + 1))
                os.makedirs(ckpt_path, exist_ok=True)
                # look for existing checkpoints
                ckpt_epochs = sorted([int(re.findall(r'\d+', s)[0]) for s in os.listdir(ckpt_path)])
                model_cls, trainer_cls = model_trainer_map.get(model_name)
                if ckpt_epochs:
                    # load latest checkpoint
                    last = ckpt_epochs[-1]
                    ckpt_file = "%s_epoch=%d.pt" % (model_name, last)
                    trainer, model = trainer_cls.load_from_file(ckpt_file)
                else:
                    # setup model
                    model = model_cls(
                        in_features=dataset.in_features,
                        n_instances=dataset.n_instances,
                        **model_params["INIT_PARAMS"]
                    )
                    # setup trainer
                    trainer = trainer_cls(
                        model=model,
                        **merge_cfg(model_params["TRAINER"], trainer_cls.get_default_cfg()),
                    )
                # data loaders
                train_ldr, test_ldr, _ = dataset.loaders(
                    batch_size=dataset_cfg["batch_size"],
                    contamination_rate=dataset_cfg["contamination_rate"],
                    validation_ratio=dataset_cfg["val_ratio"],
                    holdout=dataset_cfg["holdout"],
                )
                # train
                trainer.train(train_ldr)
                # test & evaluate
                y_test_true, test_scores, _ = trainer.test(test_ldr)
                results, _ = metrics.score_recall_precision_w_threshold(test_scores, y_test_true)
                # compile results for current run
                for k, v in results.items():
                    reporter[dataset_name][model_name][k].append(v)
            # compute mean, std for every performance metrics
            for k in ["Precision", "Recall", "F1-Score", "AUROC", "AUPR"]:
                v = reporter[dataset_name][model_name][k]
                reporter[dataset_name][model_name][k] = f"{np.mean(v):.4f}({np.std(v):.4f})"
    # store reporter
    with open("reporter.pkl", "wb") as f:
        pickle.dump(reporter, f)


def train(
        model_name: str,
        model_params: dict,
        dataset_name: str,
        dataset_path: str,
        batch_size: int,
        normal_size: float,
        n_runs: int,
        n_epochs: int,
        learning_rate: float,
        weight_decay: float,
        results_path: str,
        models_path: str,
        test_mode: bool,
        seed: int,
):
    # Dynamically load the Dataset instance
    dataset_cls = datasets_map[dataset_name]
    dataset = dataset_cls(path=dataset_path, normal_size=normal_size)
    anomaly_thresh = 1 - dataset.anomaly_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # check path
    for p in [results_path, models_path]:
        if p:
            assert os.path.exists(p), "Path %s does not exist" % p
    # Training and evaluation on different runs
    all_results = defaultdict(list)
    print("Training model {} for {} epochs on device {}".format(model_name, n_epochs, device))

    for i in range(n_runs):
        print(f"Run {i + 1} of {n_runs}")
        results = train_model(
            # model=model,
            # model_trainer=model_trainer,
            # dataset_name=dataset_name,
            model_name=model_name,
            model_params=model_params,
            n_epochs=n_epochs,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            device=device,
            model_path=models_path,
            test_mode=test_mode,
            dataset=dataset,
            batch_size=batch_size,
            seed=seed,
        )
        for k, v in results.items():
            all_results[k].append(v)
        print(results)
    all_results = average_results(all_results)
    params = dict(
        {"BatchSize": batch_size,
         "Epochs": n_epochs,
         "Threshold": anomaly_thresh},
        **model_params
    )
    # Store the average of results
    fname = store_results(all_results, params, model_name, dataset.name, dataset_path, results_path)
    print(f"Results stored in {fname}")
