import argparse
import os
import types
from collections import defaultdict

import numpy as np
import itertools as it

import pandas as pd

from src.datamanager.dataset import ArrhythmiaDataset, ThyroidDataset, IDS2017Dataset, IDS2018Dataset
from src.model.shallow import PCA
from src.trainer.adversarial import ALADTrainer
from src.trainer.density import DSEBMTrainer
from src.trainer.one_class import DeepSVDDTrainer, EdgeMLDROCCTrainer
from src.trainer.reconstruction import AutoEncoderTrainer, MemAETrainer, DAGMMTrainer
from src.trainer.shallow import PCATrainer
from src.trainer.transformers import NeuTraLADTrainer
from src.model.adversarial import ALAD
from src.model.density import DSEBM
from src.model.one_class import DeepSVDD, DROCC
from src.model.reconstruction import AutoEncoder, DAGMM, MemAutoEncoder
from src.model.transformers import NeuTraLAD
from pathlib import Path
from src.utils import metrics
from train import resolve_dataset


def argument_parser():
    parser = argparse.ArgumentParser(
        usage="\n python tune.py -d [dataset-path]"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help="Batch size",
        required=False,
        default=1024
    )
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        help='Name of the dataset',
        required=True
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Path to the dataset',
        required=True
    )
    parser.add_argument(
        "--n-epochs",
        help="Number of training epochs",
        type=int,
        default=200
    )
    return parser.parse_args()


n_layers = 4
compression_factor = 2

settings = {
    "PCA": {
        "model_cls": PCA,
        "trainer_cls": PCATrainer,
        "tunable_params": {
            "n_components": lambda x: np.arange(1, x.shape[1] - 1)
        }
    }
}


def resolve_tunable_params(dataset, tunable_params):
    params = defaultdict()
    for param_name, param_value in tunable_params.items():
        if isinstance(param_value, types.FunctionType):
            params[param_name] = param_value(dataset)
        else:
            params[param_name] = param_value
    return params


def tune():
    args = argument_parser()
    batch_size = args.batch_size
    lr = 1e-4
    n_epochs = args.n_epochs
    dataset = resolve_dataset(args.dataset, args.dataset_path)
    train_ldr, test_ldr = dataset.loaders(batch_size=batch_size, seed=42)
    print("data loaded with shape {}".format(dataset.shape))
    print("Tuning model %s on %s" % ("PCA", dataset.name))
    params = np.arange(1, dataset.shape[1] - 1)
    indexes = params
    cols = ["n_components"]
    results = []
    df = pd.DataFrame(results, index=indexes, columns=cols)
    for i, n_components in enumerate(params):
        # print("tuning with params={}".format({"n_components": n_components}))
        # Setup model and trainer
        model = PCA(n_components=n_components)
        trainer = PCATrainer(
            model=model,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            device="cuda",
            validation_ldr=None,
            ckpt_root=None
        )
        # Train model
        trainer.train(train_ldr)
        # Test model
        y_true, scores, _ = trainer.test(test_ldr)
        res, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
        precision, recall, f1 = res["Precision"], res["Recall"], res["F1-Score"]
        results.append(f1)
        print("Precision={:2.4f}, Recall={:2.4f}, F1-Score={:2.4f}".format(precision, recall, f1))
        df = pd.DataFrame(results, index=indexes[:i+1], columns=cols)
        print(df)
    best_idx = np.argmax(results)
    print("Best F1-Score={:2.4f} with param={}".format(results[best_idx], params[best_idx]))
    df.to_csv("{}_tuning.csv".format("pca"))
    # for model_name in settings.keys():
    #     print("Tuning model %s on %s" % (model_name, dataset.name))
    #     tunable_params = resolve_tunable_params(dataset, settings[model_name]["tunable_params"].values())
    #     df_index = list(tunable_params.keys())
    #     df_values = {param_name: [] for param_name in df_index}
    #     df_results = {param_name: [] for param_name in df_index}
    #     for params in tunable_params:
    #         print("Tuning with params={}".format(params))
    #         # Initialize model and trainer
    #         model_params = dict(
    #             **{"in_features": dataset.in_features, "n_instances": dataset.n_instances},
    #             **params
    #         )
    #         model = params["model_cls"](**model_params)
    #         trainer = params["trainer_cls"](
    #             model=model,
    #             batch_size=batch_size,
    #             lr=lr,
    #             n_epochs=n_epochs,
    #             device="cuda",
    #             validation_ldr=None,
    #             ckpt_root=None
    #         )
    #         # Train model
    #         trainer.train(train_ldr)
    #         # Test model
    #         y_true, scores, _ = trainer.test(test_ldr)
    #         res, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
    #         precision, recall, f1 = res["Precision"], res["Recall"], res["F1-Score"]
    #         print("Precision={:2.4f}, Recall={:2.4f}, F1-Score={:2.4f}".format(precision, recall, f1))
    #         for p in params:
    #             df_values[p]


if __name__ == "__main__":
    tune()
