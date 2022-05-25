import argparse
import os
import re

import numpy as np

from src.bootstrap import store_results
from src.datamanager.dataset import ArrhythmiaDataset, ThyroidDataset, IDS2017Dataset, IDS2018Dataset
from src.trainer.adversarial import ALADTrainer
from src.trainer.density import DSEBMTrainer
from src.trainer.one_class import DeepSVDDTrainer, EdgeMLDROCCTrainer
from src.trainer.reconstruction import AutoEncoderTrainer, MemAETrainer, DAGMMTrainer
from src.trainer.transformers import NeuTraLADTrainer
from src.model.adversarial import ALAD
from src.model.density import DSEBM
from src.model.one_class import DeepSVDD, DROCC
from src.model.reconstruction import AutoEncoder, DAGMM, MemAutoEncoder
from src.model.transformers import NeuTraLAD
from src.utils import metrics
from src.utils.utils import ids_misclf_per_label
from train import settings
import pandas as pd


def argument_parser():
    parser = argparse.ArgumentParser(
        usage="\n python testing.py"
              "--dataset [dataset name] --dataset-path [path to dataset]"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
        required=False,
        default=1024
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="Name of the dataset",
        required=True
    )
    parser.add_argument(
        "-p",
        "--dataset-path",
        type=str,
        help="Path to the dataset",
        required=True
    )
    return parser.parse_args()


def resolve_dataset(name, path):
    name = name.lower()
    if name == "arrhythmia":
        return ArrhythmiaDataset(path=path)
    elif name == "ids2017":
        return IDS2017Dataset(path=path)
    elif name == "ids2018":
        return IDS2018Dataset(path=path)
    elif name == "thyroid":
        return ThyroidDataset(path=path)
    else:
        raise Exception("unsupported dataset %s, aborting" % name)


def test():
    args = argument_parser()
    batch_size = args.batch_size
    dataset = resolve_dataset(args.dataset, args.dataset_path)
    train_ldr, test_ldr = dataset.loaders(batch_size=batch_size, seed=42)
    attack_types = np.unique(dataset.labels)
    summary, measures = [], ["Precision", "Recall", "F1-Score", "AUPR"]
    ckpt_root = args.dataset.lower()
    print("data loaded with shape {}".format(dataset.shape))
    for model_name, params in settings.items():
        model_root = os.path.join(ckpt_root, model_name.lower())
        print("Testing model %s on %s" % (model_name, args.dataset))
        # Load final model
        ckpt_files = [
            int(re.findall(r'\d+', p)[0]) for p in os.listdir(os.path.join(model_root, "checkpoints"))
        ]
        ckpt_files.sort()
        ckpt_f = os.path.join(model_root, "checkpoints", model_name.lower() + "_epoch={}.pt".format(ckpt_files[-1]))
        trainer, model = params["trainer_cls"].load_from_file(ckpt_f)
        best_epoch = np.argmax(trainer.metric_values["f1-score"]) * 5 + 1
        # Load best model
        best_ckpt_f = os.path.join(model_root, "checkpoints", model_name.lower() + "_epoch={}.pt".format(best_epoch))
        print("loading checkpoint {}".format(best_ckpt_f))
        trainer, model = trainer.load_from_file(best_ckpt_f)
        # Predict anomalies on test set
        y_test_true, test_scores, test_labels = trainer.test(test_ldr)
        results, y_pred = metrics.score_recall_precision_w_threshold(test_scores, y_test_true)
        store_results(
            results=results,
            params=dict(**trainer.get_params(), **model.get_params()),
            model_name=model_name,
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            results_path=model_root
        )
        # Convert binary classifications to multi-class
        clf_df = ids_misclf_per_label(y_pred, y_test_true, test_labels)
        clf_df = clf_df.sort_values("Accuracy", ascending=False)
        clf_df.to_csv(os.path.join(
            model_root, "{}_class_predictions.csv".format(model_name)
        ))
        # Add to summary
        tmp = []
        for atk_label in attack_types:
            tmp.append("{:2.2f}".format(clf_df.loc[atk_label, "Accuracy"] * 100))
        for metric in measures:
            tmp.append(
                "{:2.2f}".format(results[metric]*100)
            )
        if len(summary) == 0:
            summary = np.array(tmp).reshape(len(tmp), 1)
        else:
            summary = np.concatenate((
                summary, np.array(tmp).reshape(len(tmp), 1)
            ), axis=1)
    # Create summary dataframe
    summary_df = pd.DataFrame(
        summary, index=list(attack_types) + measures, columns=settings.keys()
    )
    summary_df.to_csv("{}/_export/{}_1_run_summary.csv".format(args.dataset.lower(), args.dataset.lower()))
    summary_df.to_latex("{}/_export/{}_1_run_summary.tex".format(args.dataset.lower(), args.dataset.lower()))


if __name__ == "__main__":
    test()
