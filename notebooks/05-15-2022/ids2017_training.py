import pandas as pd
import numpy as np
import os
from pyad.model.one_class import DeepSVDD
from pyad.model.reconstruction import DAGMM, MemAutoEncoder
from pyad.trainer.ids import DeepSVDDIDSTrainer, MemAEIDSTrainer, DAGMMIDSTrainer
from pyad.utils import metrics
from pyad.utils.utils import ids_misclf_per_label
from pyad.datamanager.dataset import IDS2017Dataset
from pyad.bootstrap import store_results
from pathlib import Path

from pyad.trainer.ids import AEIDSTrainer
from pyad.model.reconstruction import AutoEncoder

models_params = {
    "autoencoder": {
        "model_cls": AutoEncoder,
        "trainer_cls": AEIDSTrainer,
        "compression_factor": 2,
        "n_layers": 4,
        "act_fn": "relu",
        "keep_ckpt": False,
        "reg": 0.5,
    },
    "deepsvdd": {
        "model_cls": DeepSVDD,
        "trainer_cls": DeepSVDDIDSTrainer,
        "compression_factor": 2,
        "n_layers": 4,
        "act_fn": "relu",
        "keep_ckpt": False,
    },
    "dagmm": {
        "model_cls": DAGMM,
        "trainer_cls": DAGMMIDSTrainer,
        "n_mixtures": 4,
        "latent_dim": 1,
        "lambda_1": 0.005,
        "lambda_2": 0.1,
        "reg_covar": 1e-6,
        "compression_factor": 2,
        "act_fn": "relu",
        "n_layers": 4,
        "keep_ckpt": False,
    },
    "memae": {
        "model_cls": MemAutoEncoder,
        "trainer_cls": MemAEIDSTrainer,
        "mem_dim": 50,
        "latent_dim": 1,
        "shrink_thres": 0.0025,
        "n_layers": 4,
        "compression_factor": 2,
        "act_fn": "relu",
        "alpha": 2e-4,
        "keep_ckpt": False,
    }
}
def train():
    # Setup base folder structure
    dataset_path = "../../data/IDS2017/ids2017.csv"

    # General arguments
    batch_size = 1024
    device = "cuda"
    act_fn = "relu"
    n_layers = 4
    compression_factor = 2
    lr = 1e-4
    n_epochs = 200
    n_runs = 20

    # Load data
    dataset = IDS2017Dataset(path=dataset_path)
    attack_types = np.unique(dataset.labels)
    # Store label-wise performance
    performance_hist = {
        model_name: {attack_label: [] for attack_label in attack_types} for model_name in models_params.keys()
    }

    for model_name, params in models_params.items():
        print("Training model {} for {} runs of {} epochs each".format(model_name, n_runs, n_epochs))
        model_cls, trainer_cls = params.pop("model_cls"), params.pop("trainer_cls")
        model_params = dict(
            **params, in_features=dataset.in_features, n_instances=dataset.n_instances, device=device
        )
        for run_i in range(n_runs):
            # Create directories
            ckpt_root = Path(model_name + "/train/run_{}".format(run_i + 1))
            ckpt_root.mkdir(parents=True, exist_ok=True)
            ckpt_best_root = Path(model_name + "/best")
            ckpt_best_root.mkdir(parents=True, exist_ok=True)
            # Set data loaders
            train_ldr, test_ldr = dataset.loaders(test_pct=0.5, batch_size=batch_size)
            # Set model and trainer
            model = model_cls(**model_params)
            trainer = trainer_cls(
                model=model,
                batch_size=batch_size,
                validation_ldr=test_ldr,
                device=device,
                lr=lr,
                n_epochs=n_epochs,
                run_test_validation=True,
                ckpt_root=str(ckpt_root.absolute()),
                keep_ckpt=True,
            )
            # Train
            trainer.train(train_ldr)
            # Print best F1-Score
            idx = np.argmax(trainer.metric_values["f1-score"])
            precision = trainer.metric_values["precision"][idx]
            recall = trainer.metric_values["recall"][idx]
            f1 = trainer.metric_values["f1-score"][idx]
            best_epoch = idx * 5 + 1
            print(
                "Best epoch={}\tPrecision={:2.4f}\tRecall={:2.4f}\tBest F1-Score={:2.4f}".format(best_epoch, precision,
                                                                                                 recall, f1))
            # Print figure
            figname = str(ckpt_root.absolute()) + "/ids2017_thresh-auto.png"
            trainer.plot_metrics(figname=figname)
            # Load best model based on f1-score
            best_ckpt = str(ckpt_root.absolute()) + "/{}_epoch={}.pt".format(model_name, best_epoch)
            trainer, model = trainer_cls.load_from_file(best_ckpt, trainer, model, device)
            trainer.n_epochs = best_epoch
            best_epoch = 7 * 5 + 1
            # Evaluate best model on binary targets
            y_test_true, test_scores, test_labels = trainer.test(test_ldr)
            results = metrics.estimate_optimal_threshold(test_scores, y_test_true)
            y_pred = (test_scores >= results["Thresh_star"]).astype(int)
            store_results(
                results=results,
                params=dict(batch_size=batch_size, lr=lr, n_epochs=best_epoch, **model.get_params()),
                model_name=model_name,
                dataset="IDS2017",
                dataset_path=dataset_path,
                results_path="./"
            )
            # Save best model
            trainer.save_ckpt(str(ckpt_best_root.absolute()) + "/best_run_{}_{}.pt".format(run_i + 1, model_name))
            # Evaluate best model on every attack targets
            misclf_df = ids_misclf_per_label(y_pred, y_test_true, test_labels)
            misclf_df = misclf_df.sort_values("Misclassified ratio", ascending=False)
            misclf_df.to_csv(str(ckpt_root.absolute()) + "/{}_misclassifications.csv".format(model_name))
            for attack_label in attack_types:
                performance_hist[model_name][attack_label].append(misclf_df.loc[attack_label, "Misclassified ratio"])
    hist_fname = "./performance_hist.npy"
    np.save(hist_fname, performance_hist)
