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
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"



def main():
    dataset_path = "../../data/IDS2017/ids2017.csv"
    dataset = IDS2017Dataset(path=dataset_path)

    dagmm_model = DAGMM(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        device=device,
        n_mixtures=4,
        latent_dim=1,
        lambda_1=0.005,
        lambda_2=0.1,
        reg_covar=1e-6,
        n_layers=4,
        compression_factor=2,
        ae_act_fn="relu",
        gmm_act_fn="relu"
    )
    dagmm_trainer = DAGMMIDSTrainer(
        device=device,
        model=dagmm_model,
        batch_size=1024,
        lr=0.001,
        n_epochs=200,
    )

    # for model_name in performance_hist.keys():
    model_name = "dagmm"
    rel_path = "./%s/best" % model_name
    for p in os.listdir(rel_path):
        fname = rel_path + "/" + p
        dagmm_trainer, dagmm_model = DAGMMIDSTrainer.load_from_file(fname, dagmm_trainer, dagmm_model, device)
        print(dagmm_trainer.metric_values)


if __name__ == "__main__":
    main()
