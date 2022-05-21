import argparse
import os
import numpy as np
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
from pathlib import Path


def argument_parser():
    parser = argparse.ArgumentParser(
        usage="\n python testing.py"
              "-d [dataset-path]"
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
        '-p',
        '--dataset-path',
        type=str,
        help='Path to the dataset',
        required=True
    )
    parser.add_argument(
        '--n-runs',
        help="Number times the experiment is repeated with different subsamples",
        type=int,
        default=1
    )
    parser.add_argument(
        "--n-epochs",
        help="Number of training epochs",
        type=int,
        default=200
    )
    parser.add_argument(
        '--use-ckpt',
        help="Save checkpoints during training",
        action="store_true"

    )
    return parser.parse_args()


n_layers = 4
compression_factor = 2

settings = {
    "ALAD": {
        "model_cls": ALAD,
        "trainer_cls": ALADTrainer,
        "latent_dim": -1
    },
    "DSEBM": {
        "model_cls": DSEBM,
        "trainer_cls": DSEBMTrainer,
        "fc_1_out": 128,
        "fc_2_out": 512
    },
    "DeepSVDD": {
        "model_cls": DeepSVDD,
        "trainer_cls": DeepSVDDTrainer,
        "n_layers": n_layers,
        "compression_factor": compression_factor,
        "act_fn": "relu"
    },
    "DROCC": {
        "model_cls": DROCC,
        "trainer_cls": EdgeMLDROCCTrainer
    },
    "AutoEncoder": {
        "model_cls": AutoEncoder,
        "trainer_cls": AutoEncoderTrainer,
        "latent_dim": 1,
        "act_fn": "relu",
        "n_layers": n_layers,
        "compression_factor": compression_factor,
        "reg": 0.5,
    },
    "DAGMM": {
        "model_cls": DAGMM,
        "trainer_cls": DAGMMTrainer,
        "n_mixtures": 4,
        "latent_dim": 1,
        "lambda_1": 0.1,
        "lambda_2": 0.005,
        "reg_covar": 1e-12,
        "n_layers": n_layers,
        "compression_factor": compression_factor,
        "ae_act_fn": "relu",
        "gmm_act_fn": "tanh",
    },
    "MemAE": {
        "model_cls": MemAutoEncoder,
        "trainer_cls": MemAETrainer,
        "mem_dim": 50,
        "latent_dim": 1,
        "shrink_thres": 0.0025,
        "n_layers": n_layers,
        "compression_factor": compression_factor,
        "alpha": 2e-5,
        "act_fn": "relu",
    },
    "NeuTraLAD": {
        "model_cls": NeuTraLAD,
        "trainer_cls": NeuTraLADTrainer,
        "fc_1_out": 90,
        "fc_last_out": 32,
        "compression_unit": 20,
        "temperature": 0.07,
        "trans_type": "mul",
        "n_layers": n_layers,
        "n_transforms": 11,
        "trans_fc_in": 200,
        "trans_fc_out": -1,
    }
}


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


def train():
    args = argument_parser()
    batch_size = args.batch_size
    lr = 1e-4
    n_epochs = args.n_epochs
    use_ckpt = args.use_ckpt
    dataset = resolve_dataset(args.dataset, args.dataset_path)
    train_ldr, test_ldr = dataset.loaders(batch_size=batch_size, seed=42)
    print("data loaded with shape {}".format(dataset.shape))
    for model_name, params in settings.items():
        model_root = os.path.join(args.dataset.lower(), model_name.lower())
        ckpt_path = Path(os.path.join(model_root, "checkpoints"))
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print("Training model %s on %s" % (model_name, dataset.name))
        # Initialize model and trainer
        model_params = dict(
            **{"in_features": dataset.in_features, "n_instances": dataset.n_instances},
            **params
        )
        model = params["model_cls"](**model_params)
        trainer = params["trainer_cls"](
            model=model,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            device="cuda",
            validation_ldr=test_ldr,
            ckpt_root=str(ckpt_path.absolute()) if use_ckpt else None
        )
        # Train model
        trainer.train(train_ldr)
        # Load best results
        best_idx = np.argmax(trainer.metric_values["f1-score"])
        best_epoch = best_idx * 5 + 1
        precision, recall, f1 = trainer.metric_values["precision"][best_idx], trainer.metric_values["recall"][best_idx], \
                                trainer.metric_values["f1-score"][best_idx],
        print("Precision={:2.4f}, Recall={:2.4f}, F1-Score={:2.4f} obtained at epoch {}".format(precision, recall, f1,
                                                                                                best_epoch))
        # Display and save learning curves
        trainer.plot_metrics(figname=os.path.join(model_root, model_name + "_learning_curves.png"))


if __name__ == "__main__":
    train()
