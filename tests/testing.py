import argparse
import os
from collections import defaultdict
from src.datamanager.DataManager import DataManager
from src.datamanager.dataset import ArrhythmiaDataset, ThyroidDataset, IDS2017Dataset
from src.model.DUAD import DUAD
from src.trainer.DUADTrainer import DUADTrainer
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
        "--n-epochs",
        help="Number of training epochs",
        type=int,
        default=201
    )
    parser.add_argument(
        '--use-ckpt',
        help="Save checkpoints during training",
        action="store_true"

    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=""
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
    },
    "DUAD": {
        "model_cls": DUAD,
        "trainer_cls": DUADTrainer,
        "p0": 35.,
        "p": 30.,
        "r": 10,
        "p_s": 35.,
        "n_clusters": 20,
        "act_fn": "tanh",
        "latent_dim": 10,
        "compression_factor": 2,
        "n_layers": 4
    }
}


def resolve_dataset(name, path):
    name = name.lower()
    if name == "arrhythmia":
        return ArrhythmiaDataset(path=path)
    elif name == "ids2017":
        return IDS2017Dataset(path=path)
    elif name == "thyroid":
        return ThyroidDataset(path=path)
    else:
        raise Exception("unsupported dataset %s, aborting" % name)


def resolve_models(model_list: list) -> dict:
    to_train = defaultdict()
    for model_name in model_list:
        setting = settings.get(model_name)
        if setting:
            to_train[model_name] = setting
        else:
            raise KeyError("unknown model {}".format(model_name))
    return to_train


def main():
    args = argument_parser()
    batch_size = args.batch_size
    lr = 1e-4
    n_epochs = 200
    use_ckpt = args.use_ckpt
    selected_models = args.models or list(settings.keys())
    to_train = resolve_models(selected_models)
    dataset = resolve_dataset(args.dataset, args.dataset_path)
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size, seed=42)
    print("data loaded with shape {}".format(dataset.shape))
    for model_name, params in to_train.items():
        ckpt_path = Path(os.path.join(model_name, "checkpoints"))
        ckpt_path.mkdir(parents=True, exist_ok=True)
        print("Training model %s on %s" % (model_name, dataset.name))
        # Initialize model and trainer
        model_params = dict(
            **{"in_features": dataset.in_features, "n_instances": dataset.n_instances},
            **params
        )
        datamanager = None
        model = params["model_cls"](**model_params)
        if model_name.lower() == "duad":
            # DataManager for DUAD only
            train_set, test_set, _ = dataset.split_train_test(test_pct=0.50)
            datamanager = DataManager(train_set, test_set, batch_size=batch_size)
        trainer = params["trainer_cls"](
            model=model,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            device="cuda",
            validation_ldr=test_ldr,
            ckpt_root=str(ckpt_path.absolute()) if use_ckpt else None,
            dm=datamanager
        )
        # Train model and save checkpoint at the end
        trainer.train(train_ldr)
        ckpt_fname = model_name.lower() + ".pt"
        trainer.save_ckpt(ckpt_fname)
        del trainer
        del model
        # Load the saved checkpoint
        trainer, model = params["trainer_cls"].load_from_file(ckpt_fname)
        # Test loaded model
        y_true, scores, _ = trainer.test(test_ldr)
        res, _ = metrics.score_recall_precision_w_threshold(scores, y_true)
        print(res)
        # Display and save learning curves
        trainer.plot_metrics(figname=model_name + "_learning_curves.png")


if __name__ == "__main__":
    main()
