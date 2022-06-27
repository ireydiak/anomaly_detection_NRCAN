from pyad.lightning.reconstruction import LitAutoEncoder, LitMemAE
from pyad.datamanager.dataset import ThyroidDataset, ArrhythmiaDataset, KDD10Dataset, NSLKDDDataset
import pyad.lightning.transformers
from pyad.lightning.transformers import LitGOAD, LitNeuTraLAD
# import pyad.datamanager.data_module
# from pyad.datamanager.data_module import BaseDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY, LightningCLI
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY
from jsonargparse import ArgumentParser
import argparse

MODEL_REGISTRY.register_classes(pyad.lightning.transformers, pl.LightningModule)


# DATAMODULE_REGISTRY.register_classes(pyad.datamanager.data_module, pl.LightningDataModule)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_class_arguments(BaseDataset, "dataset")
        parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
        parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")


def argparser():
    parser = ArgumentParser()
    # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_epochs", type=int, default=50)
    # model-specific arguments
    # for model in MODEL_REGISTRY.classes:
    #     model.add_model_specific_args(parser)
    # parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
    # parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")
    return parser.parse_args()


def prepare_kdd(model_name: str):
    model_name = model_name.lower()
    data_path = "C:/Users/verdi/Documents/Datasets/KDD/3_minified/KDD10percent_minified.npz"
    dataset = KDD10Dataset(path=data_path)
    weight_decay = 1e-4

    if model_name == "litautoencoder":
        batch_size = 1024
        n_epochs = 10
        model = LitAutoEncoder(
            in_features=dataset.in_features,
            latent_dim=1,
            hidden_dims=[64, 32, 16, 8],
            activation="relu",
            weight_decay=weight_decay,
            lr=1e-3
        )
    elif model_name == "litneutralad":
        batch_size = 1024
        n_epochs = 10
        model = LitNeuTraLAD(
            in_features=dataset.in_features,
            weight_decay=weight_decay,
            lr=1e-3,
            n_transforms=11,
            trans_type="mul",
            temperature=0.1,
            trans_hidden_dims=[200, 121],
            enc_hidden_dims=[64, 64, 64, 64, 32]
        )
    elif model_name == "litgoad":
        batch_size = 64
        n_epochs = 25
        model = LitGOAD(
            in_features=dataset.in_features,
            lr=1e-4,
            weight_decay=1e-4,
            n_layers=5,
            n_transforms=64,
            feature_dim=64,
            num_hidden_nodes=32,
            lamb=0.1,
            eps=0,
            margin=1,
            batch_size=batch_size
        )
    else:
        raise Exception("unknown model %s" % model_name)
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)
    return model, train_ldr, test_ldr


def prepare_thyroid(model_name: str):
    model_name = model_name.lower()
    data_path = "../data/Thyroid/thyroid.mat"
    dataset = ThyroidDataset(path=data_path, scaler="standard")
    batch_size = 128

    if model_name == "litautoencoder":
        model = LitAutoEncoder(
            in_features=dataset.in_features,
            latent_dim=2,
            hidden_dims=[4],
            activation="relu",
            lr=1e-4,
            weight_decay=1e-4
        )
    elif model_name == "litneutralad":
        model = LitNeuTraLAD(
            in_features=dataset.in_features,
            weight_decay=1e-4,
            lr=1e-3,
            n_transforms=11,
            trans_type="mul",
            temperature=0.1,
            trans_hidden_dims=[24],
            enc_hidden_dims=[24, 24, 24, 24, 24]
        )
    elif model_name == "litmemae":
        batch_size = 32
        model = LitMemAE(
            in_features=dataset.in_features,
            mem_dim=50,
            latent_dim=1,
            enc_hidden_dims=[12, 24],
            shrink_thresh=0.0025,
            alpha=2e-4,
            activation="relu",
            lr=1e-3,
            weight_decay=1e-4
        )
    elif model_name == "litgoad":
        batch_size = 64
        model = LitGOAD(
            in_features=dataset.in_features,
            lr=1e-3,
            weight_decay=1e-4,
            n_transforms=256,
            feature_dim=32,
            num_hidden_nodes=8,
            batch_size=batch_size,
            n_layers=1,
            eps=0,
            lamb=0.1,
            margin=1,
            threshold=95.18882565959649,
        )
    else:
        raise Exception("unknown model %s" % model_name)

    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)
    return model, train_ldr, test_ldr


def prepare_nslkdd(model_name: str):
    model_name = model_name.lower()
    data_path = "../data/NSL-KDD/3_minified/NSL-KDD_minified.npz"
    dataset = NSLKDDDataset(path=data_path)

    if model_name == "litmemae":
        batch_size = 128
        model = LitMemAE(
            in_features=dataset.in_features,
            mem_dim=100,
            latent_dim=1,
            enc_hidden_dims=[120, 30, 15],
            shrink_thresh=0.00025,
            alpha=2e-4,
            activation="relu",
            lr=1e-4,
            weight_decay=1e-4
        )
    else:
        raise Exception("unknown model %s" % model_name)

    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)
    return model, train_ldr, test_ldr


def prepare_arrhythmia(model_name: str):
    model_name = model_name.lower()
    # data_path = "../data/Arrhythmia/arrhythmia_normalized.npz"
    data_path = "../data/Arrhythmia/arrhythmia.mat"
    dataset = ArrhythmiaDataset(path=data_path)
    batch_size = 32
    n_epochs = 200

    if model_name.lower() == "litautoencoder":
        model = LitAutoEncoder(
            in_features=dataset.in_features,
            latent_dim=2,
            hidden_dims=[4],
            activation="relu",
            lr=1e-4,
            weight_decay=1e-4
        )
    elif model_name == "litneutralad":
        model = LitNeuTraLAD(
            in_features=dataset.in_features,
            weight_decay=1e-4,
            lr=1e-3,
            n_transforms=11,
            trans_type="mul",
            temperature=0.1,
            trans_hidden_dims=[200, 274],
            enc_hidden_dims=[64, 64, 64, 64, 32]
        )
    elif model_name == "litmemae":
        model = LitMemAE(
            in_features=dataset.in_features,
            mem_dim=50,
            latent_dim=60,
            enc_hidden_dims=[200, 121],
            shrink_thresh=0.0025,
            alpha=2e-4,
            activation="relu",
            lr=1e-3,
            weight_decay=1e-4
        )
    elif model_name == "litgoad":
        model = LitGOAD(
            in_features=dataset.in_features,
            lr=1e-4,
            weight_decay=1e-4,
            n_transforms=256,
            feature_dim=32,
            num_hidden_nodes=8,
            batch_size=batch_size,
            n_layers=0,
            eps=0,
            lamb=0.1,
            margin=1
        )
    else:
        raise Exception("unkown model %s" % model_name)
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)

    return model, train_ldr, test_ldr


def main(args):
    dataset_name = args.dataset.lower()
    max_epochs = args.max_epochs
    # trainer
    trainer = pl.Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=max_epochs)

    if dataset_name == "arrhythmia":
        model, train_ldr, test_ldr = prepare_arrhythmia(args.model)
    elif dataset_name == "thyroid":
        model, train_ldr, test_ldr = prepare_thyroid(args.model)
    elif dataset_name == "kdd" or dataset_name == "kdd10":
        model, train_ldr, test_ldr = prepare_kdd(args.model)
    elif dataset_name == "nsl-kdd" or dataset_name == "nslkdd":
        model, train_ldr, test_ldr = prepare_nslkdd(args.model)
    else:
        model, train_ldr, test_ldr = prepare_thyroid(args.model)

    # learn
    trainer.fit(
        model=model,
        train_dataloaders=train_ldr,
    )
    # inference (predict)
    res = trainer.test(
        model=model,
        dataloaders=test_ldr
    )


# LightningArgumentParser
if __name__ == "__main__":
    # cli = MyLightningCLI()
    main(
        argparser()
    )
