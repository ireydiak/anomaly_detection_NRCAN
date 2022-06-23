from pyad.model.reconstruction import LitAutoEncoder
from pyad.datamanager.dataset import ThyroidDataset, ArrhythmiaDataset
import pyad.model.transformers
from pyad.model.transformers import LitGOAD, LitNeuTraLAD
import pyad.datamanager.data_module
from pyad.datamanager.data_module import BaseDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY, LightningCLI
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY
from jsonargparse import ArgumentParser
import argparse

MODEL_REGISTRY.register_classes(pyad.model.transformers, pl.LightningModule)
DATAMODULE_REGISTRY.register_classes(pyad.datamanager.data_module, pl.LightningDataModule)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_class_arguments(BaseDataset, "dataset")
        parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
        parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")


class VerboseStore(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError('nargs not allowed')
        super(VerboseStore, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print('Here I am, setting the ' \
              'values %r for the %r option...' % (values, option_string))
        model = MODEL_REGISTRY[values]
        model.add_model_specific_args(parser)
        # for model in MODEL_REGISTRY.classes:
        #     model.add_model_specific_args(parser)
        #     parser.add_class_arguments(model, "model")
        # setattr(namespace, self.dest, values)


def argparser():
    parser = ArgumentParser()
   # parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", type=str)
    # model-specific arguments
    # for model in MODEL_REGISTRY.classes:
    #     model.add_model_specific_args(parser)
    # parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
    # parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")
    return parser.parse_args()


def prepare_thyroid(model_name: str):
    model_name = model_name.lower()
    data_path = "../data/Thyroid/thyroid.mat"
    dataset = ThyroidDataset(path=data_path)
    batch_size = 128
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)

    if model_name == "litautoencoder":
        model = LitAutoEncoder(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            latent_dim=2,
            hidden_dims=[4],
            activation="relu"
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
    else:
        model = LitGOAD(
            in_features=dataset.in_features,
            n_transforms=256,
            feature_dim=32,
            num_hidden_nodes=8,
            batch_size=batch_size,
            n_layers=0,
            eps=0,
            lamb=0.1,
            margin=1
        )
    return model, train_ldr, test_ldr


def prepare_arrhythmia(model_name: str):
    model_name = model_name.lower()
    data_path = "../data/Arrhythmia/arrhythmia_normalized.npz"
    dataset = ArrhythmiaDataset(path=data_path)
    batch_size = 64
    if model_name.lower() == "litautoencoder":
        model = LitAutoEncoder(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            latent_dim=2,
            hidden_dims=[4],
            activation="relu"
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
    else:
        model = LitGOAD(
            in_features=dataset.in_features,
            n_transforms=256,
            feature_dim=32,
            num_hidden_nodes=8,
            batch_size=batch_size,
            n_layers=0,
            eps=0,
            lamb=0.1,
            margin=1
        )
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)

    return model, train_ldr, test_ldr


def main(args):
    dataset_name = args.dataset.lower()
    max_epochs = 50
    # trainer
    trainer = pl.Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=max_epochs)

    if dataset_name == "arrhythmia":
        model, train_ldr, test_ldr = prepare_arrhythmia(args.model)
    elif dataset_name == "thyroid":
        model, train_ldr, test_ldr = prepare_thyroid(args.model)
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
