import math

from pytorch_lightning.loggers import TensorBoardLogger

from pyad.lightning.reconstruction import LitAutoEncoder, LitMemAE
from pyad.datamanager.dataset import ThyroidDataset, ArrhythmiaDataset, KDD10Dataset, NSLKDDDataset, AbstractDataset
import pyad.lightning.transformers
from pyad.lightning.transformers import LitGOAD, LitNeuTraLAD
# import pyad.datamanager.data_module
# from pyad.datamanager.data_module import BaseDataset
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY, LightningCLI
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY
from jsonargparse import ArgumentParser
import argparse

from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from bootstrap import datasets_map
MODEL_REGISTRY.register_classes(pyad.lightning.transformers, pl.LightningModule)


# DATAMODULE_REGISTRY.register_classes(pyad.datamanager.data_module, pl.LightningDataModule)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_class_arguments(BaseDataset, "dataset")
        parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
        parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")


def argparser():
    # general parser with generic arguments
    parser = ArgumentParser(prog="")
    # parser.add_argument("--model", type=str)
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("--max_epochs", type=int, default=50)

    # training arguments
    train_parser = ArgumentParser()
    train_parser.add_argument("--model", type=str)
    train_parser.add_argument("--dataset", type=str)
    train_parser.add_argument("--max_epochs", type=int, default=50)
    # parser = pl.Trainer.add_argparse_args(parser)

    # model-specific arguments

    # for model in MODEL_REGISTRY.classes:
    #     model.add_model_specific_args(parser)
    # parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
    # parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")

    # tuning arguments
    tuning_parser = ArgumentParser()
    tuning_parser.add_argument("--num_samples", type=int, default=50)
    tuning_parser.add_argument("--num_gpus", type=int, default=None, help="set to `None` to use all available gpus")
    tuning_parser.add_argument("--model", type=str)
    tuning_parser.add_argument("--dataset", type=str)
    tuning_parser.add_argument("--max_epochs", type=int, default=50)
    tuning_parser.add_argument("--scaler", choices=["standard", "minmax"], default="minmax")
    tuning_parser.add_argument("--dataset_path", type=str)

    # add subcommands
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", train_parser)
    subcommands.add_subcommand("tune", tuning_parser)

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
        batch_size = 32
        # model = LitGOAD(
        #     in_features=dataset.in_features,
        #     lr=1e-3,
        #     weight_decay=1e-4,
        #     n_transforms=256,
        #     feature_dim=32,
        #     num_hidden_nodes=8,
        #     batch_size=batch_size,
        #     n_layers=1,
        #     eps=0,
        #     lamb=0.1,
        #     margin=1,
        #     threshold=95.18882565959649,
        # )
        model = LitGOAD(
            in_features=dataset.in_features,
            lr=4.8e-3,
            weight_decay=1e-4,
            n_transforms=128,
            feature_dim=64,
            num_hidden_nodes=32,
            batch_size=batch_size,
            n_layers=1,
            eps=0,
            lamb=0.6121,
            margin=1
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
    elif model_name == "litgoad":
        batch_size = 64
        model = LitGOAD(
            in_features=dataset.in_features,
            lr=1e-3,
            weight_decay=1e-4,
            n_transforms=256,
            feature_dim=128,
            num_hidden_nodes=128,
            batch_size=batch_size,
            n_layers=5,
            eps=0,
            lamb=0.1,
            margin=1
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


def train_tune(
        config,
        model_cls: pl.LightningModule,
        dataset,
        num_epochs: int = 10,
        num_gpus: int = 0
):
    # model
    model = model_cls(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        **config
    )
    # loaders
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=config["batch_size"])

    # trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "aupr": "AUPR",
                    "f1-score": "F1-Score"
                },
                on="test_end")
        ])
    # learn & evaluate
    trainer.fit(
        model=model,
        train_dataloaders=train_ldr,
    )
    # inference (predict)
    res = trainer.test(
        model=model,
        dataloaders=test_ldr
    )


def tune_asha(args):
    model_cls: pl.LightningModule = MODEL_REGISTRY[args.model]
    dataset_cls = datasets_map[args.dataset]
    dataset = dataset_cls(path=args.dataset_path, scaler=args.scaler)

    config = model_cls.get_ray_config(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances
    )
    # config = {
    #     "layer_1_size": tune.choice([32, 64, 128]),
    #     "layer_2_size": tune.choice([64, 128, 256]),
    #     "lr": tune.loguniform(1e-4, 1e-1),
    #     "batch_size": tune.choice([16, 32, 64, 128]),
    # }

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "aupr", "f1-score", "training_iteration"]
    )

    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        num_epochs=args.max_epochs,
        num_gpus=args.num_gpus,
        dataset=dataset,
        model_cls=model_cls
    )
    resources_per_trial = {"cpu": 1, "gpu": args.num_gpus}

    analysis = tune.run(train_fn_with_parameters,
                        resources_per_trial=resources_per_trial,
                        metric="aupr",
                        mode="max",
                        config=config,
                        num_samples=args.num_samples,
                        scheduler=scheduler,
                        progress_reporter=reporter,
                        name="tune_%s_asha" % dataset.name)

    print("Best hyperparameters found were: ", analysis.best_config)


def train(args):
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


available_subcommands = ["tune", "train"]


def main(args):
    if args.subcommand == "tune":
        tune_asha(args.tune)
    elif args.subcommand == "train":
        train(args.train)
    else:
        raise Exception("unknown subcommand %s, please choose between %s", (args.subcommand, available_subcommands))


# LightningArgumentParser
if __name__ == "__main__":
    # cli = MyLightningCLI()
    main(
        argparser()
    )
