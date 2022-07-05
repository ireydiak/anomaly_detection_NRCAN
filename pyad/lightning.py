import os
from typing import Sequence, Dict, Set

import math

from pytorch_lightning.loggers import TensorBoardLogger

from pyad.lightning.density import LitDSEBM, LitDAGMM
from pyad.lightning.one_class import LitDeepSVDD
from pyad.lightning.reconstruction import LitAutoEncoder, LitMemAE
from pyad.datamanager.dataset import ThyroidDataset, ArrhythmiaDataset, KDD10Dataset, NSLKDDDataset, AbstractDataset
import pyad.lightning
from pyad.lightning.transformers import LitGOAD, LitNeuTraLAD
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI, MODEL_REGISTRY, DATAMODULE_REGISTRY, _get_short_description
from jsonargparse import ArgumentParser, class_from_function
import argparse

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from pytorch_lightning import loggers as pl_loggers

MODEL_REGISTRY.register_classes(pyad.lightning, pl.LightningModule)
DATAMODULE_REGISTRY.register_classes(pyad.datamanager.dataset, AbstractDataset)


# DATAMODULE_REGISTRY.register_classes(pyad.datamanager.datamodule, pl.LightningDataModule)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # parser.add_class_arguments(BaseDataset, "dataset")
        parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
        parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")
        # subcommands
        # trainer_class = (
        #     self.trainer_class if isinstance(self.trainer_class, type) else class_from_function(self.trainer_class)
        # )
        # parser_subcommands = parser.add_subcommands()
        # # register all subcommands in separate subcommand parsers under the main parser
        # for subcommand in self.subcommands():
        #     subcommand_parser = ArgumentParser()
        #     #self._prepare_subcommand_parser(trainer_class, subcommand)# **kwargs.get(subcommand, {}))
        #     fn = getattr(trainer_class, subcommand)
        #     # extract the first line description in the docstring for the subcommand help message
        #     description = _get_short_description(fn)
        #     parser_subcommands.add_subcommand(subcommand, subcommand_parser, help=description)

    @staticmethod
    def subcommands() -> Dict[str, Set[str]]:
        """Defines the list of available subcommands and the arguments to skip."""
        parent_subcommands = LightningCLI.subcommands()
        # commands = dict(
        #     **parent_subcommands,
        #     **{"fit_test": {"train_dataloaders", "model", "test_dataloaders"}}
        # )
        commands = {"fit": parent_subcommands["fit"], "tune": parent_subcommands["tune"]}
        return commands


class FindRegistryAction(argparse.Action):
    def __init__(self, option_strings: Sequence[str], dest: str, registry, append_str=""):
        super().__init__(option_strings, dest)
        self.registry = registry
        self.append_str = append_str

    def __call__(self, parser, args, values, option_string=None):
        values = values.replace("-", "")
        if self.append_str:
            values = values + self.append_str
        model_cls = self.registry[values]
        setattr(args, self.dest, model_cls)


def argparser():
    # general parser with generic arguments
    parser = ArgumentParser(prog="")

    # training arguments
    train_parser = ArgumentParser()
    train_parser.add_argument("--max_epochs", type=int, default=50)
    train_parser.add_argument("--save_dir", type=str, default="../experiments/training",
                              help="path where logger will store files")
    train_parser.add_argument("--scaler", choices=["standard", "minmax"], default="minmax")

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
    tuning_parser.add_argument("--max_epochs", type=int, default=50)
    tuning_parser.add_argument("--scaler", choices=["standard", "minmax"], default="minmax")
    tuning_parser.add_argument("--dataset_path", type=str)
    tuning_parser.add_argument("--save_dir", type=str, default="../experiments/tuning",
                               help="path where ray stores tuning files")

    # add subcommands
    subcommands = parser.add_subcommands()
    subcommands.add_subcommand("train", train_parser)
    subcommands.add_subcommand("tune", tuning_parser)

    parser.add_argument("--dataset", action=FindRegistryAction, registry=DATAMODULE_REGISTRY, append_str="Dataset")
    parser.add_argument("--model", action=FindRegistryAction, registry=MODEL_REGISTRY)

    return parser.parse_args()


def prepare_kdd(model_cls):
    model_name = model_cls.__name__.lower()
    data_path = "../data/KDD10/kdd10.npy"
    dataset = KDD10Dataset(path=data_path, scaler="minmax")
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


def prepare_thyroid(model_cls):
    model_name = model_cls.__name__.lower()
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
    elif model_name == "litdsebm":
        batch_size = 128
        model = LitDSEBM(
            in_features=dataset.in_features,
            fc_1_out=128,
            fc_2_out=256,
            lr=1e-4,
            weight_decay=1e-4,
            batch_size=batch_size,
            score_metric="reconstruction"
        )
    elif model_name == "litdagmm":
        batch_size = 1024
        model = LitDAGMM(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            n_mixtures=2,
            gmm_hidden_dims=[10],
            gmm_activation="tanh",
            latent_dim=1,
            ae_hidden_dims=[12, 4],
            ae_activation="tanh",
            dropout_rate=0.5,
            lamb_1=0.1,
            lamb_2=0.005,
            reg_covar=1e-10,
            weight_decay=1e-4,
            lr=1e-4
        )
    else:
        raise Exception("unknown model %s" % model_name)

    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)
    return model, train_ldr, test_ldr


def prepare_nslkdd(model_cls):
    model_name = model_cls.__name__.lower()
    data_path = "../data/NSL-KDD/nsl-kdd.npy"
    dataset = NSLKDDDataset(path=data_path, scaler="minmax")

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
    elif model_name == "litdsebm":
        batch_size = 128
        model = LitDSEBM(
            in_features=dataset.in_features,
            fc_1_out=128,
            fc_2_out=256,
            lr=1e-4,
            weight_decay=1e-4,
            batch_size=batch_size,
            score_metric="reconstruction"
        )
    elif model_name == "litdagmm":
        batch_size = 1024
        model = LitDAGMM(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            n_mixtures=2,
            gmm_hidden_dims=[10],
            gmm_activation="tanh",
            latent_dim=1,
            ae_hidden_dims=[60, 30, 10],
            ae_activation="tanh",
            dropout_rate=0.5,
            lamb_1=0.1,
            lamb_2=0.005,
            reg_covar=1e-10,
            weight_decay=1e-4,
            lr=1e-4
        )
    else:
        raise Exception("unknown model %s" % model_name)

    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)
    return model, train_ldr, test_ldr


def prepare_arrhythmia(model_cls):
    model_name = model_cls.__name__.lower()
    data_path = "../data/Arrhythmia/arrhythmia.npy"
    dataset = ArrhythmiaDataset(path=data_path, scaler="standard")
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
        batch_size = 64
        model = LitGOAD(
            in_features=dataset.in_features,
            lr=0.00166,
            weight_decay=0,
            n_transforms=64,
            feature_dim=128,
            num_hidden_nodes=8,
            batch_size=batch_size,
            n_layers=1,
            eps=0,
            lamb=0.8325,
            margin=1,
            threshold=(1 - dataset.anomaly_ratio) * 100
        )
    elif model_name == "litdagmm":
        batch_size = 128
        model = LitDAGMM(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            n_mixtures=2,
            gmm_hidden_dims=[10],
            gmm_activation="tanh",
            latent_dim=2,
            ae_hidden_dims=[10],
            ae_activation="tanh",
            dropout_rate=0.5,
            lamb_1=0.1,
            lamb_2=0.005,
            reg_covar=1e-10,
            weight_decay=1e-4,
            lr=1e-4
        )
    elif model_name == "litdeepsvdd" or model_name == "litdsvdd":
        batch_size = 128
        model = LitDeepSVDD(
            in_features=dataset.in_features,
            feature_dim=1024,
            hidden_dims=[256, 512],
            activation="relu",
            eps=0.1,
            lr=1e-4,
            weight_decay=1e-6
        )
    else:
        raise Exception("unknown model %s" % model_name)
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


def tune_asha(args, model_cls, dataset_cls):
    dataset = dataset_cls(path=args.dataset_path, scaler=args.scaler)

    config = model_cls.get_ray_config(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances
    )

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "aupr", "f1-score"]
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
                        name="%s_%s" % (dataset.name.lower(), model_cls.__name__.lower()),
                        local_dir=args.save_dir
                        )
    print("Best hyperparameters found were: ", analysis.best_config)


def train(args, model_cls, dataset_cls):
    dataset_name = dataset_cls.name.lower()

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.save_dir,
        name=os.path.join(dataset_name, model_cls.__name__)
    )

    # trainer
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        logger=tb_logger
    )

    if dataset_name == "arrhythmia":
        model, train_ldr, test_ldr = prepare_arrhythmia(model_cls)
    elif dataset_name == "thyroid":
        model, train_ldr, test_ldr = prepare_thyroid(model_cls)
    elif dataset_name == "kdd" or dataset_name == "kdd10":
        model, train_ldr, test_ldr = prepare_kdd(model_cls)
    elif dataset_name == "nsl-kdd" or dataset_name == "nslkdd":
        model, train_ldr, test_ldr = prepare_nslkdd(model_cls)
    else:
        model, train_ldr, test_ldr = prepare_thyroid(model_cls)

    model.before_train(train_ldr)

    # learn
    trainer.fit(
        model=model,
        train_dataloaders=train_ldr
    )
    # inference (predict)
    res = trainer.test(
        model=model,
        dataloaders=test_ldr
    )[0]
    # store_results()
    a = 1


# def store_results(results_dict: dict, hparams: dict)
available_subcommands = ["tune", "train"]


def main(args):
    if args.subcommand == "tune":
        tune_asha(args.tune, args.model, args.dataset)
    elif args.subcommand == "train":
        train(args.train, args.model, args.dataset)
    else:
        raise Exception("unknown subcommand %s, please choose between %s", (args.subcommand, available_subcommands))


def main_litcli(cli):
    dataset_name = cli.datamodule.__class__.__name__.lower().replace("datamodule", "")
    datamodule = cli.datamodule
    model = cli.model

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="../experiments/training",# todo: add to argparse args.save_dir,
        name=os.path.join(dataset_name, cli.model.__class__.__name__)
    )

    # trainer
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=cli.trainer.max_epochs,
        logger=tb_logger,
    )

    # learn
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader()
    )
    # inference (predict)
    res = trainer.test(
        model=model,
        dataloaders=datamodule.test_dataloader()
    )[0]
    # store_results()
    a = 1


from pyad.trainer_override import Trainer as PyADTrainer

# LightningArgumentParser
if __name__ == "__main__":
    # main_litcli(
    #     MyLightningCLI(run=False)
    # )#, trainer_class=PyADTrainer)
    main(
        argparser()
    )
