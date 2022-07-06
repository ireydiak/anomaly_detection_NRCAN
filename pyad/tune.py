import os
from typing import Sequence, Dict, Set

import math
import pyad.datamanager.dataset
import pyad.lightning
import pytorch_lightning as pl
import argparse

from pyad.datamanager.dataset import AbstractDataset
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cli import MODEL_REGISTRY, DATAMODULE_REGISTRY
from jsonargparse import ArgumentParser
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback

MODEL_REGISTRY.register_classes(pyad.lightning, pl.LightningModule)
DATAMODULE_REGISTRY.register_classes(pyad.datamanager.dataset, AbstractDataset)
# DATAMODULE_REGISTRY.register_classes(pyad.datamanager.datamodule, pl.LightningDataModule)


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
    # tuning arguments
    tuning_parser = ArgumentParser()
    tuning_parser.add_argument("--num_samples", type=int, default=500,
                               help="number of different tuning combinations generated")
    tuning_parser.add_argument("--num_gpus", type=int, default=-1, help="set to `-1` to use all available gpus")
    tuning_parser.add_argument("--model", type=str, help="name of the model to tune")
    tuning_parser.add_argument("--scaler", choices=["standard", "minmax"], default="minmax",
                               help="data scaling strategy")
    tuning_parser.add_argument("--data_dir", type=str, help="path to the dataset file")
    tuning_parser.add_argument("--save_dir", type=str, default="../experiments/tuning",
                               help="path where ray stores tuning files")
    # add action arguments
    tuning_parser.add_argument("--dataset", action=FindRegistryAction, registry=DATAMODULE_REGISTRY,
                               append_str="Dataset")
    tuning_parser.add_argument("--model", action=FindRegistryAction, registry=MODEL_REGISTRY,
                               help="name of the model to tune")

    return tuning_parser.parse_args()


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
    # Instantiate dataset class
    dataset = dataset_cls(
        path=args.dataset_path, scaler=args.scaler
    )

    # read configuration from model class
    config = model_cls.get_ray_config(
        in_features=dataset.in_features,
        n_instances=dataset.n_instances
    )

    # setup ASHA scheduler
    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=1,
        reduction_factor=2
    )

    # setup CLI reporter
    reporter = CLIReporter(
        parameter_columns=list(config.keys()),
        metric_columns=["loss", "aupr", "f1-score"]
    )

    # setup tuning logic
    train_fn_with_parameters = tune.with_parameters(
        train_tune,
        num_epochs=args.max_epochs,
        num_gpus=args.num_gpus,
        dataset=dataset,
        model_cls=model_cls
    )
    resources_per_trial = {"cpu": 4, "gpu": args.num_gpus}

    # start tuning
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


def main(args):
    tune_asha(args.tune, args.model, args.dataset)


if __name__ == "__main__":
    main(
        argparser()
    )
