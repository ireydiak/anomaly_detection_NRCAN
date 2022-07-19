import os
import argparse
import numpy as np
import pyad.lightning
import pytorch_lightning as pl
import pyad.datamanager.datamodule
from typing import Sequence
from pytorch_lightning.utilities.cli import LightningCLI, MODEL_REGISTRY, DATAMODULE_REGISTRY
from pytorch_lightning import loggers as pl_loggers
from pyad.utils.utils import store_results

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

MODEL_REGISTRY.register_classes(pyad.lightning, pl.LightningModule)
DATAMODULE_REGISTRY.register_classes(pyad.datamanager.datamodule, pl.LightningDataModule)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
        parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")
        parser.add_argument("--save_dir", type=str, default="../experiments/training")
        parser.add_argument("--n_runs", type=int, default=1, help="number of times the experiments are repeated")


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


def train(cli, model, exp_fname):
    datamodule = cli.datamodule
    model_name = model.__class__.__name__
    base_path = os.path.join(cli.config.save_dir, exp_fname)

    # logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=cli.config.save_dir,
        name=exp_fname
    )

    # create train and test set
    datamodule.setup()

    # trainer
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=cli.trainer.max_epochs,
        logger=tb_logger,
    )

    # pre-training if needed
    model.before_train(datamodule.train_dataloader())

    # train
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader()
    )
    # test
    res = trainer.test(
        model=model,
        dataloaders=datamodule.test_dataloader()
    )[0]
    # store per-class-accuracy results
    if model.per_class_accuracy is not None:
        model.per_class_accuracy.to_csv(
            os.path.join(base_path, "{}_per_class_accuracy.csv".format(model_name.lower()))
        )
    return res


def init_model(cli):
    # TODO: replace this awkward function: the cli automatically instantiates a model
    # we need to create a fresh instance at every run
    model_args = vars(cli.config.model.init_args)
    model_args["in_features"] = cli.datamodule.in_features
    model_args["n_instances"] = cli.datamodule.n_instances
    model_args["batch_size"] = cli.datamodule.batch_size
    model_cls = MODEL_REGISTRY[cli.model.__class__.__name__]
    return model_cls(**model_args)


def main(cli):
    all_results = None
    dataset_name = cli.datamodule.__class__.__name__.lower().replace("datamodule", "")
    model_name = cli.model.__class__.__name__
    exp_fname = os.path.join(dataset_name, model_name)

    for run in range(1, cli.config.n_runs + 1):
        # instead of resetting the weights, we simply create a fresh instance of the model at every run
        model_instance = init_model(cli)
        # train the model for `trainer.max_epochs` epochs
        res = train(cli, model_instance, exp_fname)
        # keep the results in a dictionary
        if all_results is None:
            all_results = {k: [v] for k, v in res.items()}
        else:
            for k, v in res.items():
                all_results[k].append(v)
    # aggregate results (mean and std)
    all_results = {k: "{:2.4f} ({:2.4f})".format(np.mean(v), np.std(v)) for k, v in all_results.items()}
    print(all_results)
    # store the results in a simple text file
    store_results(
        all_results,
        os.path.join(cli.config.save_dir, exp_fname, f"{model_name}_results.txt")
    )


if __name__ == "__main__":
    main(
        MyLightningCLI(run=False)
    )
