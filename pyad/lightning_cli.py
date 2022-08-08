import os
import numpy as np
import torch

import pyad.lightning
import pytorch_lightning as pl
import pyad.datamanager.datamodule
from pytorch_lightning.utilities.cli import LightningCLI, MODEL_REGISTRY, DATAMODULE_REGISTRY
from pytorch_lightning import loggers as pl_loggers

from pyad.legacy import bootstrap
from pyad.utils.utils import store_results

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")

MODEL_REGISTRY.register_classes(pyad.lightning, pl.LightningModule)
DATAMODULE_REGISTRY.register_classes(pyad.datamanager.datamodule, pl.LightningDataModule)


def get_default_experiment_path():
    return os.path.join(
        os.path.abspath(__file__), "../experiments/training"
    )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.in_features", "model.init_args.in_features", apply_on="instantiate")
        parser.link_arguments("data.n_instances", "model.init_args.n_instances", apply_on="instantiate")
        parser.add_argument("--save_dir", type=str, default=get_default_experiment_path())
        parser.add_argument("--n_runs", type=int, default=1, help="number of times the experiments are repeated")


def train_legacy(cli, model, exp_fname):
    # load legacy data classes
    dataset_name = cli.datamodule.__class__.__name__.replace("DataModule", "")
    dataset_cls = bootstrap.datasets_map[dataset_name]
    dataset = dataset_cls(
        path=cli.config.data.init_args.data_dir,
        normal_size=cli.config.data.init_args.normal_size
    )
    anomaly_thresh = 1 - dataset.anomaly_ratio
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # train legacy model for a single run
    res = bootstrap.train_model(
        model_name=model.__class__.__name__,
        model_params=dict(cli.config.model.init_args),
        batch_size=cli.config.data.init_args.batch_size,
        n_epochs=cli.config.trainer.max_epochs,
        learning_rate=cli.config.model.init_args.lr,
        weight_decay=cli.config.model.init_args.weight_decay,
        test_mode=False,
        seed=42,
        device=device,
        model_path=exp_fname,
        dataset=dataset
    )
    print(res)
    return res


def sk_train(cli, model):
    datamodule = cli.datamodule
    dataset = datamodule.dataset

    # train test split
    train_idx, test_idx = dataset.train_test_split()
    train_X, train_y, train_labels = dataset[train_idx]
    test_X, test_y, test_labels = dataset[test_idx]

    # sanity checks
    # normal or abnormal data only in training and mixed labels during testing
    assert len(np.unique(train_y)) == 1, "training set contains anomalies, aborting"
    assert len(np.unique(test_y)) == 2, "test set should contain only 0s and 1s, aborting"
    assert len(np.unique(test_labels)) >= 2, "test set contains less than two distinct labels, aborting"

    # normalize
    train_X, test_X = dataset.normalize(train_X, test_X)

    # fit on training set
    model.fit(train_X)

    # test on testing set
    res = model.test(test_X, test_y, test_labels)

    return res


def nn_train(cli, model, exp_fname):
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
    model.before_train(
        datamodule.train_dataloader()
    )

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
    model_args["threshold"] = int(np.ceil((1 - cli.datamodule.anomaly_ratio) * 100))
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
        if model_instance.is_legacy:
            res = train_legacy(cli, model_instance, exp_fname)
        elif model_instance.is_nn:
            # train neural network for `trainer.max_epochs` epochs
            res = nn_train(cli, model_instance, exp_fname)
        else:
            # train sklearn (shallow) model
            res = sk_train(cli, model_instance)
        # keep the results in a dictionary
        if all_results is None:
            all_results = {k: [v] for k, v in res.items()}
        else:
            for k, v in res.items():
                if k in all_results.keys():
                    all_results[k].append(v)
                else:
                    all_results[k] = [v]
    # aggregate results (mean and std)
    all_results = {k: "{:2.4f} ({:2.4f})".format(np.mean(v), np.std(v)) for k, v in all_results.items()}
    print(all_results)
    # store the results in a simple text file
    save_path = os.path.join(cli.config.save_dir, exp_fname)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path, exist_ok=False)
    store_results(
        all_results,
        os.path.join(save_path, f"{model_name}_results.txt")
    )


if __name__ == "__main__":
    main(
        MyLightningCLI(run=False)
    )
