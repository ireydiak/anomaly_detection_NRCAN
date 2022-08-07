import argparse
import os.path
import sys
from collections import defaultdict

import bootstrap as bsp
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with datasets and differents parameters
    """
    parser = argparse.ArgumentParser(
        usage="\n python main.py"
              "-m [model] -d [dataset-path]"
              " --dataset [dataset] -e [n_epochs]"
              " --n-runs [n_runs] --batch-size [batch_size]"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Load configuration from YAML. Use this option to avoid using CLI to setup models and experiments."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=bsp.datasets_map.keys()
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        help='Path to the dataset'
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[model.name for model in bsp.available_models]
    )
    parser.add_argument(
        "--n-runs",
        help="number of runs of the experiment",
        type=int,
        default=1
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The size of the training batch"
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=200,
        help="The number of epochs"
    )
    parser.add_argument(
        "-o",
        "--results-path",
        type=str,
        default=None,
        help="Where the results will be stored"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="The learning rate"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0,
        help='weight decay for regularization')
    parser.add_argument(
        "--normal-size",
        type=float,
        default=1.0,
        help="Percentage of original normal samples to keep"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Ratio of validation set from the training set"
    )
    parser.add_argument(
        "--hold-out",
        type=float,
        default=0.0,
        help="Percentage of anomalous data to holdout for possible contamination of the training set"
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Anomaly ratio within training set"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../models",
        help="Path where the model's weights are stored and loaded"
    )
    parser.add_argument(
        "--test-mode",
        type=bool,
        default=False,
        help="Loads and test models found within model_path"
    )
    parser.add_argument(
        "--seed",
        type=float,
        default=42,
        help='the randomness seed used'
    )

    # Models params
    for model_cls in bsp.available_models:
        for params in model_cls.get_args_desc():
            param_name, datatype, default_value, help_txt = params
            param_name = param_name.replace("_", "-")
            parser.add_argument(
                "--{}-{}".format(model_cls.name.lower(), param_name),
                type=datatype,
                default=default_value,
                help=help_txt
            )

    parser.add_argument('--drop-lastbatch', dest='drop-lastbatch', action='store_true')
    parser.add_argument('--no-drop-lastbatch', dest='drop-lastbatch', action='store_false')
    parser.set_defaults(drop_lastbatch=False)

    return parser.parse_args()


def parse_config(config_file):
    with open(config_file) as f:
        data = load(f, Loader=Loader)
        err = sanity_checks(data)
    return data, err


def sanity_checks(cfg):
    assert cfg["DATASETS"], "please specify a dataset"
    assert cfg["MODELS"], "please specify models"
    error_bag = defaultdict()
    # datasets
    error_bag["DATASETS"] = defaultdict()
    for dataset_name, dataset_params in cfg["DATASETS"].items():
        error_bag["DATASETS"][dataset_name] = []
        if not bsp.datasets_map.get(dataset_name):
            error_bag["DATASETS"][dataset_name].append("dataset %s does not exist" % dataset_name)
        available_params = {"batch_size", "normal_size", "holdout", "val_ratio", "path", "contamination_rate"}
        for param_name in dataset_params.keys():
            if param_name not in available_params:
                error_bag["DATASETS"][dataset_name].append(
                    "unknown param %s, use these parameters %s" % (param_name, available_params)
                )
        # make sure path to dataset exists
        path = dataset_params["path"]
        if path:
            if not os.path.exists(path):
                error_bag["DATASETS"][dataset_name].append(
                    "path to dataset %s does not exist" % path
                )
        else:
            error_bag["DATASETS"][dataset_name].append("missing `path` parameter")
    # models and trainers
    error_bag["MODELS"] = defaultdict()
    for model_name, model_params in cfg["MODELS"].items():
        error_bag["MODELS"][model_name] = []
        if not bsp.model_trainer_map.get(model_name):
            error_bag["MODELS"][model_name].append("model %s not available" % model_name)
        model_cls, trainer_cls = bsp.model_trainer_map.get(model_name)
        try:
            model = model_cls(
                in_features=1000,
                n_instances=1000,
                **model_params["INIT_PARAMS"]
            )
            _ = trainer_cls(
                model=model,
                batch_size=64
            )
        except Exception as e:
            error_bag["MODELS"][model_name].append(
                "could not initialize model %s with exception: %s" % (model_name, e)
            )
        # trainers
        trainer_params = {"n_epochs", "learning_rate", "weight_decay", "momentum"}
        for param_name in model_params["TRAINER"].keys():
            if param_name not in trainer_params:
                error_bag["MODELS"][model_name].append(
                    "unknown parameter %s, use these parameters %s" % (param_name, trainer_params)
                )
    return error_bag


def train_cli(args):
    model_name = args.model.lower()
    model_params = {k.replace("%s_" % model_name, ""): v for k, v in vars(args).items() if
                    k.lower().startswith(model_name)}

    bsp.train(
        model_name=args.model,
        model_params=model_params,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        normal_size=args.normal_size,
        n_runs=args.n_runs,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        results_path=args.results_path,
        models_path=args.model_path,
        test_mode=args.test_mode,
        seed=args.seed,
    )


def train_cfg(args):
    cfg, err = parse_config(args.config)
    cfg_ok = True
    for key, items in err.items():
        for item_name, item_val in items.items():
            for err_msg in item_val:
                cfg_ok = False
                print("Error in %s -> %s -> %s" % (key, item_name, err_msg))
    if cfg_ok:
        bsp.train_from_cfg(cfg)
        sys.exit(1)


def main(args):
    if args.config:
        train_cfg(args)
    else:
        train_cli(args)


if __name__ == "__main__":
    main(
        argument_parser()
    )
