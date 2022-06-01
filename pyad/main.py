import argparse
import bootstrap as bootstrap
from bootstrap import datasets_map, available_models


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
        "--dataset",
        type=str,
        choices=datasets_map.keys(),
        required=True
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        help='Path to the dataset',
        required=True
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=[model.name for model in available_models],
        required=True
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
        help="The size of the training batch",
        required=True
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
    for model_cls in available_models:
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


if __name__ == "__main__":
    args = argument_parser()
    model_name = args.model.lower()
    model_params = {k.replace("%s_" % model_name, ""): v for k, v in vars(args).items() if k.lower().startswith(model_name)}

    bootstrap.train(
        model_name=args.model,
        model_params=model_params,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        normal_size=args.normal_size,
        corruption_ratio=args.rho,
        n_runs=args.n_runs,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        results_path=args.results_path,
        models_path=args.model_path,
        test_mode=args.test_mode,
        seed=args.seed,
        holdout=args.hold_out,
        contamination_r=args.rho,
        drop_lastbatch=args.drop_lastbatch,
        validation_ratio=args.val_ratio,
    )
