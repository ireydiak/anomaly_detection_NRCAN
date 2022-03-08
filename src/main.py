import argparse
import src.bootstrap as bootstrap
from src.bootstrap import available_datasets, available_models


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
        '-m',
        '--model',
        type=str,
        choices=available_models,
        required=True
    )
    parser.add_argument(
        '-d',
        '--dataset-path',
        type=str,
        help='Path to the dataset',
        required=True
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="The size of the training batch",
        required = True
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=available_datasets,
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
        '--n-runs',
        help='number of runs of the experiment',
        type=int,
        default=1
    )
    parser.add_argument(
        "-o",
        "--results-path",
        type=str,
        default=None,
        help="Where the results will be stored"
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help="The learning rate"
    )
    parser.add_argument(
        "--pct",
        type=float,
        default=1.0,
        help="Percentage of original data to keep"
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
        default="./",
        help="Path where the model's weights are stored and loaded"
    )
    parser.add_argument(
        "--test-mode",
        type=bool,
        default=False,
        help="Loads and test models found within model_path"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()

    bootstrap.train(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        pct=args.pct,
        corruption_ratio=args.rho,
        n_runs=args.n_runs,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        results_path=args.results_path,
        models_path=args.model_path,
        test_mode=args.test_mode
    )
