import os
import argparse
from ray import tune as ray_tune
from ray.tune.schedulers import ASHAScheduler
from pyad.bootstrap import resolve_dataset, datasets_map
from pyad.tuning.reconstruction import AutoEncoderTuner, MemAETuner
from datetime import datetime as dt


tuner_map = {
    "AutoEncoder": AutoEncoderTuner,
    "MemAE": MemAETuner
}


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=datasets_map.keys(),
        required=True
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--max-num-epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--models",
        nargs="+"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False
    )
    parser.add_argument(
        "--export-path",
        type=str,
        default="../results"
    )
    return parser.parse_args()


def resolve_config(dataset, model_cls) -> dict:
    return dict(
        lr=ray_tune.loguniform(1e-4, 1e-1),
        dataset=dataset,
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        **dataset.get_tunable_params(),
        **model_cls.get_tunable_params(n_instances=dataset.n_instances, in_features=dataset.in_features)
    )


def tune(
        dataset_name: str,
        dataset_path: str,
        models: list,
        max_num_epochs: int,
        num_samples: int,
        export_path: str,
        experiment_name: str = None,
):
    dataset = resolve_dataset(dataset_name, dataset_path)
    tuners = [tuner_map.get(m) for m in models]
    assert all([tuner is not None for tuner in tuners]), "invalid tuners provided, make sure given models exist"
    for tuner_cls in tuners:
        # Setup
        cfg = resolve_config(dataset, tuner_cls)
        experiment_name = experiment_name or os.path.join(
            tuner_cls.__name__, "{}".format(dt.now().strftime("%d_%m_%Y_%H-%M-%S"))
        )
        export_path = os.path.join(export_path, experiment_name)
        os.makedirs(export_path, exist_ok=True)
        scheduler = ASHAScheduler(
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2
        )
        # Run tuning
        result = ray_tune.run(
            tuner_cls,
            # ray_tune.with_parameters(train_cifar),
            resources_per_trial={"cpu": 2, "gpu": 1},
            config=cfg,
            metric="aupr",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
            name=experiment_name
        )
        # Display best result
        best_trial = result.get_best_trial("aupr", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final train loss: {}".format(best_trial.last_result["train_loss"]))
        print("Best trial final validation loss: {}".format(best_trial.last_result["val_loss"]))
        print("Best trial final validation aupr: {}".format(best_trial.last_result["aupr"]))
        result_df = result.dataframe().sort_values(by="aupr", ascending=False)
        # Save tuning summary in dataframe
        result_df.to_csv(
            os.path.join(export_path, "%s_tuning_results.csv" % tuner_cls.__name__)
        )


if __name__ == "__main__":
    args = parser_args()
    tune(
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        models=args.models,
        max_num_epochs=args.max_num_epochs,
        num_samples=args.num_samples,
        export_path=args.export_path
    )
