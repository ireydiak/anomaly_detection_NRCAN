import numpy as np
import torch
import os
from collections import defaultdict
from datetime import datetime as dt

from torch.utils.data import DataLoader

from src.model.adversarial import ALAD
from src.model.base import BaseModel
from src.model.density import DSEBM
from src.model.DUAD import DUAD
from src.model.one_class import DeepSVDD
from src.model.transformers import NeuTraLAD
from src.model.reconstruction import AutoEncoder as AE, DAGMM, MemAutoEncoder as MemAE, SOMDAGMM
from src.model.shallow import RecForest, OCSVM
from src.trainer.adversarial import ALADTrainer
from src.trainer.base import BaseTrainer
from src.trainer.density import DSEBMTrainer
from src.trainer.one_class import DeepSVDDTrainer
from src.trainer.reconstruction import AutoEncoderTrainer as AETrainer, DAGMMTrainer, MemAETrainer, SOMDAGMMTrainer
from src.trainer.shallow import OCSVMTrainer, RecForestTrainer
from src.trainer.transformers import NeuTraLADTrainer
from src.trainer.DUADTrainer import DUADTrainer
from src.utils.utils import average_results
from src.datamanager.dataset import AbstractDataset
from src.datamanager.DataManager import DataManager
from src.datamanager.dataset import ArrhythmiaDataset, KDD10Dataset, NSLKDDDataset, IDS2018Dataset, USBIDSDataset, \
    ThyroidDataset

available_models = [
    "AE",
    "ALAD",
    "DAGMM",
    "SOM-DAGMM",
    "MemAE",
    "DUAD",
    "OC-SVM",
    "RecForest",
    "DSEBM",
    "DeepSVDD",
    "NeuTraLAD"
]
available_datasets = [
    "Arrhythmia",
    "KDD10",
    "NSLKDD",
    "IDS2018",
    "USBIDS",
    "Thyroid"
]


def store_results(results: dict, params: dict, model_name: str, dataset: str, path: str):
    output_dir = f'../results/{dataset}'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    fname = output_dir + '/' + f'{model_name}_results.txt'
    with open(fname, 'a') as f:
        hdr = "Experiments on {}\n".format(dt.now().strftime("%d/%m/%Y %H:%M:%S"))
        f.write(hdr)
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
        f.write(f'{dataset} ({path.split("/")[-1].split(".")[0]})\n')
        f.write(", ".join([f"{param_name}={param_val}" for param_name, param_val in params.items()]) + "\n")
        f.write("\n".join([f"{met_name}: {res}" for met_name, res in results.items()]) + "\n")
        f.write("-".join("" for _ in range(len(hdr))) + "\n")
    return fname


def store_model(model, model_name: str, dataset: str):
    output_dir = f'../models/{dataset}/{model_name}/{dt.now().strftime("%d_%m_%Y_%H_%M_%S")}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(f"{output_dir}/model")


model_trainer_map = {
    # Deep Models
    "ALAD": (ALAD, ALADTrainer),
    "AE": (AE, AETrainer),
    "DAGMM": (DAGMM, DAGMMTrainer),
    "DSEBM": (DSEBM, DSEBMTrainer),
    "DUAD": (DUAD, DUADTrainer),
    "MemAE": (MemAE, MemAETrainer),
    "DeepSVDD": (DeepSVDD, DeepSVDDTrainer),
    "SOM-DAGMM": (SOMDAGMM, SOMDAGMMTrainer),
    "NeuTraLAD": (NeuTraLAD, NeuTraLADTrainer),
    # Shallow Models
    "OC-SVM": (OCSVM, OCSVMTrainer),
    "RecForest": (RecForest, RecForestTrainer)
}


def resolve_model_trainer(
        model_name: str,
        dataset: AbstractDataset,
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str
):
    model_trainer_tuple = model_trainer_map.get(model_name, None)
    assert model_trainer_tuple, "Model %s not found" % model_name
    model, trainer = model_trainer_tuple
    model = model(
        dataset_name=dataset.name,
        in_features=dataset.in_features,
        n_instances=dataset.n_instances,
        device=device
    )
    trainer = trainer(
        model=model,
        lr=learning_rate,
        n_epochs=n_epochs,
        batch_size=batch_size,
        device=device
    )

    return model, trainer


#
# def resolve_trainer(trainer_str: str, optimizer_factory, **kwargs):
#     model, trainer = None, None
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     in_features = dataset.get_shape()[1]
#     latent_dim = kwargs.get("latent_dim", in_features // 2)
#     reg_covar = kwargs.get("reg_covar", 1e-12)
#     if trainer_str == 'DAGMM' or trainer_str == 'SOM-DAGMM' or trainer_str == 'AE':
#         if dataset.name == 'Arrhythmia' or (dataset.name == 'Thyroid' and trainer_str != 'DAGMM'):
#             enc_layers = [(D, 10, nn.Tanh()), (10, L, None)]
#             dec_layers = [(L, 10, nn.Tanh()), (10, D, None)]
#             gmm_layers = [(L + 2, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
#         elif dataset.name == 'Thyroid' and trainer_str == 'DAGMM':
#             enc_layers = [(D, 12, nn.Tanh()), (12, 4, nn.Tanh()), (4, L, None)]
#             dec_layers = [(L, 4, nn.Tanh()), (4, 12, nn.Tanh()), (12, D, None)]
#             gmm_layers = [(L + 2, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
#         else:
#             enc_layers = [(D, 60, nn.Tanh()), (60, 30, nn.Tanh()), (30, 10, nn.Tanh()), (10, L, None)]
#             dec_layers = [(L, 10, nn.Tanh()), (10, 30, nn.Tanh()), (30, 60, nn.Tanh()), (60, D, None)]
#             gmm_layers = [(L + 2, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]
#
#         if trainer_str == 'DAGMM':
#             model = DAGMM(D, ae_layers=(enc_layers, dec_layers), gmm_layers=gmm_layers, reg_covar=reg_covar)
#             trainer = DAGMMTrainTestManager(
#                 model=model, dm=dm, optimizer_factory=optimizer_factory,
#             )
#         elif trainer_str == 'AE':
#             model = AE(enc_layers, dec_layers)
#             trainer = AETrainer(
#                 model=model, dm=dm, optimizer_factory=optimizer_factory
#             )
#         else:
#             gmm_input = kwargs.get('n_som', 1) * 2 + kwargs.get('latent_dim', 1) + 2
#             if dataset.name == 'Arrhythmia':
#                 gmm_layers = [(gmm_input, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 2, nn.Softmax(dim=-1))]
#             else:
#                 gmm_layers = [(gmm_input, 10, nn.Tanh()), (None, None, nn.Dropout(0.5)), (10, 4, nn.Softmax(dim=-1))]
#
#             dagmm = DAGMM(
#                 dataset.get_shape()[1],
#                 ae_layers=(enc_layers, dec_layers),
#                 gmm_layers=gmm_layers, reg_covar=reg_covar
#             )
#             # set these values according to the used dataset
#             grid_length = int(np.sqrt(5 * np.sqrt(len(dataset)))) // 2
#             grid_length = 32 if grid_length > 32 else grid_length
#             som_args = {
#                 "x": grid_length,
#                 "y": grid_length,
#                 "lr": 0.6,
#                 "neighborhood_function": "bubble",
#                 'n_epoch': 8000,
#                 'n_som': kwargs.get('n_som')
#             }
#             model = SOMDAGMM(dataset.get_shape()[1], dagmm, som_args=som_args)
#             trainer = SOMDAGMMTrainer(
#                 model=model, dm=dm, optimizer_factory=optimizer_factory
#             )
#             som_train_data = dataset.split_train_test()[0]
#             data = [som_train_data[i][0] for i in range(len(som_train_data))]
#             trainer.train_som(data)
#
#     elif trainer_str == 'MemAE':
#         print(f"training on {device}")
#         model = MemAE(dataset.name, D, device).to(device)
#         alpha = kwargs.get("alpha", 2e-4)
#         trainer = MemAETrainer(
#             alpha=alpha,
#             model=model,
#             device=device,
#             lr=kwargs.get("learning_rate"),
#             batch_size=batch_size,
#             n_epochs=kwargs.get("num_epochs", 200)
#         )
#     elif trainer_str == "DUAD":
#         model = DUAD(D, 10)
#         trainer = DUADTrainer(model=model, dm=dm, optimizer_factory=optimizer_factory, device=device,
#                               p=kwargs.get('p_s'), p_0=kwargs.get('p_0'), r=kwargs.get('r'),
#                               num_cluster=kwargs.get('num_cluster'))
#     elif trainer_str == 'ALAD':
#         # bsize = kwargs.get('batch_size', None)
#         model = ALAD(D, L, device=device).to(device)
#         trainer = ALADTrainer(
#             model=model,
#             dm=dm,
#             device=device,
#             learning_rate=lr,
#             L=L
#         )
#     elif trainer_str == 'DeepSVDD':
#         model = DeepSVDD(D)
#         trainer = DeepSVDDTrainer(
#             model,
#             optimizer_factory=optimizer_factory,
#             dm=dm,
#             R=kwargs.get('R'),
#             c=kwargs.get('c'),
#             device=device
#         )
#
#     elif trainer_str == 'DSEBM':
#         # bsize = kwargs.get('batch_size', None)
#         lr = kwargs.get('learning_rate', None)
#
#         assert batch_size and lr
#         model = DSEBM(D, dataset=dataset.name).to(device)
#         trainer = DSEBMTrainer(
#             model=model,
#             dm=dm,
#             device=device,
#             batch=batch_size, dim=D, learning_rate=lr,
#         )
#     elif trainer_str == 'NeurTraAD':
#         # bsize = kwargs.get('batch_size', None)
#         lr = kwargs.get('learning_rate', None)
#
#         assert batch_size and lr
#
#         # Load a pretrained model in case it should be used
#
#         model = NeuTraLAD(D, device=device, temperature=0.07, dataset=dataset.name).to(device)
#         trainer = NeuTraADTrainer(
#             model=model,
#             dm=dm,
#             device=device,
#             optimizer_factory=optimizer_factory,
#             L=L, learning_rate=lr,
#         )
#
#     return model, trainer
#
#
# def train_shallow_model():
#     X_train, _ = get_X_from_loader(dm.get_train_set())
#     X_test, y_test = get_X_from_loader(dm.get_test_set())
#     print(f'Starting training: {args.model}')
#
#     all_results = defaultdict(list)
#     for r in range(n_runs):
#         print(f"Run number {r}/{n_runs}")
#
#         # Create the model with the appropriate parameters.
#         if args.model == 'RECFOREST':
#             model = RecForest(n_jobs=-1, random_state=-1)
#         elif args.model == 'OC-SVM':
#             print(f"Using nu = {args.nu}.")
#             model = OneClassSVM(kernel='rbf', gamma='scale', shrinking=False, verbose=True, nu=args.nu)
#         else:
#             print(f"'{args.model}' is not a supported sklearn model.")
#             exit(1)
#
#         model.fit(X_train)
#         print('Finished learning process')
#
#         anomaly_score_train = []
#         anomaly_score_test = []
#
#         # prediction for the training set
#         for i, X_i in enumerate(dm.get_train_set(), 0):
#             # OC-SVM predicts -1 for outliers (and 1 for inliers), however we want outliers to be 1.
#             # So we negate the predictions.
#             if args.model == 'OC-SVM':
#                 anomaly_score_train.append(-model.predict(X_i[0].numpy()))
#             else:
#                 anomaly_score_train.append(model.predict(X_i[0].numpy()))
#         anomaly_score_train = np.concatenate(anomaly_score_train)
#
#         # prediction for the test set
#         y_test = []
#         for i, X_i in enumerate(dm.get_test_set(), 0):
#             # OC-SVM predicts -1 for outliers (and 1 for inliers), however we want outliers to be 1.
#             # So we negate the predictions.
#             if args.model == 'OC-SVM':
#                 anomaly_score_test.append(-model.predict(X_i[0].numpy()))
#             else:
#                 anomaly_score_test.append(model.predict(X_i[0].numpy()))
#             y_test.append(X_i[1].numpy())
#
#         anomaly_score_test = np.concatenate(anomaly_score_test)
#         y_test = np.concatenate(y_test)
#
#         anomaly_score = np.concatenate([anomaly_score_train, anomaly_score_test])
#         # dump metrics with different thresholds
#         res = score_recall_precision(anomaly_score, anomaly_score_test, y_test)
#         # results = score_recall_precision_w_thresold(
#         #     anomaly_score, anomaly_score_test, y_test, pos_label=1, threshold=args.p_threshold
#         # )
#         for k, v in res.items():
#             all_results[k].append(v)
#
#     print('Averaging results')
#     final_results = average_results(all_results)
#     store_results(final_results, params, args.model, args.dataset, args.dataset_path)


def train_model(
        model: BaseModel,
        model_trainer: BaseTrainer,
        train_ldr: DataLoader,
        test_ldr: DataLoader,
        dataset_name: str,
        n_runs: int,
        thresh: float,
        device: str,
        model_path: str,
        test_mode: bool
):
    # Training and evaluation on different runs
    all_results = defaultdict(list)

    if test_mode:
        for model_file_name in os.listdir(model_path):
            model = BaseModel.load(f"{model_path}/{model_file_name}")
            model = model.to(device)
            model_trainer.model = model
            print("Evaluating the model on test set")
            # We test with the minority samples as the positive class
            y_train_true, train_scores = model_trainer.test(train_ldr)
            y_test_true, test_scores = model_trainer.test(test_ldr)
            y_true = np.concatenate((y_train_true, y_test_true), axis=0)
            scores = np.concatenate((train_scores, test_scores), axis=0)
            print("Evaluating model")
            results = model_trainer.estimate_optimal_threshold(scores, test_scores, y_test_true)
            for k, v in results.items():
                all_results[k].append(v)
    else:
        for i in range(n_runs):
            print(f"Run {i + 1} of {n_runs}")
            #_ = model_trainer.train(datamanager.get_train_set())
            _ = model_trainer.train(train_ldr)
            print("Finished learning process")
            print("Evaluating model on test set")
            # We test with the minority samples as the positive class
            y_train_true, train_scores = model_trainer.test(train_ldr)
            y_test_true, test_scores = model_trainer.test(test_ldr)
            y_true = np.concatenate((y_train_true, y_test_true), axis=0)
            scores = np.concatenate((train_scores, test_scores), axis=0)
            print("Evaluating model")
            results = model_trainer.estimate_optimal_threshold(scores, test_scores, y_test_true)
            # y_true, scores = model_trainer.test(datamanager.get_test_set())
            # results = model_trainer.evaluate(y_true, scores, thresh)
            print(results)
            for k, v in results.items():
                all_results[k].append(v)
            store_model(model, model.name, dataset_name)
            model.reset()

    # Compute mean and standard deviation of the performance metrics
    print("Averaging results ...")
    return average_results(all_results)


def train(
        model_name: str,
        dataset_name: str,
        dataset_path: str,
        batch_size: int,
        pct: float,
        corruption_ratio: float,
        n_runs: int,
        n_epochs: int,
        learning_rate: float,
        results_path: str,
        models_path: str,
        test_mode: bool
):
    # Dynamically load the Dataset instance
    clsname = globals()[f'{dataset_name}Dataset']
    dataset = clsname(path=dataset_path, pct=pct)
    anomaly_thresh = 1 - dataset.anomaly_ratio

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # split data in train and test sets
    # we train only on the majority class
    train_ldr, test_ldr = dataset.loaders(batch_size=batch_size, seed=42)
    #train_set, test_set = dataset.train_test_split(test_ratio=0.50, corruption_ratio=corruption_ratio)
    #dm = DataManager(train_set, test_set, batch_size=batch_size)

    # check path
    for p in [results_path, models_path]:
        if p:
            assert os.path.exists(p), "Path %s does not exist" % p

    model, model_trainer = resolve_model_trainer(
        model_name=model_name,
        dataset=dataset,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        device=device
    )
    res = train_model(
        model=model,
        model_trainer=model_trainer,
        train_ldr=train_ldr,
        test_ldr=test_ldr,
        dataset_name=dataset_name,
        n_runs=n_runs,
        device=device,
        thresh=anomaly_thresh,
        model_path=models_path,
        test_mode=test_mode
    )
    print(res)
    params = dict(
        {"BatchSize": batch_size, "Epochs": n_epochs, "CorruptionRatio": corruption_ratio,
         "Threshold": anomaly_thresh},
        **model.get_params()
    )
    # Store the average of results
    fname = store_results(res, params, model_name, dataset.name, dataset_path)
    print(f"Results stored in {fname}")
