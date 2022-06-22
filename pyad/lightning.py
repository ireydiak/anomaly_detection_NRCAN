from pyad.model.reconstruction import LitAutoEncoder
from pyad.datamanager.dataset import ThyroidDataset, ArrhythmiaDataset
from pyad.model.transformers import LitGOAD
import pytorch_lightning as pl


def main():
    # data_path = "../data/Thyroid/thyroid.mat"
    data_path = "../data/Arrhythmia/arrhythmia_normalized.npz"
    #dataset = ThyroidDataset(path=data_path)
    dataset = ArrhythmiaDataset(path=data_path)
    batch_size = 64
    train_ldr, test_ldr, _ = dataset.loaders(batch_size=batch_size)

    model_name = "goad"
    if model_name.lower() == "autoencoder":
        model = LitAutoEncoder(
            in_features=dataset.in_features,
            n_instances=dataset.n_instances,
            latent_dim=2,
            hidden_dims=[4],
            activation="relu"
        )
    else:
        model = LitGOAD(
            in_features=dataset.in_features,
            n_transforms=256,
            feature_dim=32,
            num_hidden_nodes=8,
            batch_size=batch_size,
            n_layers=0,
            eps=0,
            lamb=0.1,
            margin=1
        )
    # trainer
    trainer = pl.Trainer(precision=16, accelerator="gpu", devices=1, max_epochs=200)
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
    # scores, y_true, _ = trainer.predict(
    #     model=model,
    #     dataloaders=test_ldr,
    # )
    # evaluate

    #y_test_true, test_scores, _ = trainer.test(test_ldr)
    #results, _ = metrics.score_recall_precision_w_threshold(test_scores, y_test_true)
    a = 1


if __name__ == "__main__":
    main()
