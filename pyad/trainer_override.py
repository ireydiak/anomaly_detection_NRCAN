import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from typing import Optional, Union
from pytorch_lightning import loggers as pl_loggers
import os


class Trainer(pl.Trainer):

    def __init__(self, **kwargs):
        # ../experiments/training
        # tb_logger = pl_loggers.TensorBoardLogger(
        #     save_dir=logger_save_dir, #.save_dir,
        #     name=logger_name #os.path.join(dataset_name, model_cls.__name__)
        # )
        super(Trainer, self).__init__(**kwargs)

    def fit_test(
            self,
            model: "pl.LightningModule",
            train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
            test_dataloaders: Optional[EVAL_DATALOADERS] = None,
            ckpt_path: Optional[str] = None,
            datamodule: Optional[str] = None
    ) -> None:
        r"""
        Runs the full optimization routine and tests.

        Perform one evaluation epoch over the test set.
        It's separated from fit to make sure you never run on your test set until you want to.

        Args:
           model: The model to test.

           dataloaders: A :class:`torch.utils.data.DataLoader` or a sequence of them,
               or a :class:`~pytorch_lightning.core.datamodule.LightningDataModule` specifying test samples.

           ckpt_path: Either ``best`` or path to the checkpoint you wish to test.
               If ``None`` and the model instance was passed, use the current weights.
               Otherwise, the best model checkpoint from the previous ``trainer.fit`` call will be loaded
               if a checkpoint callback is configured.

           verbose: If True, prints the test results.

           datamodule: An instance of :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.

        Returns:
           List of dictionaries with metrics logged during the test phase, e.g., in model- or callback hooks
           like :meth:`~pytorch_lightning.core.lightning.LightningModule.test_step`,
           :meth:`~pytorch_lightning.core.lightning.LightningModule.test_epoch_end`, etc.
           The length of the list corresponds to the number of test dataloaders used.
        """
        self._call_and_handle_interrupt(
            self._fit_test_impl, model, train_dataloaders, test_dataloaders, datamodule, ckpt_path
        )

    def _fit_test_impl(
            self,
            model: "pl.LightningModule",
            train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
            test_dataloaders: Optional[EVAL_DATALOADERS] = None,
            ckpt_path: Optional[str] = None,
            datamodule: Optional[str] = None
    ):
        # # if a datamodule comes in as the second arg, then fix it for the user
        # if isinstance(train_dataloaders, LightningDataModule):
        #     datamodule = train_dataloaders
        #     train_dataloaders = None
        # # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        # if (train_dataloaders is not None or test_dataloaders is not None) and datamodule is not None:
        #     raise MisconfigurationException(
        #         "You cannot pass `train_dataloader` or `val_dataloaders` to `trainer.fit(datamodule=...)`"
        #     )
        self.fit(
            model=model,
            train_dataloaders=train_dataloaders,
            ckpt_path=ckpt_path,
            datamodule=datamodule
        )
        self.test(
            model=model,
            dataloaders=test_dataloaders,
            ckpt_path=ckpt_path,
            datamodule=datamodule
        )
