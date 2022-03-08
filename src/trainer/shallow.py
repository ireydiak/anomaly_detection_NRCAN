import torch

from src.trainer.base import BaseShallowTrainer


class RecForestTrainer(BaseShallowTrainer):

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }


class OCSVMTrainer(BaseShallowTrainer):

    def get_params(self) -> dict:
        return {
            **self.model.get_params()
        }
