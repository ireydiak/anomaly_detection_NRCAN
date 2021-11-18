from . import AbstractDataset


class ThyroidDataset(AbstractDataset):

    name = 'Thyroid'

    def npz_key(self):
        return "thyroid"
