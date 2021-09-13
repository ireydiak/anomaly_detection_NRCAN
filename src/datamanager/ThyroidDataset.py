from . import AbstractDataset


class ThyroidDataset(AbstractDataset):

    name = 'ann_thyroid'

    def npz_key(self):
        return "thyroid"
