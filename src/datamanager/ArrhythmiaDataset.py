from . import AbstractDataset


class ArrhythmiaDataset(AbstractDataset):

    name = 'Arrhythmia'

    def npz_key(self):
        return "arrhythmia"
