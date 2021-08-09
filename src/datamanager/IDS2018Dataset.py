from .AbstractDataset import AbstractDataset


class IDS2018Dataset(AbstractDataset):

    name = 'IDS2018'

    def npz_key(self):
        return "ids2018"
