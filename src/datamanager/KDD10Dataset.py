from .AbstractDataset import AbstractDataset

NPZ_FILENAME = 'kdd10_train.npz'
BASE_PATH = '../data'


class KDD10Dataset(AbstractDataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """
    name = 'KDD10'

    def npz_key(self):
        return "kdd"
