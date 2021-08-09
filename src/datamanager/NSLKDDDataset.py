from . import AbstractDataset


class NSLKDDDataset(AbstractDataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """
    name = 'NSLKDD'

    def npz_key(self):
        return "kdd"
