from . import AbstractDataset

#from .AbstractDataset import AbstractDataset


class ArrhythmiaDataset(AbstractDataset):

    name = 'Arrhythmia'

    def npz_key(self):
        return "arrhythmia"


class IDS2018Dataset(AbstractDataset):

    name = 'IDS2018'

    def npz_key(self):
        return "ids2018"


NPZ_FILENAME = 'kdd10_train.npz'
BASE_PATH = '../data'


class KDD10Dataset(AbstractDataset):
    """
    This class is used to load KDD Cup 10% dataset as a pytorch Dataset
    """
    name = 'KDD10'

    def npz_key(self):
        return "kdd"


class NSLKDDDataset(AbstractDataset):
    """
    This class is used to load NSL-KDD Cup dataset as a pytorch Dataset
    """
    name = 'NSLKDD'

    def npz_key(self):
        return "kdd"


class ThyroidDataset(AbstractDataset):

    name = 'Thyroid'

    def npz_key(self):
        return "thyroid"


class USBIDSDataset(AbstractDataset):

    name = 'USBIDS'

    def npz_key(self):
        return "usbids"
