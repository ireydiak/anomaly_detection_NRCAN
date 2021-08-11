from . import AbstractDataset


class USBIDSDataset(AbstractDataset):

    name = 'USBIDS'

    def npz_key(self):
        return "usbids"
