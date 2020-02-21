from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch._utils import _accumulate

from .registry import DATALOADERS
from .base import BaseDataloader


@DATALOADERS.register_module
class RawDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        datasets = ConcatDataset(dataset)
        super(RawDataloader, self).__init__(datasets, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers,
                                            pin_memory=True)
