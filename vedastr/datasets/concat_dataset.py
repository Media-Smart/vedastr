from torch.utils.data import ConcatDataset

from .registry import DATASETS
from .builder import build_datasets


@DATASETS.register_module
class ConcatDatasets(ConcatDataset):

    def __init__(self, datasets, transform=None, character='abcdefghijklmnopqrstuvwxyz0123456789',
                 batch_max_length=25, data_filter_off=False):
        assert isinstance(datasets, list)
        _params = dict(
            transform=transform,
            batch_max_length=batch_max_length,
            data_filter_off=data_filter_off,
            character=character,
        )

        datasets = build_datasets(datasets, default_args=_params)
        super(ConcatDatasets, self).__init__(datasets=datasets)
