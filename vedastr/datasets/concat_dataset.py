from torch.utils.data import ConcatDataset as _ConcatDataset

from .builder import build_datasets
from .registry import DATASETS


@DATASETS.register_module
class ConcatDatasets(_ConcatDataset):

    def __init__(self, datasets: list, batch_ratio: list = None, **kwargs):
        assert isinstance(datasets, list)
        datasets = build_datasets(datasets, default_args=kwargs)

        data_range = [len(dataset) for dataset in datasets]
        self.data_range = [sum(data_range[:i]) for i in range(1, len(data_range) + 1)]
        self.batch_ratio = batch_ratio

        super(ConcatDatasets, self).__init__(datasets=datasets)
