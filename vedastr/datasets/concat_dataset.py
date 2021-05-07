from torch.utils.data import ConcatDataset as _ConcatDataset

from .builder import build_datasets
from .registry import DATASETS


@DATASETS.register_module
class ConcatDatasets(_ConcatDataset):
    """ Concat different datasets.

        Args:
        datasets (list[dict]): A list of which each elements is a dataset cfg.
        batch_ratio (list[float]): Ratio of corresponding dataset will be used
                in constructing a batch. It makes effect only with balance
                sampler.
        **kwargs:

    """

    def __init__(self,
                 datasets: list,
                 batch_ratio: list = None,
                 **kwargs):
        assert isinstance(datasets, list)
        datasets = build_datasets(datasets, default_args=kwargs)
        self.root = ''.join([ds.root for ds in datasets])
        data_range = [len(dataset) for dataset in datasets]
        self.data_range = [
            sum(data_range[:i]) for i in range(1,
                                               len(data_range) + 1)
        ]
        self.batch_ratio = batch_ratio
        if self.batch_ratio is not None:
            assert len(self.batch_ratio) == len(datasets), \
                'The length of batch_ratio and datasets should be equal. ' \
                f'But got {len(self.batch_ratio)} batch_ratio and ' \
                f'{len(datasets)} datasets.'
        super(ConcatDatasets, self).__init__(datasets=datasets)
