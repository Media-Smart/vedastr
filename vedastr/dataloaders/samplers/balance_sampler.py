import copy
import random
from torch.utils.data import Sampler

from .registry import SAMPLER


@SAMPLER.register_module
class BalanceSampler(Sampler):
    """
    Balance Sampler for Dataloader. Based on a given ratio, BalanceSampler provide
    an iterable over the index of given dataset.
    Arguments:
        dataset (Dataset): In this project, the dataset should be the an instance of
                          vedastr.dataset.ConcarDataset. You can implement a dataset
                          which should have attr: data_range, batch_ratio by yourself.
        batch_size (int):  how many samplers per batch
        shuffle (bool): The indices will be shuffled at each epoch.
        oversample (bool): Set to True to oversample smaller sampler set.
        downsample (bool): Set to True to downsample bigger sampler set.
    .. warning:: If both oversample and downsample is True, BanlanceSampler will do
                oversample first. That means downsample will do no effect.
    """

    def __init__(self, dataset, batch_size, shuffle, oversample=False, downsample=False):
        assert hasattr(dataset, 'data_range')
        assert hasattr(dataset, 'batch_ratio')
        self.dataset = dataset
        self.samples_range = dataset.data_range
        self.batch_ratio = dataset.batch_ratio
        self.batch_size = batch_size
        self.oversample = oversample
        self.downsample = downsample
        self.shuffle = shuffle
        self._generate_indices_()

    @property
    def _num_samples(self):
        return self.num_samples

    @_num_samples.setter
    def _num_samples(self, v):
        self.num_samples = v

    def _generate_indices_(self):
        self._num_samples = len(self.dataset)
        indices_ = []
        # TODO, elegant
        for idx, v in enumerate(self.samples_range):
            if idx == 0:
                temp = list(range(v))
                if self.shuffle:
                    random.shuffle(temp)
                indices_.append(temp)
            else:
                temp = list(range(self.samples_range[idx - 1], v))
                if self.shuffle:
                    random.shuffle(temp)
                indices_.append(temp)
        if self.oversample:
            indices_ = self._oversample(indices_)
        if self.downsample:
            indices_ = self._downsample(indices_)
        return indices_

    def __iter__(self):
        indices_ = self._generate_indices_()
        total_nums = len(self) // self.batch_size
        sizes = [int(self.batch_size * br) for br in self.batch_ratio]
        final_index = [total_nums * size for size in sizes]
        indices = []
        for idx2 in range(total_nums):
            for idx3, size in enumerate(sizes):
                indices += indices_[idx3][idx2 * size:(idx2 + 1) * size]
        for idx4, index in enumerate(final_index):
            indices += indices_[idx4][index:]
        return iter(indices)

    def _oversample(self, indices):
        max_len = max([len(index) for index in indices])
        result_indices = []
        for idx, index in enumerate(indices):
            current_len = len(index)
            need_num = max_len - current_len
            total_nums = need_num // current_len
            mod_nums = need_num % current_len
            for _ in range(total_nums):
                new_index = copy.copy(index)
                if self.shuffle:
                    random.shuffle(new_index)
                index += new_index
            index += random.sample(index, mod_nums)
            result_indices.append(index)
        self._num_samples = max_len * len(indices)
        return result_indices

    def _downsample(self, indices):
        min_len = min([len(index) for index in indices])
        result_indices = []
        for idx, index in enumerate(indices):
            index = random.sample(index, min_len)
            result_indices.append(index)
        self._num_samples = min_len * len(indices)
        return result_indices

    def __len__(self):
        return self._num_samples
