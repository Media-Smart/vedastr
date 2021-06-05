from torch.utils.data import DistributedSampler

from .registry import DISTSAMPLER
from ...utils import get_dist_info


@DISTSAMPLER.register_module
class DefaultSampler(DistributedSampler):
    """Default distributed sampler."""

    def __init__(self,
                 dataset,
                 shuffle: bool = True,
                 seed=0,
                 drop_last=False):
        if seed is None:
            seed = 0
        rank, num_replicas = get_dist_info()
        super().__init__(dataset=dataset,
                         num_replicas=num_replicas,
                         rank=rank,
                         shuffle=shuffle,
                         seed=seed,
                         drop_last=drop_last,
                         )
