from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate

from .registry import DATALOADERS


@DATALOADERS.register_module
class TestDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, pin_memory=True, sampler=None, batch_sampler=None,
                 collate_fn=default_collate, drop_last=False, time_out=0, worker_init_fn=None):
        datasets = ConcatDataset(dataset)

        super(TestDataloader, self).__init__(datasets, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers,
                                             pin_memory=pin_memory, sampler=sampler, batch_sampler=batch_sampler,
                                             collate_fn=collate_fn, drop_last=drop_last, timeout=time_out,
                                             worker_init_fn=worker_init_fn)
