from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate

from .registry import DATALOADERS
from .base import BaseDataloader


@DATALOADERS.register_module
class BatchRandomDataloader(BaseDataloader):
    def __init__(self, dataset, batch_size, each_usage, shuffle=False, num_workers=4, pin_memory=True,
                 sampler=None, batch_sampler=None, collate_fn=default_collate, drop_last=False, time_out=0,
                 worker_init_fn=None):
        dataset = [ConcatDataset(dataset)]

        super(BatchRandomDataloader, self).__init__(dataset=dataset,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    shuffle=shuffle,
                                                    each_usage=each_usage)

        _dataloader = DataLoader(
            self.dataset, batch_size=batch_size,
            shuffle=shuffle, num_workers=num_workers,
            pin_memory=pin_memory, sampler=sampler, batch_sampler=batch_sampler,
            collate_fn=collate_fn, drop_last=drop_last, timeout=time_out, worker_init_fn=worker_init_fn
        )

        self.data_loader_list.append(_dataloader)
        self.dataloader_iter_list.append(iter(_dataloader))
