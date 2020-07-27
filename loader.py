from torch.utils.data import ConcatDataset as _ConcatDataset
from vedastr.dataloaders import build_dataloader
from vedastr.dataloaders.samplers import build_sampler
from vedastr.datasets import build_datasets
from vedastr.transforms import build_transform
import configs.new_rosetta as nr


class Dataset1:
    def __getitem__(self, item):
        return item * -1

    def __len__(self):
        return 10


class Dataset2:
    def __getitem__(self, item):
        return item

    def __len__(self):
        return 4


class ConcatDatasets(_ConcatDataset):

    def __init__(self, datasets: list, batch_ratio: list = None, **kwargs):
        assert isinstance(datasets, list)

        data_range = [len(dataset) for dataset in datasets]
        self.data_range = [sum(data_range[:i]) for i in range(1, len(data_range) + 1)]
        self.batch_ratio = batch_ratio

        super(ConcatDatasets, self).__init__(datasets=datasets)


def main():
    size = (32, 100)
    mean, std = 0.5, 0.5

    train_sensitive = False
    train_character = '0123456789abcdefghijklmnopqrstuvwxyz'
    test_sensitive = False
    test_character = '0123456789abcdefghijklmnopqrstuvwxyz'
    batch_size = 4
    batch_max_length = 25
    data_filter_off = False
    test_dataset_params = dict(
        batch_max_length=batch_max_length,
        data_filter_off=data_filter_off,
        character=test_character,
    )

    trans_cfg = transform = [
        dict(type='Sensitive', sensitive=train_sensitive),
        dict(type='ColorToGray'),
        dict(type='Resize', size=size),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=mean, std=std),
    ]
    base_cfg = dict(type='LmdbDataset', root=r'D:\DATA_ALL\STR\lmdb\CUTE80',
                    **test_dataset_params)
    base_cfg2 = dict(type='LmdbDataset', root=r'D:\DATA_ALL\STR\MJ_test',
                     **test_dataset_params)
    data_cfg = dict(
        type='ConcatDatasets',
        datasets=[
            Dataset1(),
            Dataset2(),
        ],
        **test_dataset_params,
        batch_ratio=[0.5, 0.5]
    )
    datasets = ConcatDatasets(datasets=[Dataset2(), Dataset1()], batch_ratio=[0.5, 0.5])



    sampler_cfg = dict(
        type='BalanceSampler',
        batch_size=batch_size,
        shuffle=True,
        oversample=True,
        downsample=False,
    )
    loader_cfg = dict(
        type='DataLoader',
        num_workers=0,
        drop_last=False,
        batch_size=batch_size,
    )

    # build trans
    transform = build_transform(trans_cfg)

    datasets = build_datasets(nr.train['data']['train']['dataset'], dict(transform=transform))
    sampler = build_sampler(nr.train['data']['train']['sampler'], dict(dataset=datasets))
    # print(len(sampler))
    dataloader = build_dataloader(nr.train['data']['train']['dataloader'], dict(dataset=datasets, sampler=sampler))
    for img in dataloader:
        print('yes')


if __name__ == '__main__':
    main()
