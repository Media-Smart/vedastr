import torch
import torchvision.models as tm
from torch.utils.data import ConcatDataset as _ConcatDataset
from vedastr.dataloaders import build_dataloader
from vedastr.dataloaders.samplers import build_sampler
from vedastr.datasets import build_datasets
from vedastr.transforms import build_transform
from vedastr.lr_schedulers import build_lr_scheduler
from vedastr.optimizers import build_optimizer
import configs.new_rosetta as nr


class Dataset1:
    def __getitem__(self, item):
        return item * -1

    def __len__(self):
        return 10


class Dataset2:
    def __init__(self):
        self.a = 1

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
    model = tm.resnet18(False).cuda()
    op_cfg = dict(type='Adam', lr=1.0, params=model.parameters())
    lr_scheduler = dict(type='StepLR',
                        iter_based=False,
                        milestones=[3, 5],
                        )
    opt = build_optimizer(op_cfg)
    lrs = build_lr_scheduler(lr_scheduler, dict(
        max_epochs=20,
        niter_per_epoch=2,
        optimizer=opt
    ))
    iter_based = lrs._iter_based
    # for j in range(10):
    #
    #     for i in range(3):
    #         opt.step()
    #         lrs.iter_nums()
    #     if not iter_based:
    #         lrs.step()
    #     for param_group in opt.param_groups:
    #         print(param_group['lr'])

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
    # datasets = build_datasets(data_cfg, dict(transform=transform))
    sampler = build_sampler(sampler_cfg, dict(dataset=datasets))
    # print(len(sampler))
    dataloader = build_dataloader(loader_cfg, dict(dataset=datasets, sampler=sampler))

    count = 0
    for img in dataloader:
        count += 1
        print('yes %s %s' % (count, len(dataloader)))


if __name__ == '__main__':
    main()
