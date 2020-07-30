import string
size = (64, 256)
mean, std = 0.5, 0.5

character = 'abcdefghijklmnopqrstuvwxyz0123456789'
character = string.printable[:96]
sensitive = False
batch_max_length = 25

data_filter_off = False
dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter_off=data_filter_off,
    character=character,
)

dataset1 = dict(type='LmdbDataset', root=r'D:\DATA_ALL\STR\MJ_test')
dataset2 = dict(type='LmdbDataset', root=r'D:\DATA_ALL\STR\lmdb\CUTE80')

dataset = dict(
    type='ConcatDatasets',
    datasets=[
        dataset1,
        dataset2
    ],
    **dataset_params,
)

transforms = [
    dict(type='Sensitive', sensitive=sensitive),
    dict(type='ColorToGray'),
    dict(type='Resize', size=size),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=mean, std=std),
]
