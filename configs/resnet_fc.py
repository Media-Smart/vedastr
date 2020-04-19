# work dir
root_workdir = 'workdir/'

# seed
seed = 1111

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
batch_size = 192
mean, std = 0.5, 0.5  # normalize mean and std
size = (32, 100)
batch_max_length = 25
data_filter_off = False
sensitive = False
character = '0123456789abcdefghijklmnopqrstuvwxyz'  # need character

# dataset params
dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter_off=data_filter_off,
)

data_root = './data/data_lmdb_release/'

# train data
train_root = data_root + 'training/'
## MJ dataset
train_root_mj = train_root + 'MJ/'
mj_folder_names = ['/MJ_test', 'MJ_valid', 'MJ_train']
## ST dataset
train_root_st = train_root + 'ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name) for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

# valid
valid_root = data_root + 'validation/'
valid_dataset = [dict(type='LmdbDataset', root=valid_root, **dataset_params)]

# test
test_root = data_root + 'evaluation/'
test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077', 'IIIT5k_3000', 'SVT', 'SVTP']
test_dataset = [dict(type='LmdbDataset', root=test_root + folder_name, **dataset_params) for folder_name in
                test_folder_names]

# transforms
transforms = [
    dict(type='Sensitive', sensitive=sensitive),
    dict(type='ColorToGray'),
    dict(type='Resize', size=size),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=mean, std=std),
]

data = dict(
    train=dict(
        transforms=transforms,
        datasets=[
            dict(
                type='ConcatDatasets',
                datasets=train_dataset_mj,
                **dataset_params,
            ),
            dict(
                type='ConcatDatasets',
                datasets=train_dataset_st,
                **dataset_params,
            ),
        ],
        loader=dict(
            type='BatchBalanceDataloader',
            batch_size=batch_size,
            each_batch_ratio=[0.5, 0.5],
            each_usage=[1.0, 1.0],
            shuffle=True,
        ),
    ),
    val=dict(
        transforms=transforms,
        datasets=valid_dataset,
        loader=dict(
            type='TestDataloader',
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
    ),
    test=dict(
        transforms=transforms,
        datasets=test_dataset,
        loader=dict(
            type='TestDataloader',
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
    ),
)

# 3. converter
converter = dict(
    type='FCConverter',
    character=character,
)

# 4. model

num_class = 37
norm_cfg = dict(type='BN')
model = dict(
    type='GModel',
    need_text=False,
    body=dict(
        type='GBody',
        pipelines=[
            dict(
                type='FeatureExtractorComponent',
                from_layer='input',
                to_layer='cnn_feat',
                arch=dict(
                    encoder=dict(
                        backbone=dict(
                            type='GResNet',
                            layers=[
                                ('conv', dict(type='ConvModule', in_channels=1, out_channels=64, kernel_size=3,
                                              stride=1, padding=1, bias=False, norm_cfg=norm_cfg)),
                                ('conv', dict(type='ConvModule', in_channels=64, out_channels=128, kernel_size=3,
                                              stride=1, padding=1, bias=False, norm_cfg=norm_cfg)),
                                ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                ('block', dict(block_name='BasicBlock', planes=256, blocks=1, stride=1)),
                                ('conv', dict(type='ConvModule', in_channels=256, out_channels=256, kernel_size=3,
                                              stride=1, padding=1, bias=False, norm_cfg=norm_cfg)),
                                ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                ('block', dict(block_name='BasicBlock', planes=512, blocks=2, stride=1)),
                                ('conv', dict(type='ConvModule', in_channels=512, out_channels=512, kernel_size=3,
                                              stride=1, padding=1, bias=False, norm_cfg=norm_cfg)),
                                ('pool', dict(type='MaxPool2d', kernel_size=2, stride=(2, 1), padding=(0, 1))),
                                ('block', dict(block_name='BasicBlock', planes=1024, blocks=5, stride=1)),
                                ('conv', dict(type='ConvModule', in_channels=1024, out_channels=1024, kernel_size=3,
                                              stride=1, padding=1, bias=False, norm_cfg=norm_cfg)),
                                ('block', dict(block_name='BasicBlock', planes=1024, blocks=3, stride=1)),
                                ('conv', dict(type='ConvModule', in_channels=1024, out_channels=1024, kernel_size=2,
                                              stride=(2, 1), padding=(0, 1), bias=False, norm_cfg=norm_cfg)),
                                ('conv', dict(type='ConvModule', in_channels=1024, out_channels=1024, kernel_size=2,
                                              stride=1, padding=0, bias=False, norm_cfg=norm_cfg)),
                            ],
                        ),
                    ),
                    collect=dict(type='CollectBlock', from_layer='c4'),
                ),
            ),
        ],
    ),
    head=dict(
        type='FCHead',
        in_channels=1024,
        out_channels=962,
        num_class=num_class,
        from_layer='cnn_feat',
        batch_max_length=batch_max_length,
        activation=None,
        pool=dict(
            type='AdaptiveAvgPool2d',
            output_size=1,
        ),
    ),
)

## 4.1 resume
resume = None

# 5. criterion
criterion = dict(type='CrossEntropyLoss', ignore_index=37)

# 6. optim
optimizer = dict(type='Adam', lr=0.001)

# 7. lr scheduler
lr_scheduler = dict(type='StepLR', niter_per_epoch=100000, max_epochs=3, milestones=[100000, 200000])

# 8. runner
max_iterations = 300000
runner = dict(
    type='Runner',
    iterations=max_iterations,
    trainval_ratio=2000,
    snapshot_interval=20000,
    grad_clip=0,
)

# 9. device
gpu_id = '0'
