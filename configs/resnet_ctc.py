###############################################################################
# 1. deploy
size = (32, 100)
mean, std = 0.5, 0.5

character = 'abcdefghijklmnopqrstuvwxyz0123456789'
sensitive = False
batch_max_length = 25

norm_cfg = dict(type='BN')
num_class = len(character) + 1

deploy = dict(
    transform=[
        dict(type='Sensitive', sensitive=sensitive, need_character=character),
        dict(type='ToGray'),
        dict(type='Resize', size=size),
        dict(type='Normalize', mean=mean, std=std),
        dict(type='ToTensor'),
    ],
    converter=dict(
        type='CTCConverter',
        character=character,
        batch_max_length=batch_max_length,
    ),
    model=dict(
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
                                    ('conv', dict(type='ConvModule', in_channels=1, out_channels=32, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=norm_cfg)),
                                    ('conv', dict(type='ConvModule', in_channels=32, out_channels=64, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=norm_cfg)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                    ('block', dict(block_name='BasicBlock', planes=128, blocks=1, stride=1)),
                                    ('conv', dict(type='ConvModule', in_channels=128, out_channels=128, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=norm_cfg)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                    ('block', dict(block_name='BasicBlock', planes=256, blocks=2, stride=1)),
                                    ('conv', dict(type='ConvModule', in_channels=256, out_channels=256, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=norm_cfg)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=(2, 1), padding=(0, 1))),
                                    ('block', dict(block_name='BasicBlock', planes=512, blocks=5, stride=1)),
                                    ('conv', dict(type='ConvModule', in_channels=512, out_channels=512, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=norm_cfg)),
                                    ('block', dict(block_name='BasicBlock', planes=512, blocks=3, stride=1)),
                                    ('conv', dict(type='ConvModule', in_channels=512, out_channels=512, kernel_size=2,
                                                  stride=(2, 1), padding=(0, 1), norm_cfg=norm_cfg)),
                                    ('conv', dict(type='ConvModule', in_channels=512, out_channels=512, kernel_size=2,
                                                  stride=1, padding=0, norm_cfg=norm_cfg)),
                                ],
                            ),
                        ),
                        collect=dict(type='CollectBlock', from_layer='c4'),
                    ),
                ),
            ],
        ),
        head=dict(
            type='CTCHead',
            from_layer='cnn_feat',
            num_class=num_class,
            in_channels=512,
            pool=dict(
                type='AdaptiveAvgPool2d',
                output_size=(1, None),
            ),
        ),
    ),
)

###############################################################################
# 2.common

common = dict(
    seed=1111,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metric=dict(type='Accuracy'),
)
###############################################################################

dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter=True,
    character=character,
)

test_dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter=False,
    character=character,
)


data_root = '../../../../dataset/str/data/data_lmdb_release/'

###############################################################################
# 3. test

batch_size = 192

# data
test_root = data_root + 'evaluation/'
# test_folder_names = ['CUTE80']
test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077',
                     'IIIT5k_3000', 'SVT', 'SVTP']
test_dataset = [dict(type='LmdbDataset', root=test_root + f_name,
                     **test_dataset_params) for f_name in test_folder_names]

test = dict(
    data=dict(
        dataloader=dict(
            type='DataLoader',
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
        dataset=test_dataset,
        transform=deploy['transform'],
    ),
    postprocess_cfg=dict(
        sensitive=sensitive,
        character=character,
    ),
)

###############################################################################
# 4. train

root_workdir = 'workdir'

# train data
train_root = data_root + 'training/'
# MJ dataset
train_root_mj = train_root + 'MJ/'
mj_folder_names = ['MJ_test', 'MJ_valid', 'MJ_train']
# ST dataset
train_root_st = train_root + 'ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name)
                    for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

# valid
valid_root = data_root + 'validation/'
valid_dataset = dict(type='LmdbDataset', root=valid_root, **test_dataset_params)

# train transforms
train_transforms = [
    dict(type='Sensitive', sensitive=sensitive, need_character=character),
    dict(type='ToGray'),
    dict(type='Resize', size=size),
    dict(type='Normalize', mean=mean, std=std),
    dict(type='ToTensor'),
]

max_iterations = 300000
milestones = [150000, 250000]

train = dict(
    data=dict(
        train=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=batch_size,
                num_workers=4,
            ),
            sampler=dict(
                type='BalanceSampler',
                batch_size=batch_size,
                shuffle=True,
                oversample=True,
            ),
            dataset=dict(
                type='ConcatDatasets',
                datasets=[
                    dict(
                        type='ConcatDatasets',
                        datasets=train_dataset_mj,
                    ),
                    dict(
                        type='ConcatDatasets',
                        datasets=train_dataset_st,
                    )
                ],
                batch_ratio=[0.5, 0.5],
                **dataset_params,
            ),
            transform=train_transforms,
        ),
        val=dict(
            dataloader=dict(
                type='DataLoader',
                batch_size=batch_size,
                num_workers=4,
                shuffle=False,
            ),
            dataset=valid_dataset,
            transform=deploy['transform'],
        ),
    ),
    optimizer=dict(type='Adadelta', lr=1.0, rho=0.95, eps=1e-8),
    criterion=dict(type='CTCLoss', zero_infinity=True),
    lr_scheduler=dict(type='StepLR',
                      iter_based=True,
                      milestones=milestones,
                      ),
    max_iterations=max_iterations,
    log_interval=10,
    trainval_ratio=2000,
    snapshot_interval=20000,
    save_best=True,
    resume=None,
)
