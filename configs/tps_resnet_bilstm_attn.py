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
size = 32, 100
batch_max_length = 25
data_filter_off = False
sensitive = False
character = 'abcdefghijklmnopqrstuvwxyz0123456789'  # need character

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
    type='AttnConverter',
    character=character,
    batch_max_length=batch_max_length,
)

# 4. model
F = 20  # number of rectification points
norm_cfg = dict(type='BN')
num_class = 38
num_steps = 26
model = dict(
    type='GModel',
    need_text=True,
    body=dict(
        type='GBody',
        pipelines=[
            dict(
                type='RectificatorComponent',
                from_layer='input',
                to_layer='rect',
                arch=dict(
                    type='TPS_STN',
                    F=F,
                    input_size=size,
                    output_size=size,
                    stn=dict(
                        feature_extractor=dict(
                            encoder=dict(
                                backbone=dict(
                                    type='GVGG',
                                    layers=[
                                        ('conv', dict(type='ConvModule', in_channels=1, out_channels=64,
                                                      kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg)),
                                        ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2)),
                                        ('conv', dict(type='ConvModule', in_channels=64, out_channels=128,
                                                      kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg)),
                                        ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2)),
                                        ('conv', dict(type='ConvModule', in_channels=128, out_channels=256,
                                                      kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg)),
                                        ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2)),
                                        ('conv', dict(type='ConvModule', in_channels=256, out_channels=512,
                                                      kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg)),
                                    ],
                                ),
                            ),
                            collect=dict(type='CollectBlock', from_layer='c3')
                        ),
                        pool=dict(type='AdaptiveAvgPool2d', output_size=1),
                        head=[
                            dict(type='FCModule', in_channels=512, out_channels=256),
                            dict(type='FCModule', in_channels=256, out_channels=F * 2, activation=None)
                        ],
                    ),
                ),
            ),
            dict(
                type='FeatureExtractorComponent',
                from_layer='rect',
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
            dict(
                type='SequenceEncoderComponent',
                from_layer='cnn_feat',
                to_layer='rnn_feat',
                arch=dict(
                    type='RNN',
                    input_pool=dict(type='AdaptiveAvgPool2d', output_size=(1, None)),
                    layers=[
                        ('rnn',
                         dict(type='LSTM', input_size=512, hidden_size=256, bidirectional=True, batch_first=True)),
                        ('fc', dict(type='Linear', in_features=512, out_features=256)),
                        ('rnn',
                         dict(type='LSTM', input_size=256, hidden_size=256, bidirectional=True, batch_first=True)),
                        ('fc', dict(type='Linear', in_features=512, out_features=256)),
                    ],
                ),
            ),
        ],
    ),
    head=dict(
        type='AttHead',
        num_class=num_class,
        num_steps=num_steps,
        cell=dict(
            type='LSTMCell',
            input_size=256 + num_class,
            hidden_size=256,
        ),
        input_attention_block=dict(
            type='CellAttentionBlock',
            feat=dict(
                from_layer='rnn_feat',
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                bias=False,
                activation=None,
            ),
            hidden=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                activation=None,
            ),
            fusion_method='add',
            post=dict(
                type='ConvModule',
                in_channels=256,
                out_channels=1,
                kernel_size=1,
                bias=False,
                activation='tanh',
                order=('act', 'conv', 'norm'),
            ),
            post_activation='softmax',
        ),
        generator=dict(
            type='Linear',
            in_features=256,
            out_features=num_class,
        ),
    ),
)

## 4.1 resume
resume = None

# 5. criterion
criterion = dict(type='CrossEntropyLoss', ignore_index=0)

# 6. optim
optimizer = dict(type='Adadelta', lr=1.0, rho=0.95, eps=1e-8)

# 7. lr scheduler

lr_scheduler = dict(type='StepLR', niter_per_epoch=100000, max_epochs=3, milestones=[150000, 250000])

# 8. runner
max_iterations = 300000
runner = dict(
    type='Runner',
    iterations=max_iterations,
    trainval_ratio=2000,
    snapshot_interval=20000,
    grad_clip=5,
)

# 9. device
gpu_id = '0'
