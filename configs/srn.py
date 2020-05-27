# work dir
root_workdir = 'workdir/'

# seed
seed = 6

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
batch_size = 256
mean, std = 0.5, 0.5  # normalize mean and std
size = (64, 256)
mode = 'nearest'
fill = 0
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
# test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077', 'IIIT5k_3000', 'SVT', 'SVTP']
test_folder_names = ['IIIT5k_3000']
test_dataset = [dict(type='LmdbDataset', root=test_root + folder_name, **dataset_params) for folder_name in
                test_folder_names]

# transforms
train_transforms = [
    dict(type='Sensitive', sensitive=sensitive),
    dict(type='KeepHorizontal', clockwise=False),
    dict(type='Resize', size=size, keep_ratio=True, mode=mode),
    dict(type='RandomScale', scales=(0.25, 1.0), step=0.25, mode=mode, p=0.5),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5),
    dict(type='MotionBlur', blur_limit=5, p=0.5),
    dict(type='GaussianNoise', var_limit=(10, 50), mean=0, p=0.5),
    dict(type='RandomPerspective', distortion_scale=0.3, mode=mode, p=0.5),
    dict(type='RandomRotation', degrees=10, expand=False, fill=fill, mode=mode, p=1.0),
    dict(type='PadIfNeeded', size=size, fill=fill),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=mean, std=std),
]
test_transforms = [
    dict(type='Sensitive', sensitive=sensitive),
    dict(type='KeepHorizontal', clockwise=False),
    dict(type='Resize', size=size, keep_ratio=True, mode=mode),
    dict(type='PadIfNeeded', size=size, fill=fill),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=mean, std=std),
]

data = dict(
    train=dict(
        transforms=train_transforms,
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
            type='BatchRandomDataloader',
            batch_size=batch_size,
            each_batch_ratio=[1],
            each_usage=[1.0],
            shuffle=True,
        ),
    ),
    val=dict(
        transforms=test_transforms,
        datasets=valid_dataset,
        loader=dict(
            type='TestDataloader',
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
    ),
    test=dict(
        transforms=test_transforms,
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
    batch_max_length=batch_max_length,
)

# 4. model
num_class = 37
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
                            type='ResNet',
                            arch='resnet50',
                        ),
                    ),
                    decoder=dict(
                        type='GFPN',
                        neck=[
                            dict(
                                type='JunctionBlock',
                                top_down=None,
                                lateral=dict(
                                    from_layer='c5',
                                    type='ConvModule',
                                    in_channels=2048,
                                    out_channels=512,
                                    kernel_size=1,
                                    norm_cfg=None,
                                    activation=None,
                                ),
                                post=None,
                                to_layer='p5',
                            ),  # 32
                            dict(
                                type='JunctionBlock',
                                fusion_method='add',
                                top_down=dict(
                                    from_layer='p5',
                                    upsample=dict(
                                        type='Upsample',
                                        scale_factor=2,
                                        mode=mode,
                                    ),
                                ),
                                lateral=dict(
                                    from_layer='c4',
                                    type='ConvModule',
                                    in_channels=1024,
                                    out_channels=512,
                                    kernel_size=1,
                                    norm_cfg=None,
                                    activation=None,
                                ),
                                post=None,
                                to_layer='p4',
                            ),  # 16
                            dict(
                                type='JunctionBlock',
                                fusion_method='add',
                                top_down=dict(
                                    from_layer='p4',
                                    upsample=dict(
                                        type='Upsample',
                                        scale_factor=2,
                                        mode=mode,
                                    ),
                                ),
                                lateral=dict(
                                    from_layer='c3',
                                    type='ConvModule',
                                    in_channels=512,
                                    out_channels=512,
                                    kernel_size=1,
                                    norm_cfg=None,
                                    activation=None,
                                ),
                                post=dict(
                                    type='ConvModule',
                                    in_channels=512,
                                    out_channels=512,
                                    kernel_size=3,
                                    padding=1,
                                    norm_cfg=None,
                                    activation=None,
                                ),
                                to_layer='p3',
                            ),  # 8
                        ],
                    ),
                    collect=dict(type='CollectBlock', from_layer='p3'),
                ),
            ),
            dict(
                type='SequenceEncoderComponent',
                from_layer='cnn_feat',
                to_layer='tf_feat',
                arch=dict(
                    type='Transformer',
                    num_layers=2,
                    d_model=512,
                    nhead=8,
                    dim_feedforward=512,
                    dropout=0.1,
                    activation='relu',
                    norm_cfg=None,
                    use_pos_encode=True,
                    pos_encode_len=256,
                ),
            ),
            dict(
                type='BrickComponent',
                from_layer='tf_feat',
                to_layer='seq_feat',
                arch=dict(
                    type='PVABlock',
                    num_steps=batch_max_length+1,
                    in_channels=512,
                    embedding_channels=512,
                    inner_channels=512,
                ),
            ),
        ],
    ),
    head=dict(
        type='Head',
        from_layer='seq_feat',
        generator=dict(
            type='FCModule',
            in_channels=512,
            out_channels=num_class,
            bias=True,
            activation=None,
        ),
    ),
)

## 4.1 resume
resume = None

# 5. criterion
criterion = dict(type='CrossEntropyLoss', ignore_index=num_class)

# 6. optim
optimizer = dict(type='Adam', lr=1e-4)

# 7. lr scheduler
epochs = 7
decay_epochs = [3,5]
niter_per_epoch = int(55000 * 256 / batch_size)
milestones = [niter_per_epoch * epoch for epoch in decay_epochs]
max_iterations = epochs * niter_per_epoch
lr_scheduler = dict(type='StepLR', niter_per_epoch=niter_per_epoch, max_epochs=epochs, milestones=milestones, gamma=0.1, warmup_epochs=1)

# 8. runner
runner = dict(
    type='Runner',
    iterations=max_iterations,
    trainval_ratio=2000,
    snapshot_interval=niter_per_epoch,
)

# 9. device
gpu_id = '0,5,6,9'
