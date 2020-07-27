# work directory
root_workdir = 'workdir'

###############################################################################
size = (32, 100)
crop_size = 224
padding_value = 127.5
mean, std = 0.5, 0.5

train_sensitive = True
train_character = '0123456789abcdefghijklmnopq' \
                  'rstuvwxyzABCDEFGHIJKLMNOPQRS' \
                  'TUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'  # need character
test_sensitive = False
test_character = '0123456789abcdefghijklmnopqrstuvwxyz'
batch_size = 32
batch_max_length = 25
fill = 0
mode = 'nearest'

dropout = 0.1
n_e = 9
n_d = 3
hidden_dim = 256
n_head = 8
batch_norm = dict(type='BN')
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)
num_class = len(train_character) + 1
num_steps = batch_max_length + 1

# 1. deploy
deploy = dict(
    gpu_id='3',
    transform=[
        dict(type='Sensitive', sensitive=test_sensitive),
        dict(type='ColorToGray'),
        dict(type='Resize', size=size),
        dict(type='ToTensor'),
        dict(type='Normalize', mean=mean, std=std),
    ],
    model=dict(
        type='GModel',
        need_text=True,
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
                                                  stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('conv', dict(type='ConvModule', in_channels=64, out_channels=128, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                    ('conv', dict(type='ConvModule', in_channels=128, out_channels=256, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('conv', dict(type='ConvModule', in_channels=256, out_channels=512, kernel_size=3,
                                                  stride=1, padding=1, norm_cfg=batch_norm)),
                                    ('pool', dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0)),
                                ],
                            ),
                        ),
                        collect=dict(type='CollectBlock', from_layer='c2'),
                    ),
                ),
                dict(
                    type='SequenceEncoderComponent',
                    from_layer='cnn_feat',
                    to_layer='src',
                    arch=dict(
                        type='TransformerEncoder',
                        position_encoder=dict(
                            type='Adaptive2DPositionEncoder',
                            in_channels=hidden_dim,
                            max_h=100,
                            max_w=100,
                            dropout=dropout,
                        ),
                        encoder_layer=dict(
                            type='TransformerEncoderLayer2D',
                            attention=dict(
                                type='MultiHeadAttention',
                                in_channels=hidden_dim,
                                k_channels=hidden_dim,
                                v_channels=hidden_dim,
                                n_head=n_head,
                                dropout=dropout,
                            ),
                            attention_norm=layer_norm,
                            feedforward=dict(
                                type='Feedforward',
                                layers=[
                                    dict(type='ConvModule', in_channels=hidden_dim, out_channels=hidden_dim * 4,
                                         kernel_size=3, padding=1,
                                         activation='relu', dropout=dropout),
                                    dict(type='ConvModule', in_channels=hidden_dim * 4, out_channels=hidden_dim,
                                         kernel_size=3, padding=1,
                                         activation=None, dropout=dropout),
                                ]
                            ),
                            feedforward_norm=layer_norm,
                        ),
                        num_layers=n_e,
                    ),
                ),
            ],
        ),
        head=dict(
            type='TransformerHead',
            src_from='src',
            decoder=dict(
                type='TransformerDecoder',
                position_encoder=dict(
                    type='PositionEncoder1D',
                    in_channels=hidden_dim,
                    max_len=100,
                    dropout=dropout,
                ),
                decoder_layer=dict(
                    type='TransformerDecoderLayer1D',
                    self_attention=dict(
                        type='MultiHeadAttention',
                        in_channels=hidden_dim,
                        k_channels=hidden_dim,
                        v_channels=hidden_dim,
                        n_head=n_head,
                        dropout=dropout,
                    ),
                    self_attention_norm=layer_norm,
                    attention=dict(
                        type='MultiHeadAttention',
                        in_channels=hidden_dim,
                        k_channels=hidden_dim,
                        v_channels=hidden_dim,
                        n_head=n_head,
                        dropout=dropout,
                    ),
                    attention_norm=layer_norm,
                    feedforward=dict(
                        type='Feedforward',
                        layers=[
                            dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 4, bias=True,
                                 activation='relu',
                                 dropout=dropout),
                            dict(type='FCModule', in_channels=hidden_dim * 4, out_channels=hidden_dim, bias=True,
                                 activation=None,
                                 dropout=dropout),
                        ]
                    ),
                    feedforward_norm=layer_norm,
                ),
                num_layers=n_d,
            ),
            generator=dict(
                type='Linear',
                in_features=hidden_dim,
                out_features=num_class,
            ),
            embedding=dict(
                type='Embedding',
                num_embeddings=num_class + 1,
                embedding_dim=hidden_dim,
                padding_idx=num_class,
            ),
            num_steps=num_steps,
            pad_id=num_class,
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
    converter=dict(
        type='AttnConverter',
        character=train_character,
        batch_max_length=batch_max_length,
        go_last=True,
    ),
    metric=dict(type='Accuracy'),
)
###############################################################################
data_filter_off = False
train_dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter_off=data_filter_off,
    character=train_character,
)
test_dataset_params = dict(
    batch_max_length=batch_max_length,
    data_filter_off=data_filter_off,
    character=test_character,
)
data_root = '/DATA7_DB7/data/sjun/github/vedastr/data/data_lmdb_release/'

# train data
train_root = data_root + 'training/'
## MJ dataset
train_root_mj = train_root + 'MJ/'
mj_folder_names = ['/MJ_test', 'MJ_valid', 'MJ_train']
## ST dataset
train_root_st = train_root + 'ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name)
                    for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

# valid
valid_root = data_root + 'validation/'
valid_dataset = [dict(type='LmdbDataset',
                      root=valid_root,
                      **test_dataset_params)
                 ]

# test
test_root = data_root + 'evaluation/'
test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077',
                     'IIIT5k_3000', 'SVT', 'SVTP']
test_dataset = [dict(type='LmdbDataset', root=test_root + f_name,
                     **test_dataset_params) for f_name in test_folder_names]

###############################################################################
# 3. test
test = dict(
    data=dict(
        dataloader=dict(
            type='DataLoader',
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
        ),
        dataset=dict(
            type='ConcatDatasets',
            datasets=test_dataset,
        ),
        transform=deploy['transform'],
    ),
    postprocess_cfg=dict(
        sensitive=test_sensitive,
        character=test_character,
    ),
)

###############################################################################
train_transforms = [
    dict(type='Sensitive', sensitive=train_sensitive),
    dict(type='ColorToGray'),
    dict(type='RandomNormalRotation', mean=0, std=34, expand=True,
         center=None, fill=fill, mode=mode, p=0.5),
    dict(type='Resize', size=size),
    dict(type='ToTensor'),
    dict(type='Normalize', mean=mean, std=std),
]

max_epochs = 6
milestones = [2, 4]

# 4. train
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
                **train_dataset_params,
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
    optimizer=dict(type='Adam', lr=1e-4),
    criterion=dict(type='CrossEntropyLoss', ignore_index=num_class),
    lr_scheduler=dict(type='StepLR',
                      iter_based=False,
                      milestones=milestones,
                      gamma=0.1,
                      warmup_epochs=0.1,
                      ),
    max_epochs=max_epochs,
    log_interval=10,
    trainval_ratio=2000,
    snapshot_interval=20000,
    save_best=True,
    resume=None,
)
