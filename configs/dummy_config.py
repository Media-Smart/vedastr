###############################################################################
# 1. deploy
deploy = dict(
    gpu_id='',
    transform=dict(),
    model=dict(),
    deply_dataset=dict(),
)

###############################################################################
# 2. common
common = dict(
    seed=int,
    logger=dict(),
    cudnn_deterministic=bool,
    cudnn_benchmark=bool,
    converter=dict(),
    metric=dict(),
)

###############################################################################
# 3. test

test = dict(
    data=dict(
        dataloader=dict(),
        dataset=[dict()],
        transform=deploy['transform'],
    ),
    postprocess_cfg=dict(),
)

###############################################################################
# 4. train

train = dict(
    data=dict(
        train=dict(
            dataloader=dict(),
            dataset=[dict()],
            transform=[dict()]
        ),
        val=dict(
            dataloader=dict(),
            dataset=[dict()],
            transform=deploy['transform'],
        ),
    ),
    optimizer=dict(),
    criterion=dict(),
    lr_scheduler=dict(),
    max_iterations=int,
    log_interval=int,
    trainval_ratio=int,
    snapshot_interval=int,
    save_best=bool,
    resume=(str, None),
)
