max_iters = 40000
num = 10

assert max_iters % num == 0
interval = int(max_iters / num)

# optimizer
optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None,
)

# learning policy
# learning policy
param_scheduler = [
    # 线性学习率预热调度器
    dict(type='LinearLR',
         start_factor=1e-7,
         by_epoch=False,  # 按迭代更新学习率
         begin=0,
         end=1500),  # 预热前 50 次迭代
    # 主学习率调度器
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=1.0,
        begin=1500,
        end=max_iters,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=interval, val_begin=0)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, save_optimizer=False, save_param_scheduler=False,
        interval=interval, max_keep_ckpts=1,
        # save_best=['mIoU', 'mPrecision', 'mRecall'],
        save_best=['mIoU', 'RIoU0.50', 'RIoU0.75'],
        rule=['greater']),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, ),
)

randomness = dict(seed=54978327)
