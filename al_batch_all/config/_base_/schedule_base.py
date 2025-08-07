_max_epochs = 10
_num = 10

assert _max_epochs % _num == 0
_interval = int(_max_epochs / _num)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=None,
)

# learning policy
# learning policy
# param_scheduler = [
#     # 线性学习率预热调度器
#     dict(type='LinearLR',
#          start_factor=1e-6,
#          by_epoch=True,  # 按迭代更新学习率
#          begin=0,
#          end=1),  # 预热前 50 次迭代
#     # 主学习率调度器
#     dict(
#         type='PolyLR',
#         eta_min=1e-7,
#         power=1.0,
#         begin=1,
#         end=max_epochs,
#         by_epoch=True)
# ]
param_scheduler = dict(
    type='PolyLR',
    eta_min=1e-7,
    power=1.0,
    begin=1,
    end=_max_epochs,
    by_epoch=True)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=_max_epochs, val_begin=0, val_interval=_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True,
        interval=_interval, max_keep_ckpts=1,
        # save_best=['mIoU', 'mPrecision', 'mRecall'],
        save_best=['mIoU'],  # 'mIoUC'
        rule=['greater']),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, ),
)

# randomness = dict(seed=857341)
randomness = dict(seed=7214732)