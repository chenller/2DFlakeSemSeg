_base_ = [
    './_base_/schedule_base.py',
    './_base_/default_runtime.py',
]
num_classes = 2
crop_size = (768, 768)
data_preprocessor = dict(
    type='SegDataPreProcessor', _scope_='mmseg',
    size=crop_size,
    mean=[137.72834198963133, 114.5914692446968, 127.20275534819349, ],
    std=[18.11057238792935, 17.217833654534715, 14.975338896736353, ],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=64),
)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ext-UniRepLKNet',
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.3,
        kernel_sizes=None,
        with_cp=True,
        attempt_use_lk_impl=False,
        init_cfg=dict(type='Pretrained',
                      checkpoint="/home/yansu/mmlabmat/segmantation/mmseg-extension-test/pretrained/unireplknet/unireplknet_b_in22k_pretrain.pth",
                      # filepath or None
                      )
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    test_cfg=dict(crop_size=(3200, 3200,), mode='slide', stride=(2944, 2944,)),
)

# By default, models are trained on 2 GPUs with 4 images per GPU
# train_dataloader = dict(num_batch_per_epoch=5)
# val_dataloader = dict(num_batch_per_epoch=5)
# test_dataloader = val_dataloader

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='./runsbatch/graphene_batch1_UniRepLKNet_b/local'),
        # dict(type='TensorboardVisBackend', save_dir='runs/tb'),
        dict(type='MLflowVisBackend', save_dir='./runsbatch/graphene_batch1_UniRepLKNet_b/mlruns',
             exp_name='batch1',
             run_name='UniRepLKNet_b',
             ),
    ])
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend', save_dir='tmp/local'),
#         dict(type='TensorboardVisBackend', save_dir='tmp/tb'),
#         dict(type='MLflowVisBackend', save_dir='tmp/mlruns',
#              exp_name='upernet_internimage_b_512_80k_b2d2_2dmat.py', ),
#     ])
