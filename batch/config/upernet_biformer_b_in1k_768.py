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
    pretrained='/home/yansu/mmlabmat/segmantation/mmseg-extension-test/pretrained/biformer/biformer_base_best.pth',
    backbone=dict(
        type='ext-BiFormer',
        depth=[4, 4, 18, 4],
        embed_dim=[96, 192, 384, 768],
        mlp_ratios=[3, 3, 3, 3],
        n_win=8,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[96, 192, 384, 768],
        head_dim=32,
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        pre_norm=True,
        pe=None,
        auto_pad=True,
        use_checkpoint_stages=[],
        drop_path_rate=0.4),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
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
        in_channels=384,
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
    test_cfg=dict(crop_size=(832, 832,), mode='slide', stride=(768, 768,)),
    # test_cfg=dict(crop_size=(3072, 3072,), mode='slide', stride=(2560, 2560,)),
)

# By default, models are trained on 2 GPUs with 4 images per GPU
# train_dataloader = dict(num_batch_per_epoch=5)
# val_dataloader = dict(num_batch_per_epoch=5)
# test_dataloader = val_dataloader

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='./runsbatch/graphene_batch1_BiFormer_b/local'),
        # dict(type='TensorboardVisBackend', save_dir='runs/tb'),
        dict(type='MLflowVisBackend', save_dir='./runsbatch/graphene_batch1_BiFormer_b/mlruns',
             exp_name='batch1',
             run_name='BiFormer_b',
             ),
    ])
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend', save_dir='tmp/local'),
#         dict(type='TensorboardVisBackend', save_dir='tmp/tb'),
#         dict(type='MLflowVisBackend', save_dir='tmp/mlruns',
#              exp_name='upernet_internimage_b_512_80k_b2d2_2dmat.py', ),
#     ])
