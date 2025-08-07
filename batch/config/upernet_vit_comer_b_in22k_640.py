_base_ = [
    './_base_/schedule_base.py',
    './_base_/default_runtime.py',
]
num_classes = 2
crop_size = (640, 640)
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
    pretrained='/home/yansu/mmlabmat/segmantation/mmseg-extension-test/pretrained/vit-comer/deit_base_patch16_224-b5f2ef4d.pth',
    backbone=dict(
        type='ext-ViTCoMer',
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.3,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        dim_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        window_attn=[
            False, False, False, False, False, False, False, False, False,
            False, False, False
        ],
        window_size=[
            None, None, None, None, None, None, None, None, None, None, None,
            None
        ]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
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
        in_channels=768,
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
    # test_cfg=dict(crop_size=(768, 768,), mode='slide', stride=(640, 640,)),
    test_cfg=dict(crop_size=(704, 704,), mode='slide', stride=(640, 640,)),
)

# By default, models are trained on 2 GPUs with 4 images per GPU
# train_dataloader = dict(num_batch_per_epoch=5)
# val_dataloader = dict(num_batch_per_epoch=5)
# test_dataloader = val_dataloader

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='./runsbatch/graphene_batch1_ViTCoMer_b/local'),
        # dict(type='TensorboardVisBackend', save_dir='runs/tb'),
        dict(type='MLflowVisBackend', save_dir='./runsbatch/graphene_batch1_ViTCoMer_b/mlruns',
             exp_name='batch1',
             run_name='ViTCoMer_b',
             ),
    ])
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend', save_dir='tmp/local'),
#         dict(type='TensorboardVisBackend', save_dir='tmp/tb'),
#         dict(type='MLflowVisBackend', save_dir='tmp/mlruns',
#              exp_name='upernet_internimage_b_512_80k_b2d2_2dmat.py', ),
#     ])
