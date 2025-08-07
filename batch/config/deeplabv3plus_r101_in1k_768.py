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
    pretrained='/home/yansu/paper/pretrained/resnet101_v1c-e67eebb6.pth',
    data_preprocessor=data_preprocessor,
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=101,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    decode_head=dict(
        align_corners=False,
        c1_channels=48,
        c1_in_channels=256,
        channels=512,
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        type='DepthwiseSeparableASPPHead'),
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    test_cfg=dict(crop_size=(2048, 2048,), mode='slide', stride=(2048 - 128, 2048 - 128,)),
)

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='./runsbatch/graphene_batch1_UniRepLKNet_b/local'),
        # dict(type='TensorboardVisBackend', save_dir='runs/tb'),
        dict(type='MLflowVisBackend', save_dir='./runsbatch/graphene_batch1_UniRepLKNet_b/mlruns',
             exp_name='batch1',
             run_name='deeplabv3plus',
             ),
    ])
