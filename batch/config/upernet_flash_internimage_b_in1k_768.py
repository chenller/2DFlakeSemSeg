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
        type='ext-FlashInternImage',
        channels=112,
        core_op='DCNv4',
        depths=[4, 4, 21, 4, ],
        drop_path_rate=0.3,
        dw_kernel_size=3,
        groups=[7, 14, 28, 56, ],
        init_cfg=dict(
            checkpoint="/home/yansu/mmlabmat/segmantation/mmseg-extension-test/pretrained/flash_internimage/flash_intern_image_b_1k_224.pth",
            # 'https://huggingface.co/OpenGVLab/DCNv4/resolve/main/flash_intern_image_b_1k_224.pth',
            type='Pretrained'),
        layer_scale=1.0,
        mlp_ratio=4.0,
        norm_layer='LN',
        offset_scale=0.5,
        out_indices=(0, 1, 2, 3,),
        post_norm=True,
        with_cp=False),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        in_channels=[112, 224, 448, 896, ],
        in_index=[0, 1, 2, 3, ],
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        pool_scales=(1, 2, 3, 6,),
        type='UPerHead'),
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        in_channels=448,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=num_classes,
        num_convs=1,
        type='FCNHead'),
    train_cfg=dict(),
    # test_cfg=dict(mode='whole'),
    # test_cfg=dict(crop_size=(3072, 3072,), mode='slide', stride=(2560, 2560,)),
    test_cfg=dict(crop_size=(3200, 3200,), mode='slide', stride=(2944, 2944,)),

)

# By default, models are trained on 2 GPUs with 4 images per GPU
# train_dataloader = dict(num_batch_per_epoch=5)
# val_dataloader = dict(num_batch_per_epoch=5)
# test_dataloader = val_dataloader

visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend', save_dir='./runsbatch/graphene_batch1_FlashInternImage_b/local'),
        # dict(type='TensorboardVisBackend', save_dir='runs/tb'),
        dict(type='MLflowVisBackend', save_dir='./runsbatch/graphene_batch1_FlashInternImage_b/mlruns',
             exp_name='batch1',
             run_name='FlashInternImage_b',
             ),
    ])
# visualizer = dict(
#     vis_backends=[
#         dict(type='LocalVisBackend', save_dir='tmp/local'),
#         dict(type='TensorboardVisBackend', save_dir='tmp/tb'),
#         dict(type='MLflowVisBackend', save_dir='tmp/mlruns',
#              exp_name='upernet_internimage_b_512_80k_b2d2_2dmat.py', ),
#     ])
