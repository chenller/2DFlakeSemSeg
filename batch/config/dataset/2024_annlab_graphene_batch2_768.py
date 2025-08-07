# dataset settings
num_batch = 2  # [1,2,3,4]
classes_name = ['background', 'Thin-Layer']
dataset_type = '2dmat-Mat2dDataset'

crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=None),
    # dict(type='MultiClassification2BinaryClassification'),
    dict(
        type='RandomResize',
        scale=(2560, 1536),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),

    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=[1 / 3, 1 / 3], direction=['horizontal', 'vertical']),
    dict(type='PhotoMetricDistortion',
         brightness_delta=16,
         contrast_range=(0.75, 1.25),
         saturation_range=(0.75, 1.25),
         hue_delta=9),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=None),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

dataset_train_list = []
dataset_val_list = []

for i in range(1, num_batch + 1):
    data_root = f'/home/share/annlab_2dmat_2024/batchs/graphene/{i}/'
    dataset_train_list.append(
        dict(type=dataset_type, data_root=data_root, pipeline=train_pipeline,
             data_prefix=dict(img_path='train2024', seg_map_path='annotations_semseg/train2024'),
             img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=None,
             classes_name=classes_name)
    )
    dataset_val_list.append(
        dict(type=dataset_type, data_root=data_root, pipeline=test_pipeline,
             data_prefix=dict(img_path='val2024', seg_map_path='annotations_semseg/val2024'),
             img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=None,
             classes_name=classes_name)
    )

train_dataset = dict(type='ConcatDataset', datasets=dataset_train_list, _scope_='mmseg')
val_dataset = dict(type='ConcatDataset', datasets=dataset_val_list, _scope_='mmseg')
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset,
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(type='ext-RegionIoU', thresholds=[0.5, 0.75], skip_class_num=[0], area_filter=100, ),
    # dict(type='ext-IoUDICMetric'),
    # dict(type='ext-ConfusionMatrixMetric'),
    dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore']),
]
test_evaluator = val_evaluator

del crop_size
del num_batch
del dataset_type
del classes_name
del dataset_train_list
del dataset_val_list
del train_pipeline
del test_pipeline
del train_dataset
del val_dataset
# from pprint import pprint
#
# pprint(train_dataloader)
# pprint(val_dataloader)
