optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=None,
)
param_scheduler = dict(
    type='PolyLR',
    eta_min=1e-7,
    power=1.0,
    begin=0,
    end=10,
    by_epoch=True)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_begin=0, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=1, max_keep_ckpts=1, save_best=['mIoU', ], rule=['greater']),
)

# sampling_strategy = dict(type='EntropySampling')  # The strategy used to select samples for labeling.
# exp_name = 'upernet_flash_internimage_EntropySampling'
# sampling_strategy = dict(type='MarginSampling')  # The strategy used to select samples for labeling.
# exp_name = 'upernet_flash_internimage_MarginSampling'
# sampling_strategy = dict(type='LeastConfidence')  # The strategy used to select samples for labeling.
# exp_name = 'upernet_flash_internimage_LeastConfidence'
sampling_strategy = dict(type='RandomSampling')
exp_name = 'upernet_flash_internimage_RandomSampling'

model_load_policy = None  # The policy used to determine which model to load (None, 'best', 'latest').

valid_metrics_name = None
# valid_metrics_name = ['accuracy/top1', 'mIoU', 'mIoUC', 'mIoUI', 'mIoUCQ', 'mIoUC1']

stage = 1  # resume, 从第几阶段开始训练，一共有num_iter阶段(初始化和num_iter次迭代)，范围[0, num_iter]
# resume_config = None  # dict(stage=0, model_load_from='', manager_load_from='')
resume_config = dict(
    find_from='/home/yansu/mmlabmat/AL/batchall/run_s3_e30_n10/RandomSampling1/0.029821/work_dir',
    # model_load_from='/home/yansu/AL/seg1/run_multi/RandomSampling2/0.049305/work_dir/best_mIoU_epoch_900.pth',
    # manager_load_from='/home/yansu/AL/seg1/run_multi/RandomSampling2/0.049305/work_dir/alstate_0.pkl'
)

# load_from_al = '/home/yansu/AL/ALMM-seg/runs/upernet_flash_internimage_EntropySampling_old1/0.199848/work_dir/alstate_0.pkl'
# load_from_al_model = '/home/yansu/AL/ALMM-seg/runs/upernet_flash_internimage_EntropySampling_old1/0.199848/work_dir/epoch_80.pth'
initialize_labeled_id = None

benchmark_metrics = None
# benchmark_metrics = {
#     'exp1': {'0.1': {'mIoU': 55},
#              '0.2': {'mIoU': 76},
#              '0.5': {'mIoU': 35},
#              },
# }

debug = False
# The initial number of samples to be selected for labeling.
# If int, represents the exact number of samples; if float, represents the percentage of the dataset.
init_selection_size = 0.03  # The number of samples to select at the start of the active learning process.
final_selection_size = 0.30  # The number of samples to select at the end of the active learning process.

num_iter = 9  # The number of iterations for the active learning loop.

# stage_scheduler = None
# _stage_epoch = [330, 175, 120, 95, 80, 70, 60, 55, 50, 50]  # graphene
_stage_epoch = [732, 378, 264, 204, 168, 144, 126, 114, 102, 96]  # MoS2
_num = 6
# _stage_epoch = [12, 10, 8, 6, 4]
# _num = 1
_stage_lr = [0.01] * 10
stage_scheduler = dict(
    train_cfg=[dict(max_epochs=i, val_interval=i // _num) for i in _stage_epoch],
    param_scheduler=[dict(end=i) for i in _stage_epoch],
    default_hooks=[dict(checkpoint=dict(interval=i // _num))
                   for i in _stage_epoch],
    optim_wrapper=[dict(optimizer=dict(weight_decay=i)) for i in _stage_lr],
)

train_dataloader = dict(
    dataset=dict(_scope_='mmseg')
)
# val_dataloader = dict(num_batch_per_epoch=5)
# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='ext-ToyBackbone',
#         in_dim=3),
# )

# import numpy as np
# num = np.array([500, 250, 160, 120, 100, 80, 70, 60, 60, 50])
# ratio=np.array(list(range(1,11)))/10
# iter=num*ratio
# print(iter)

# import numpy as np
#
# ratio = np.array(list(range(5, 41, 5))) / 100
# iter = 50
# num = iter / ratio
# print(num)
# num = [1000, 500, 330, 250, 200, 160, 140, 120]

# ratio = np.array([0.2, 0.4, 0.6, 0.8])
# 60/(np.array(list(range(1,11)))/10)
# [600, 300, 200, 150, 120, 100, 80, 70, 60, 50]

# bz=4
# bz_new=2
# epoch=bz*iter/num/bz_new

# import numpy as np
# num = np.array([791 * 1, 791 * 2, 791 * 3, 791 * 5, ])

# initialize_labeled_id = [
#     '002575.jpg', '002811.jpg', '002954.jpg', '003143.jpg', '003173.jpg', '003194.jpg', '003659.jpg', '003965.jpg',
#     '004025.jpg', '004045.jpg', '004203.jpg', '004243.jpg', '004266.jpg', '004272.jpg', '004284.jpg', '004346.jpg',
#     '004382.jpg', '004421.jpg', '004657.jpg', '005020.jpg', '005061.jpg', '005196.jpg', '005240.jpg', '005799.jpg',
#     '005948.jpg', '006013.jpg', '006043.jpg', '006128.jpg', '006216.jpg', '006410.jpg', '006561.jpg', '006700.jpg',
#     '006818.jpg', '006842.jpg', '007075.jpg', '007083.jpg', '007145.jpg', '007290.jpg', '007977.jpg', '008229.jpg']

# import random
#
# random.shuffle(initialize_labeled_id)
# random.shuffle(initialize_labeled_id)
# random.shuffle(initialize_labeled_id)
# initialize_labeled_id = initialize_labeled_id[:40]
# initialize_labeled_id.sort()
# print(len(initialize_labeled_id))
# print(initialize_labeled_id)
