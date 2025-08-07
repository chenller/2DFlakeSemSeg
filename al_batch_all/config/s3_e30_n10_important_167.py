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

stage = 0  # resume, 从第几阶段开始训练，一共有num_iter阶段(初始化和num_iter次迭代)，范围[0, num_iter]
resume_config = None  # dict(stage=0, model_load_from='', manager_load_from='')
# resume_config = dict(stage=0,
#                      model_load_from='/home/yansu/AL/seg/runs_state0/RandomSampling/0.099874/work_dir/best_mIoU_epoch_350.pth',
#                      manager_load_from='/home/yansu/AL/seg/runs_state0/RandomSampling/0.099874/work_dir/alstate_0.pkl')

# load_from_al = '/home/yansu/AL/ALMM-seg/runs/upernet_flash_internimage_EntropySampling_old1/0.199848/work_dir/alstate_0.pkl'
# load_from_al_model = '/home/yansu/AL/ALMM-seg/runs/upernet_flash_internimage_EntropySampling_old1/0.199848/work_dir/epoch_80.pth'
# initialize_labeled_id = None

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
final_selection_size = 0.06  # The number of samples to select at the end of the active learning process.

num_iter = 0  # The number of iterations for the active learning loop.

# stage_scheduler = None
# _stage_epoch = [500, 250, 160, 125, 100, 80, 70, 60, 55, 50]
# _stage_epoch = [260, 135, 90, 70, 60, 50, 45, 40, 35, 30]
_stage_epoch = [400]
_num = 10
# _stage_epoch = [12, 10, 8, 6, 4]
# _num = 2
_stage_lr = [0.01] * 1
stage_scheduler = dict(
    train_cfg=[dict(max_epochs=i, val_interval=i // _num) for i in _stage_epoch],
    param_scheduler=[dict(end=i) for i in _stage_epoch],
    default_hooks=[dict(checkpoint=dict(interval=i // _num))
                   for i in _stage_epoch],
    optim_wrapper=[dict(optimizer=dict(weight_decay=i)) for i in _stage_lr],
)

train_dataloader = dict(
    dataset=dict(_scope_='mmseg'),
    # persistent_workers=False,
    # num_batch_per_epoch=5
)

# val_dataloader = dict(
#     persistent_workers=False,
# )
# val_dataloader = dict(num_batch_per_epoch=5)

# model = dict(
#     backbone=dict(
#         _delete_=True,
#         type='ext-ToyBackbone',
#         in_dim=3),
# )

# import numpy as np
# bz=4
# bz_new=2
# iter= np.array([40000, 40000 ,60000, 70000, 80000 ])
# num=np.array([0.2,0.4,0.6,0.8,1])*3957
# epoch=iter*bz/num
# epoch_new=epoch/bz_new


# import numpy as np
#
# epoch = 10 / np.linspace(0.03, 0.3, 10)
# bl = np.linspace(1.0, 1.5, 10)
# epoch = epoch * bl
# epoch

initialize_labeled_id = [
    '003732.jpg', '006302.jpg', '003679.jpg', '006301.jpg', '006830.jpg', '005899.jpg', '003347.jpg', '005028.jpg',
    '006423.jpg', '006128.jpg', '005288.jpg', '005334.jpg', '006735.jpg', '006081.jpg', '007289.jpg', '002212.jpg',
    '002234.jpg', '003622.jpg', '006129.jpg', '003373.jpg', '003619.jpg', '003756.jpg', '006053.jpg', '006592.jpg',
    '007034.jpg', '003755.jpg', '007259.jpg', '008187.jpg', '003099.jpg', '003603.jpg', '004507.jpg', '004786.jpg',
    '005403.jpg', '003430.jpg', '005598.jpg', '003773.jpg', '003910.jpg', '006722.jpg', '002208.jpg', '002832.jpg',
    '005139.jpg', '005872.jpg', '006765.jpg', '002893.jpg', '003292.jpg', '004683.jpg', '004880.jpg', '005774.jpg',
    '006568.jpg', '002214.jpg', '005053.jpg', '006259.jpg', '006685.jpg', '003865.jpg', '005177.jpg', '006443.jpg',
    '003453.jpg', '003652.jpg', '005704.jpg', '008078.jpg', '004985.jpg', '002125.jpg', '002177.jpg', '002235.jpg',
    '003826.jpg', '005190.jpg', '006363.jpg', '007068.jpg', '003336.jpg', '005119.jpg', '005245.jpg', '006317.jpg',
    '005631.jpg', '006913.jpg', '005996.jpg', '003493.jpg', '003848.jpg', '005017.jpg', '004835.jpg', '005131.jpg',
    '005718.jpg', '005790.jpg', '005799.jpg', '005895.jpg', '006098.jpg', '006314.jpg', '006365.jpg', '008241.jpg',
    '003446.jpg', '006276.jpg', '006670.jpg', '007135.jpg', '007230.jpg', '002570.jpg', '003733.jpg', '007116.jpg',
    '003088.jpg', '003558.jpg', '003841.jpg', '004496.jpg', '004921.jpg', '005276.jpg', '006458.jpg', '007019.jpg',
    '008200.jpg', '005446.jpg', '007107.jpg', '008108.jpg', '002935.jpg', '002951.jpg', '005658.jpg', '003270.jpg',
    '004264.jpg', '005101.jpg', '006566.jpg', '007137.jpg', '006538.jpg', '004621.jpg', '003600.jpg', '004417.jpg',
    '004971.jpg', '005675.jpg', '006143.jpg', '003212.jpg', '004005.jpg', '005480.jpg', '006819.jpg', '006905.jpg',
    '007293.jpg', '008228.jpg', '003103.jpg', '003422.jpg', '003550.jpg', '004511.jpg', '006190.jpg', '006454.jpg',
    '006717.jpg', '006972.jpg', '007024.jpg', '003166.jpg', '003857.jpg', '004557.jpg', '005112.jpg', '006506.jpg',
    '008225.jpg', '003580.jpg', '004250.jpg', '005927.jpg', '003581.jpg', '005348.jpg', '005429.jpg', '006841.jpg',
    '007003.jpg', '003286.jpg', '003344.jpg', '004233.jpg', '005663.jpg', '005685.jpg', '006686.jpg', '003847.jpg',
    '006398.jpg', '002853.jpg', '004422.jpg', '005968.jpg', '006018.jpg', '006892.jpg', '006902.jpg']

# import random
#
# random.shuffle(initialize_labeled_id)
# random.shuffle(initialize_labeled_id)
# random.shuffle(initialize_labeled_id)
# initialize_labeled_id = initialize_labeled_id[:40]
# initialize_labeled_id.sort()
# print(len(initialize_labeled_id))
# print(initialize_labeled_id)
