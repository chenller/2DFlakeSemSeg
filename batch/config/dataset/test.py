from mmengine.config import Config
from mmseg.registry import DATASETS
import mmsegext
import mmseg2dmat

for i in range(1,5):
    cfg = Config.fromfile(f'./2024_annlab_graphene_batch{i}_768x768.py')
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    print(len(dataset))
    d = dataset[0]
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    print(len(dataset))
    d = dataset[0]
# for i in range(10):
#     d = dataset[i]
#     inputs = d['inputs']
#     print(inputs.shape)
#     gt_sem_seg = d['data_samples'].gt_sem_seg.data
#     print(gt_sem_seg.min(), gt_sem_seg.max())
