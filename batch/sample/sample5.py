from __future__ import annotations

from pathlib import Path
from pprint import pprint

import torch
from mmengine import Config
from mmseg.datasets.basesegdataset import Compose as PipelineCompose
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
import mmsegext
import mmseg2dmat
from tqdm import tqdm

from mmsegalext.strategy import ALSTRATEGY, Strategy
from mmsegalext.alselect import ALSample


def get_image_path(root, root_rm):
    src = Path(root).glob('**/*.jpg')
    src = {i.name: i for i in src}
    src_rm = {}
    for _root_rm in root_rm:
        _src_rm = Path(_root_rm).glob('**/*.jpg')
        _src_rm = {i.name: i for i in _src_rm}
        src_rm.update(_src_rm)

    src_use = {k: v for k, v in src.items() if k not in src_rm}

    print(f'{len(src)=} {len(src_rm)=} {len(src_use)=}')
    return list(src_use.values())


if __name__ == '__main__':
    i = 5
    root = '/home/share/annlab_2dmat_2024/coco_old/MoS2/'
    root_rm = ['/home/share/annlab_2dmat_2024/batchs/MoS2/', ]
    image_path = get_image_path(root, root_rm)
    image_path = [str(p) for p in image_path]
    print(len(image_path))
    # last batch
    save_filepath = f'./sample_image/sample_image_filepath{i}.py'
    _cfg = Config.fromstring('', file_format='.py')
    _cfg.merge_from_dict(dict(sample_image_filepath=image_path))
    if not Path(save_filepath).parent.exists():
        Path(save_filepath).parent.mkdir(parents=True)
    _cfg.dump(save_filepath)
    # last batch ned

    # cfg_merge = dict(model=dict(test_cfg=dict(mode='slide', crop_size=(3072 + 128, 3072 + 128,),
    #                                           stride=(2560 + 128, 2560 + 128,))))
    # num = 990
    # # num = 10
    # # # image_path=image_path[:10]
    # find_root = f'/home/yansu/paper/batch/runsbatch/graphene_batch{i - 1}_FlashInternImage/work_dirs'
    # cfg_fp = str(list(Path(find_root).glob('*.py'))[0])
    # weight_fp = str(list(Path(find_root).glob('best_mIoU_iter_*.pth'))[0])
    #
    # print(cfg_fp)
    # print(weight_fp)
    #
    # # als = ALSample(cfg_fp=cfg_fp, weight_fp=weight_fp, strategy=dict(type='MarginSampling'), cfg_merge=cfg_merge)
    # als = ALSample(cfg_fp=cfg_fp, weight_fp=weight_fp, strategy=dict(type='MarginSampling'), )
    # als.sample(image_path, num=num, save_filepath=f'./sample_image/sample_image_filepath{i}.py')
    from dataset_split import split
    #
    split(i)
