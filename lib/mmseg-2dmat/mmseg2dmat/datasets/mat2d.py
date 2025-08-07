# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module(name='2dmat-Mat2dDataset')
class Mat2dDataset(BaseSegDataset):
    """ADE20K dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        # classes=('background', 'thickness1', 'thickness2', 'thickness3', 'thickness4', 'thickness5', 'thickness6'),
        # palette=[[255, 255, 255],
        #          [255, 50, 50], [50, 255, 50], [50, 50, 255],
        #          [255, 255, 50], [50, 255, 255], [255, 50, 255], ]
        classes=tuple(['background'] + [f'layer{i}' for i in range(1, 52)]),
        palette=[
            [0, 192, 64], [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64], [128, 192, 64],
            [0, 160, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192], [128, 128, 0], [128, 0, 32],
            [0, 160, 0], [0, 0, 0], [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
            [192, 128, 32], [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
            [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32], [128, 96, 0],
            [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0], [0, 128, 192],
            [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192], [0, 0, 32],
            [64, 160, 128], [128, 64, 64], [128, 0, 160],
        ],
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 classes_name: list[str] = None,
                 **kwargs) -> None:
        # print(len(self.METAINFO['palette']),len(self.METAINFO['classes']))
        if classes_name is not None:
            assert isinstance(classes_name,
                              list), f"chanller error : 'classes_name' type must be 'list', but got '{type(classes_name)}'"
            assert all([isinstance(i, str) for i in
                        classes_name]), f'chanller error : List elements must be str, but got {classes_name}'
            self.METAINFO['classes'] = tuple(classes_name)
            self.METAINFO['palette'] = self.METAINFO['palette'][:len(classes_name)]
            assert len(self.METAINFO['classes']) == len(self.METAINFO['palette'])
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
if __name__=='__main__':
    a=Mat2dDataset()