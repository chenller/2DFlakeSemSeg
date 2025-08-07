import numpy as np
from mmcv.transforms.base import BaseTransform
from mmseg.registry import TRANSFORMS
import cv2


@TRANSFORMS.register_module()
class MultiClassification2BinaryClassification(BaseTransform):
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' in results.keys():
            gt_sem_seg = results['gt_seg_map']
            index = (gt_sem_seg != 0) & (gt_sem_seg != self.ignore_index)
            gt_sem_seg[index] = 1
            results['gt_seg_map'] = gt_sem_seg
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module(name='mmseg2dmat-RandomChoose')
class RandomChoose(BaseTransform):
    def __init__(self, transform_choose_list: list[float, BaseTransform] = []):
        super().__init__()
        self.transform_choose_list = []
        self.transform_p = []
        for p, t in transform_choose_list:
            assert p >= 0
            T = TRANSFORMS.build(t)
            self.transform_choose_list.append(T)
            self.transform_p.append(p)
        self.transform_p = np.array(self.transform_p) / np.sum(self.transform_p)
        self.transform_p = np.cumsum(self.transform_p)

    def transform(self, results: dict) -> dict:
        p = np.random.rand()
        T = None
        for i in range(len(self.transform_p)):
            if p < self.transform_p[i]:
                T = self.transform_choose_list[i]
                break
        assert T is not None
        results = T.transform(results)
        return results

    def __repr__(self):
        return 'mmseg2dmat-RandomChoose'


@TRANSFORMS.register_module(name='mmseg2dmat-ConverterChannel')
class ConverterChannel(BaseTransform):
    mode_list = ['hsv', 'lab']

    def __init__(self, mode: str = None):
        """

        :param mode: str 'hsv', 'HSV', 'lab', 'LAB'
        """
        super().__init__()
        self.mode = mode.lower()
        if self.mode is not None:
            assert isinstance(self.mode, str)
            assert self.mode in self.mode_list

    def transform(self, results: dict) -> dict:
        if 'img' in results:
            img = results['img']
            if self.mode == 'hsv':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.mode == 'lab':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            results['img'] = img
        return results

    def __repr__(self):
        return f'mmseg2dmat-ConverterChannel(mode = {self.mode})'


@TRANSFORMS.register_module(name='mmseg2dmat-AnnotationsOffset')
class AnnotationsOffset(BaseTransform):
    def __init__(self, offset: int, num_classes: int,ignore_index: int = 255):
        self.offset = offset
        self.num_classes = num_classes
        self.ignore_index=ignore_index

    def transform(self, results: dict) -> dict:
        if 'gt_seg_map' in results.keys():
            gt_seg_map=results['gt_seg_map']
            gt_seg_map[gt_seg_map!=self.ignore_index]+=self.offset
            results['gt_seg_map']=gt_seg_map
            # print(np.unique(gt_seg_map))
            results['annotations_offset']=self.offset
            results['annotations_num_classes'] = self.num_classes
        return results

    def __repr__(self):
        return f'mmseg2dmat-AnnotationsOffset(offset={self.offset}, num_classes={self.num_classes})'
