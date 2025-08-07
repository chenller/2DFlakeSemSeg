from __future__ import annotations

from pprint import pprint

import numpy as np
from mmpretrain.datasets import CIFAR10
from torch.utils.data import Dataset
from mmengine.registry import DATASETS

from mmengine.logging import print_log
import logging
from mmengine.dataset import BaseDataset as _mmengine_BaseDataset
from mmengine.dataset import Compose as PipelineCompose

LABELED = True
UNLABELED = False


@DATASETS.register_module()
class ALBaseDataset(Dataset):
    mode = LABELED
    labeled_index_list: list = []
    unlabeled_index_list: list = []
    _state_dict: dict = {'data_list': [], 'labeled_index_list': [], 'unlabeled_index_list': []}

    def __init__(self, train_dataset_cfg: dict, val_pipeline: list):
        super().__init__()
        dataset_cfg = self.__check_serialize_data(train_dataset_cfg)
        self.dataset: _mmengine_BaseDataset = DATASETS.build(dataset_cfg)
        # self.dataset.data_list = np.random.choice(self.dataset.data_list, 2000, False)
        # self.dataset.data_list = sorted(self.dataset.data_list,
        #                                 key=lambda x: (x['img'][0, 0, 0], x['img'][1, 0, 0], x['img'][2, 0, 0]))
        self._state_dict['data_list'] = self.dataset.data_list
        self.METAINFO = self.dataset.METAINFO
        self.metainfo=self.METAINFO
        scope = train_dataset_cfg['_scope_']
        for i in range(len(val_pipeline)):
            val_pipeline[i]['_scope_'] = scope
        self.query_pipeline = PipelineCompose(val_pipeline)

        self.labeled_idxs = np.zeros(len(self.dataset.data_list), dtype=bool)
        # self.initialize_labels(percentage=0.05)

    def state_dict(self) -> dict:
        return self._state_dict

    def load_state_dict(self, state_dict: str | dict, strict: bool = True, assign: bool = False):
        if isinstance(state_dict, dict):
            if 'data_list' in state_dict.keys():
                self.dataset.data_list = state_dict['data_list']
                self.labeled_index_list = state_dict['labeled_index_list']
                self.unlabeled_index_list = state_dict['unlabeled_index_list']
                self.labeled_idxs[:] = False
                self.labeled_idxs[self.labeled_index_list] = True
                self.switch_labeled_mode()
            else:
                assert False

    def update(self, sample_idx_list: list[int] = None, unlabeled_idx_list: list[int] = None):
        if sample_idx_list is not None:
            self.labeled_idxs[sample_idx_list] = True
        elif unlabeled_idx_list is not None:
            sample_idx_list = [self.unlabeled_index_list[i] for i in unlabeled_idx_list]
            self.labeled_idxs[sample_idx_list] = True
        else:
            assert False, f'No sample_idx_list or unlabeled_idx_list, got sample_idx_list:{sample_idx_list} and unlabeled_idx_list:{unlabeled_idx_list}'
        self.update_labeled_unlabeled_list()

    @property
    def length_all(self) -> int:
        return len(self.dataset.data_list)

    @property
    def length_labeled(self) -> int:
        return len(self.labeled_index_list)

    @property
    def length_unlabeled(self) -> int:
        return len(self.unlabeled_index_list)

    @property
    def percentage_labeled(self) -> float:
        """ 0 - 100 """
        return len(self.labeled_index_list) / self.length_all

    def update_labeled_unlabeled_list(self):
        self.labeled_index_list = np.where(self.labeled_idxs == True)[0].tolist()
        self.unlabeled_index_list = np.where(self.labeled_idxs == False)[0].tolist()
        self._state_dict['labeled_index_list'].append(self.labeled_index_list)
        self._state_dict['unlabeled_index_list'].append(self.unlabeled_index_list)

    def initialize_labeled(self, size: int | float = None, initialize_labeled_id: list[str] = None):
        if initialize_labeled_id is not None:
            idx = []
            for i, infos in enumerate(self.dataset.data_list):
                img_path: str = infos['img_path']
                if img_path.split('/')[-1] in initialize_labeled_id:
                    idx.append(i)
        else:
            length = len(self.labeled_idxs)
            if size <= 1.0:
                size = int(length * size)
            assert size <= length
            idx = np.random.choice(range(length), size, replace=False)
        assert len(idx) != 0
        self.labeled_idxs[idx] = True
        self.update_labeled_unlabeled_list()

    def switch_labeled_mode(self):
        self.mode = LABELED

    def switch_unlabeled_mode(self):
        self.mode = UNLABELED

    @staticmethod
    def __check_serialize_data(dataset_cfg: dict):
        dataset_cfg['serialize_data'] = False
        return dataset_cfg

    def __query_getitem__(self, idx: int) -> dict:
        if not self.dataset._fully_initialized:
            print_log('Please call `full_init()` method manually to accelerate the speed.', logger='current',
                      level=logging.WARNING)
            self.dataset.full_init()

        if self.dataset.test_mode:
            data = self.__query_prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.dataset.max_refetch + 1):
            data = self.__query_prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self.dataset._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.dataset.max_refetch}! '
                        'Please check your image path and pipeline')

    def __query_prepare_data(self, idx):
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.dataset.get_data_info(idx)
        return self.query_pipeline(data_info)

    def __getitem__(self, item: int) -> dict:
        if self.mode == LABELED:
            idx = self.labeled_index_list[item]
            return self.dataset.__getitem__(idx)
        else:
            idx = self.unlabeled_index_list[item]
            return self.__query_getitem__(idx)

    def __len__(self) -> int:
        # return 10
        if self.mode == LABELED:
            return len(self.labeled_index_list)
        else:
            return len(self.unlabeled_index_list)


if __name__ == '__main__':
    CFG=1
    dataset = ALBaseDataset(dataset_cfg=CFG)
    # print(dataset[0])
    # print(dataset[1])
    dataset.switch_labeled_mode()
    print(dataset[0])
    print(len(dataset))
    dataset.switch_unlabeled_mode()
    print(dataset[0])
    print(len(dataset))

    dataset.switch_labeled_mode()
