# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import pickle
import platform
import random
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import mlflow
import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.visualization import Visualizer
from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.checkpoint import (_load_checkpoint, _load_checkpoint_to_model,
                                        find_latest_checkpoint, save_checkpoint,
                                        weights_to_cpu)
from mmengine.runner.log_processor import LogProcessor
from mmengine.runner.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from mmengine.runner.priority import Priority, get_priority
from mmengine.runner.utils import _get_batch_size, set_random_seed

ConfigType = Union[Dict, Config, ConfigDict]
ParamSchedulerType = Union[List[_ParamScheduler], Dict[str,
List[_ParamScheduler]]]
OptimWrapperType = Union[OptimWrapper, OptimWrapperDict]

from .aldataset import ALBaseDataset
from mmengine.runner import Runner


class _SlicedDataset:

    def __init__(self, dataset, length) -> None:
        self._dataset = dataset
        self._length = length

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return self._length


class ALRunner(Runner):
    al_train_dataset: ALBaseDataset = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for k, v in self.visualizer._vis_backends.items():
        #     self.logger.info(f'{v._env_initialized=}')

    def get_dataloader(self) -> (dict, dict):
        return copy.deepcopy(self._train_dataloader), copy.deepcopy(self._val_dataloader)

    def set_train_dataset(self, dataset):
        self.al_train_dataset = dataset
        self.logger.info(f"AL --> Set dataset successfully !")

    def train_without_after_run(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()

        # try to enable activation_checkpointing feature
        modules = self.cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')

        model = self.train_loop.run()  # type: ignore
        return model

    def train_with_after_run(self):
        self.call_hook('after_run')

    def build_dataloader(self, dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks.


        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if dataset_cfg == self._val_dataloader['dataset']:
            if isinstance(dataset_cfg, dict):
                dataset = DATASETS.build(dataset_cfg)
                if hasattr(dataset, 'full_init'):
                    dataset.full_init()
            else:
                # fallback to raise error in dataloader
                # if `dataset_cfg` is not a valid type
                dataset = dataset_cfg
            self.logger.info('AL [ALRunner] --> val dataset')
        else:
            dataset = self.al_train_dataset
            self.logger.info('AL [ALRunner] --> train dataset')
        self.logger.info(f'AL [ALRunner] --> dataset length : {len(dataset)}')

        num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
        if num_batch_per_epoch is not None:
            world_size = get_world_size()
            num_samples = (
                    num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
                    world_size)
            dataset = _SlicedDataset(dataset, num_samples)
        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    'type of worker_init_fn should be string or callable '
                    f'object, but got {type(worker_init_fn_type)}')
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                              **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                            dict(type='pseudo_collate'))
        if isinstance(collate_fn_cfg, dict):
            collate_fn_type = collate_fn_cfg.pop('type')
            if isinstance(collate_fn_type, str):
                collate_fn = FUNCTIONS.get(collate_fn_type)
            else:
                collate_fn = collate_fn_type
            collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
        elif callable(collate_fn_cfg):
            collate_fn = collate_fn_cfg
        else:
            raise TypeError(
                'collate_fn should be a dict or callable object, but got '
                f'{collate_fn_cfg}')
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader

    def build_visualizer(
            self,
            visualizer: Optional[Union[Visualizer,
            Dict]] = None) -> Visualizer:
        """Build a global asscessable Visualizer.

        Args:
            visualizer (Visualizer or dict, optional): A Visualizer object
                or a dict to build Visualizer object. If ``visualizer`` is a
                Visualizer object, just returns itself. If not specified,
                default config will be used to build Visualizer object.
                Defaults to None.

        Returns:
            Visualizer: A Visualizer object build from ``visualizer``.
        """
        if visualizer is None:
            visualizer = dict(
                name=self._experiment_name,
                vis_backends=[dict(type='LocalVisBackend')],
                save_dir=self._log_dir)
            return Visualizer.get_instance(**visualizer)

        if isinstance(visualizer, Visualizer):
            return visualizer

        if isinstance(visualizer, dict):
            # ensure visualizer containing name key
            visualizer['name'] = self._experiment_name
            # visualizer.setdefault('name', self._experiment_name)
            visualizer.setdefault('save_dir', self._log_dir)
            mlflow.end_run()
            return VISUALIZERS.build(visualizer)
        else:
            raise TypeError(
                'visualizer should be Visualizer object, a dict or None, '
                f'but got {visualizer}')


if __name__ == '__main__':
    ...
    # Runner.from_cfg(1)
