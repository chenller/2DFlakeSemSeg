from __future__ import annotations
import os
import copy
import pickle
import time
import traceback
from collections import defaultdict, OrderedDict
from pprint import pprint
import random
from pathlib import Path
import numpy as np
from mmengine.runner import set_random_seed
from mmengine.visualization import Visualizer
from mmseg.registry import DATASETS
import mlflow
import mmengine
import torch
from mmengine.config import Config
from mmseg.visualization import SegLocalVisualizer

from .alrunner import ALRunner
from .aldataset import ALBaseDataset
from .alutils import MetricsExtractor
from .strategy import ALSTRATEGY, Strategy

from .toymodel import *


class ALManager:
    runner: ALRunner = None
    train_dataset: ALBaseDataset = None
    query_dataloader_config: dict = None
    strategy: Strategy = None
    work_dir: str | Path = None
    exp_name: str = None
    run_name: str = None
    stage: int = 0
    stage_scheduler: dict[str, list] = None
    stage_select_size: list[int] = None
    _metrics_history: dict[float, dict[str, float]] = dict()  # {stage_length:{metric_name:value}}

    def __init__(self, runner_cfg: Config,
                 init_selection_size: int | float = 0.05,
                 final_selection_size: int | float = 0.9,
                 num_iter: int = 2,
                 sampling_strategy: str = 'EntropySampling',
                 model_load_policy: str = 'latest',
                 exp_name: str = 'exp12',
                 stage_scheduler: dict = None,
                 stage: int = None,
                 initialize_labeled_id: list = None,
                 resume_config: dict = None,
                 debug: bool = False,
                 ):
        r"""

        :param runner_cfg:
        :param init_selection_size: The initial number of samples to be selected for labeling. If int, represents the exact number of samples; if float, represents the percentage of the dataset.
        :param final_selection_size: The number of samples to select at the end of the active learning process.
        :param num_iter: The number of iterations for the active learning loop.
        :param sampling_strategy: The strategy used to select samples for labeling.
        :param model_load_policy: The policy used to determine which model to load ('random', 'best', 'latest').
        :param exp_name: The name of the experiment.
        :param load_from: The path to load the model from.
        """

        self.runner_cfg = runner_cfg
        self.init_selection_size = init_selection_size
        self.final_selection_size = final_selection_size
        self.num_iter = num_iter
        self.sampling_strategy = sampling_strategy
        self.model_load_policy = model_load_policy
        self.exp_name = exp_name
        self.stage = stage
        self.stage_scheduler = stage_scheduler
        self.initialize_labeled_id = initialize_labeled_id
        self.resume = resume_config
        self.debug = debug

        assert model_load_policy in (None, 'best', 'latest')
        if resume_config is not None:
            if 'find_from' not in resume_config.keys():
                assert 'model_load_from' in resume_config.keys() and 'manager_load_from' in resume_config.keys()
            else:
                self.log_info(f"resume_config contains find_from, will load from the best model and manager "
                              f"in the dir {resume_config['find_from']=}, 'model_load_from' and 'manager_load_from' will be ignored.")

        # init
        seed = self.runner_cfg['randomness']['seed']
        set_random_seed(seed)

        self.valid_metrics_name = self.runner_cfg['valid_metrics_name']
        self.benchmark_metrics = self.runner_cfg['benchmark_metrics']
        self.work_dir = self.runner_cfg['work_dir']
        if stage is None:
            self.stage = 0
        else:
            assert isinstance(self.stage, int)
            assert 0 <= stage <= self.num_iter, f"stage must be an integer between 0 and num_iter-1 [0,{self.num_iter - 1}], got stage : {self.stage}"
            self.stage = stage
        self.strategy = ALSTRATEGY.build(sampling_strategy)
        self.runner_cfg['experiment_name'] = self.exp_name

    @classmethod
    def from_cfg(cls, cfg: dict) -> 'Runner':
        r""" Build a runner from config.
        :param cfg: dict
        :return:
        """
        cfg = copy.deepcopy(cfg)
        alm = cls(runner_cfg=cfg, init_selection_size=cfg['init_selection_size'],
                  final_selection_size=cfg['final_selection_size'], num_iter=cfg['num_iter'],
                  sampling_strategy=cfg['sampling_strategy'], model_load_policy=cfg['model_load_policy'],
                  exp_name=cfg['exp_name'], stage=cfg['stage'], debug=cfg['debug'],
                  stage_scheduler=cfg['stage_scheduler'], initialize_labeled_id=cfg['initialize_labeled_id'],
                  resume_config=cfg['resume_config'],
                  )
        return alm

    def summarize_experiments(self):
        try:
            def log_metrics(work_dir, fliter=None):
                directory_to_search = Path(work_dir)
                extractor = MetricsExtractor(directory_to_search)
                log_files = extractor.get_log_files()
                name_epoch_metric = extractor.parse_logs(log_files)
                metrics_history = extractor.compile_metrics(name_epoch_metric)
                max_metrics = extractor.calculate_max_metrics(fliter=fliter)
                return name_epoch_metric, metrics_history, max_metrics

            dirs = list(Path(self.runner_cfg['work_dir']).parent.parent.parent.glob('*'))
            for dir in dirs:
                if dir.is_dir():
                    name_epoch_metric, metrics_history, max_metrics = log_metrics(dir, fliter=self.valid_metrics_name)
                    for metric_name, run_metric in max_metrics.items():
                        _run_metric = run_metric.to_dict()
                        for run_name, value in _run_metric.items():
                            mlflow.log_metric(f"max/{metric_name}-{dir.name}", value,
                                              step=round(float(run_name) * 10000))
                    self.log_info(f'{name_epoch_metric=}')
                    self.log_info(f'{max_metrics=}')
        except Exception as e:
            self.log_info('\n' + '>>>' * 20)
            self.log_info(f'{traceback.format_exc()}')
            self.log_info('\n' + '<<<' * 20)

    def train(self):
        self.init_train_dataset()
        self.init_visualizer(run_name=f'{self.train_dataset.percentage_labeled:.6f}')

        while self.stage <= self.num_iter:  # 0, 1, 2, ..., self.num_iter
            if self.stage == 0:
                self.scheduler_for_train(self.stage)

                self.loop_one()

                self.save_state()
            else:
                if self.resume is not None:
                    self.load_and_resume()
                    self.resume = None
                else:
                    self.load_best_weight()
                self.query(self.stage_select_size[self.stage])
                self.init_visualizer(run_name=f'{self.train_dataset.percentage_labeled:.6f}')
                self.scheduler_for_train(self.stage)
                self.loop_one()

                self.save_state()
            self.stage += 1

    def init_train_dataset(self):
        if self.debug:
            self.runner_cfg.update(dict(train_dataloader=dict(num_batch_per_epoch=5),
                                        val_dataloader=dict(num_batch_per_epoch=5)))
        # 初始化数据集config
        self.query_dataloader_config = copy.deepcopy(self.runner_cfg['val_dataloader'])
        self.query_dataloader_config['dataset'] = copy.deepcopy(self.runner_cfg['train_dataloader']['dataset'])
        self.train_dataset = ALBaseDataset(self.runner_cfg['train_dataloader']['dataset'],
                                           self.runner_cfg['val_dataloader']['dataset']['pipeline'])
        # 初始化第一批数据集，用户选择或者随机选择
        self.train_dataset.initialize_labeled(size=self.init_selection_size,
                                              initialize_labeled_id=self.initialize_labeled_id)
        self.log_info(f'{self.initialize_labeled_id=}')
        # 计算每个stage的选择样本数，结果为list[int]
        self.stage_select_size = []
        if self.num_iter!=0:
            stage_select_size_start = self.train_dataset.length_labeled
            stage_select_size_end = self.final_selection_size if self.final_selection_size > 1.0 else \
                round(self.train_dataset.length_all * self.final_selection_size)
            base, extra = divmod(stage_select_size_end - stage_select_size_start, self.num_iter)
            self.stage_select_size = [stage_select_size_start] + sorted([base + (i < extra) for i in range(self.num_iter)])
            assert sum(self.stage_select_size) == stage_select_size_end
            self.log_info(f'{self.stage_select_size=}, {stage_select_size_end=}')
        else:
            self.stage_select_size = [self.train_dataset.length_labeled]
            self.log_info(f'{self.stage_select_size=}')


    def scheduler_for_train(self, stage):
        if self.stage_scheduler is not None:
            d = dict()
            for k, v in self.stage_scheduler.items():
                d[k] = v[stage]
            self.runner_cfg.merge_from_dict(d)
            self.log_info(f"stage_scheduler : {d}")

    def query(self, size: int | float):
        if size <= 1:  # isinstance(num, float)
            size = int(size * self.train_dataset.length_all)
        if 'RandomSampling' in str(self.strategy):
            result_list = list(range(self.train_dataset.length_unlabeled))
        else:
            result_list = self.predict()
        sample_idx, unlabeled_idx = self.strategy.sample(result_list, size)
        self.train_dataset.update(sample_idx_list=sample_idx, unlabeled_idx_list=unlabeled_idx)

    def loop_one(self):
        self.runner = runner = ALRunner.from_cfg(self.runner_cfg)
        runner.set_train_dataset(self.train_dataset)
        # runner.train()
        runner.train_without_after_run()
        # self.summarize_experiments()
        runner.train_with_after_run()

    def predict(self):
        self.train_dataset.switch_unlabeled_mode()
        query_dataloader = self.runner.build_dataloader(self.query_dataloader_config)

        model = self.runner.model
        model.eval()
        result_list = []
        for idx, data_batch in enumerate(query_dataloader):
            with torch.no_grad():
                pred = model.val_step(data_batch)
                assessment_value = self.strategy.assessment_value(pred)
            result_list += assessment_value
            if (idx + 1) % 50 == 0:
                self.log_info(f'Iter(predict) [{idx}]')
            if self.debug and (idx > 5):
                break
            # print(len(pred))
        self.train_dataset.switch_labeled_mode()
        return result_list

    def init_visualizer(self, run_name='run',
                        LocalVisBackend: bool = True,
                        MLflowVisBackend: bool = True,
                        TensorboardVisBackend: bool = False):
        work_dir = self.work_dir
        exp_name = self.exp_name
        assert '/' not in exp_name
        assert '/' not in run_name
        self.run_name = str(run_name)

        vis_backends = []
        if LocalVisBackend:
            vis_backends.append(dict(type='LocalVisBackend',
                                     save_dir=str(Path(work_dir) / exp_name / run_name / 'local')))
        if MLflowVisBackend:
            vis_backends.append(dict(type='MLflowVisBackend', save_dir=str(Path(work_dir) / exp_name / 'mlruns'),
                                     exp_name=exp_name, run_name=run_name, ))
        if TensorboardVisBackend:
            vis_backends.append(dict(type='TensorboardVisBackend',
                                     save_dir=str(Path(work_dir) / exp_name / run_name / 'tb')))
        vis_cfg = dict(visualizer=dict(vis_backends=vis_backends),
                       work_dir=str(Path(work_dir) / exp_name / run_name / 'work_dir'))
        self.runner_cfg.merge_from_dict(vis_cfg)

    def load_best_weight(self):
        if self.runner is not None:
            for instance in self.runner.hooks:
                if 'CheckpointHook' == instance.__class__.__name__:
                    best_ckpt_path = instance.best_ckpt_path
                    p = torch.load(best_ckpt_path)
                    self.runner.model.load_state_dict(p['state_dict'], strict=True)
                    self.log_info(f'query load from {best_ckpt_path=}')
        else:
            self.log_info(f'query load from None')

    def log_info(self, msg: str):
        if self.runner is not None:
            self.runner.logger.info(f"AL --> [{self.stage}/{self.num_iter}] {msg}")
        else:
            print(f"AL --> [{self.stage}/{self.num_iter}] {msg}")

    def state_dict(self) -> dict:
        d = {'stage': self.stage, 'dataset': self.train_dataset.state_dict(),
             'batch_selection_size_list_by_stage': self.stage_select_size}
        return d

    def save_state(self):
        sd = self.state_dict()
        path = Path(self.runner_cfg['work_dir']) / f'alstate_{self.stage}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(sd, f)
        self.log_info(f'Checkpoints will be saved to {str(path)}')

    def load_and_resume(self):
        def wait_file_exit(dirpath: str | Path, find_str: str):
            dirpath = Path(dirpath)
            while True:  # 不存在'alstate_*.pkl'
                fps = list(Path(dirpath).glob(f'**/{find_str}'))
                if len(fps) == 0:
                    print(f"list(Path({str(dirpath)}).glob(f'**/{find_str}'))")
                    print(f"File '{find_str}' does not exist, wait for 60 seconds to check again. "
                          f"The search path is '{str(dirpath)}'.")
                    time.sleep(60)
                else:
                    print(f"File '{find_str}' does exist. The search path is {str(dirpath)}.")
                    print(list(Path(dirpath).glob(f'**/{find_str}')))
                    break

        if 'find_from' not in self.resume.keys():
            manager_load_from = self.resume['manager_load_from']
            model_load_from = self.resume['model_load_from']
        else:
            wait_file_exit(self.resume['find_from'], 'alstate_*.pkl')
            manager_load_from = str(list(Path(self.resume['find_from']).glob('**/alstate_*.pkl'))[-1])
            model_load_from = str(list(Path(self.resume['find_from']).glob('**/best_*.pth'))[-1])

        assert Path(manager_load_from).exists()
        with open(manager_load_from, 'rb') as f:
            s = pickle.load(f)
            stage = self.stage - 1
            s['dataset']['labeled_index_list'] = list(s['dataset']['labeled_index_list'][stage])
            s['dataset']['unlabeled_index_list'] = list(s['dataset']['unlabeled_index_list'][stage])
            self.train_dataset.load_state_dict(s['dataset'])
            self.stage_select_size = s['batch_selection_size_list_by_stage']
        self.runner_cfg.load_from = model_load_from
        self.runner = ALRunner.from_cfg(self.runner_cfg)
        self.runner.set_train_dataset(self.train_dataset)
        # 加载模型权重
        assert model_load_from is not None
        p = torch.load(model_load_from)
        model_state_dict = p['state_dict']
        self.runner.model.load_state_dict(model_state_dict, strict=True)
        self.log_info(f'resume from : {self.resume}')
        self.log_info(f'{manager_load_from=}')
        self.log_info(f'{model_load_from=}')
        self.log_info(f'{self.runner._has_loaded=}')
        self.resume = None


if __name__ == '__main__':
    cfg_root = './config/upernet_flash_internimage_b_in1k_768.py'
    cfg_merge_root = ['./config/2024_annlab_graphene_batch4_768.py', './config/activelearning.py']
    cfg = Config.fromfile(cfg_root)
    for i in cfg_merge_root:
        print(f'cfg_merge_root : {i}')
        cfg.merge_from_dict(Config.fromfile(i).to_dict())
    cfg.work_dir = './runs'
    alm = ALManager.from_cfg(cfg)
    alm.train()
