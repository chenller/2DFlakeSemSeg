from __future__ import annotations

from pathlib import Path
from pprint import pprint

import torch
from mmengine import Config
from tqdm import tqdm


from mmseg.datasets.basesegdataset import Compose as PipelineCompose
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS

from .almanager import ALManager
from .alrunner import ALRunner
from .strategy import ALSTRATEGY, Strategy

class ALSample:
    pipeline: PipelineCompose
    model: BaseSegmentor
    strategy: Strategy

    def __init__(self, cfg_fp, weight_fp, strategy, cfg_merge: dict = None):
        self.cfg_fp = cfg_fp
        self.weight_fp = weight_fp
        self.strategy = strategy

        self.cfg = Config.fromfile(self.cfg_fp)
        if cfg_merge:
            self.cfg.merge_from_dict(cfg_merge)
        # self.init_pipeline()

    def init_strategy(self):
        if isinstance(self.strategy, dict):
            self.strategy = ALSTRATEGY.build(self.strategy)
        else:
            assert False
        return self.strategy

    def init_model(self):
        p = torch.load(self.weight_fp)
        state_dict = p['state_dict']
        model_cfg = self.cfg['model']
        pprint(model_cfg)
        self.model: 'BaseSegmentor' = MODELS.build(model_cfg)
        self.model.load_state_dict(state_dict=state_dict, strict=True)
        self.model.eval()
        return self.model

    def init_pipeline(self):
        dataset_cfg = self.cfg['val_dataloader']['dataset']
        if dataset_cfg['type'] == 'ConcatDataset':
            dataset_cfg = dataset_cfg['datasets'][0]
        pipeline_cfg = dataset_cfg['pipeline']
        pprint(pipeline_cfg)
        pipeline_cfg_use = []
        for cfg in pipeline_cfg:
            if 'LoadAnnotations' not in cfg['type']:
                cfg['_scope_'] = 'mmseg'
                pipeline_cfg_use.append(cfg)
        pipeline = PipelineCompose(pipeline_cfg_use)
        print(pipeline)
        self.pipeline = pipeline
        return self.pipeline

    def sample(self, img_filepath: list[str | Path], num: int = 1,
               save_filepath: str | Path = 'sample_image_filepath.py'):
        assert str(save_filepath).endswith('.py')
        pipeline = self.init_pipeline()
        strategy = self.init_strategy()
        model = self.init_model().cuda()

        pred_list = []
        for i, img_fp in tqdm(enumerate(img_filepath), total=len(img_filepath)):
            data_info = {'img_path': str(img_fp), }
            data = pipeline(data_info)
            inputs = data['inputs'].unsqueeze(0)
            data_samples = data['data_samples']
            data_samples.set_data(dict(sample_idx=i))

            with torch.no_grad():
                preds = model.val_step(data=dict(inputs=inputs.cuda(), data_samples=[data_samples]))
                assessment_value = strategy.assessment_value(preds)
            pred_list += assessment_value

            # if i>20:
            #     break

        sample_idx, unlabeled_idx = strategy.sample(pred_list, num)
        img_filepaths = [str(img_filepath[i]) for i in sample_idx]

        _cfg = Config.fromstring('', file_format='.py')
        _cfg.merge_from_dict(dict(sample_image_filepath=img_filepaths))
        if not Path(save_filepath).parent.exists():
            Path(save_filepath).parent.mkdir(parents=True)
        _cfg.dump(save_filepath)


if __name__ == '__main__':
    cfg_root = './config/upernet_flash_internimage_b_in1k_768.py'
    cfg = Config.fromfile(cfg_root)
    cfg.work_dir = './runs_select'

