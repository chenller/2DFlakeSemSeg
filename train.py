# Copyright (c) OpenMMLab. All rights reserved.


import argparse
import logging
import os
import os.path as osp
import sys
from pprint import pprint
import copy

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

import mmsegext
import mmseg2dmat
from mmsegext.evaluation.metrics.region_metrics import RegionIoU

debug_cfg = dict(
    visualizer=dict(vis_backends=[
        dict(type='LocalVisBackend', save_dir='runs_temp/local/temp'),
        dict(type='MLflowVisBackend', save_dir='runs_temp/mlruns', exp_name='exp_name', run_name='run_name', ), ]),
    work_dir='runs_temp/work_dir',
    train_dataloader=dict(num_batch_per_epoch=11),
    # val_dataloader=dict(num_batch_per_epoch=11),
    train_cfg=dict(type='IterBasedTrainLoop', max_iters=40, val_interval=1, val_begin=0)
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--debug', type=bool, default=False, help='debug')
    parser.add_argument('--config', help='train config file path')
    parser.add_argument('--config-merge', help='merge config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    print('=' * 20, ' Argument ', '>' * 20)
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    print('<' * 20, ' Argument ', '=' * 20)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    if args.config_merge is not None:
        cfg_merge = Config.fromfile(args.config_merge)
        cfg.merge_from_dict(cfg_merge.to_dict())

    if args.cfg_options is not None:
        cfg_options_copy = copy.deepcopy(args.cfg_options)
        for k, v in args.cfg_options.items():
            if 'cfg.' in k:
                cfg_options_copy.pop(k)
                print(type(k), type(v))
                if isinstance(v, str):
                    exec(f"{k}='{v}'")
                    print(f"{k}='{v}'")
                else:
                    exec(f"{k}={v}")
                    print(f"{k}={v}")
        print(cfg_options_copy)
        cfg.merge_from_dict(cfg_options_copy)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'
    if args.debug:
        cfg.merge_from_dict(debug_cfg)

    # resume training
    cfg.resume = args.resume

    # pprint(cfg.to_dict())
    if args.cfg_options is not None:
        for k, v in args.cfg_options.items():
            if 'cfg.' in k:
                c = f'print(\'{k}\',\':\',type({k}),\'=\',{k})'
                exec(c)
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':

    pprint(sys.argv)
    if sys.argv.__len__() == 1:
        params = {
            # '--config': './config/upernet_flash_internimage_b_in1k_768.py',
            '--work-dir': './runsbatch_graphene/work_dirs/batch6/internimage_b',
            # '--config': './config/upernet_unireplknet_b_in22k_768.py',
            # '--work-dir': './runsbatch_graphene/work_dirs/batch6/unireplknet_b',
            # '--config': './config/upernet_biformer_b_in1k_768.py',
            # '--work-dir': './runsbatch_graphene/work_dirs/batch6/biformer_b',

            '--config': './config/unet_768.py',
            # '--config': './config/deeplabv3plus_r101_in1k_768.py',

            '--config-merge': './config/dataset/2024_annlab_graphene_batch1_768.py',

            # '--config': './config/upernet_vit_comer_b_in22k_640.py',
            # '--work-dir': './runsbatch_graphene/work_dirs/batch1/vit_comer_b',
            # '--config-merge': './config/dataset/2024_annlab_graphene_batch1_640.py',

            '--debug': 'True',

        }
        cfg_options = [
            'cfg.visualizer.vis_backends[0].save_dir="save_dir"',
            'cfg.visualizer.vis_backends[1].exp_name="exp_name"',
            'cfg.visualizer.vis_backends[1].run_name="run_name"',
            "cfg.param_scheduler[1].end=20000",
            "cfg.train_cfg.max_iters=20000",
            "cfg.train_cfg.val_interval=2000",
            "cfg.default_hooks.checkpoint.interval=2000",
        ]
        pprint(params)
        for k, v in params.items():
            sys.argv.append(k)
            sys.argv.append(v)
        for cfg in cfg_options:
            sys.argv.append('--cfg-options')
            sys.argv.append(cfg)
    main()
