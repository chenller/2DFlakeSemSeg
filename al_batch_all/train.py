import argparse
import os
import sys
from pprint import pprint

from mmengine import Config, DictAction
import torch

from mmsegalext.almanager import ALManager

import mmseg2dmat
import mmsegext

# 获取当前工作目录
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process command line arguments")
    # 添加字符串类型的参数
    parser.add_argument("--cfg", type=str, help="Root path for configuration", default='')
    parser.add_argument("--work_dir", type=str, help="Working directory", default='./runs')
    parser.add_argument("--exp_name", type=str, help="Experiment name", default='exp')
    parser.add_argument("--seed", type=int, help="Seed")
    # 添加枚举类型参数，用于指定不同的采样策略
    parser.add_argument('--strategy', type=str, default="RandomSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 ], help="query strategy")
    # 添加可选的字符串列表参数
    parser.add_argument("--cfg_merge", nargs='*',
                        help="List of additional configuration paths to merge", metavar="CFG_PATH")

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
    args = parser.parse_args()
    print("=" * 20 + '>' * 20)
    print(f"Config Root: {args.cfg}")
    print(f"Config Merge Roots: {args.cfg_merge}")
    print(f"Work Directory: {args.work_dir}")
    print(f"Sampling Strategy: {args.strategy}")
    print(f"Experiment Name: {args.exp_name}")
    print("<" * 20 + '=' * 20)
    return args


def main():
    args = parse_arguments()
    cfg = Config.fromfile(args.cfg)
    for i in args.cfg_merge:
        print(f'cfg_merge_root : {i}')
        cfg.merge_from_dict(Config.fromfile(i).to_dict())
    cfg.merge_from_dict(dict(work_dir=args.work_dir, exp_name=args.exp_name,
                             sampling_strategy=dict(type=args.strategy), ))
    if args.seed:
        cfg.merge_from_dict(dict(randomness=dict(seed=args.seed)))
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print("=" * 20 + '>' * 20)
    print(f'{args.cfg=}')
    print(f'{args.cfg_merge=}')
    print(f'{args.cfg_options=}')
    print(f'{cfg.work_dir=}')
    print(f'{cfg.exp_name=}')
    print(f'{cfg.sampling_strategy=}')
    print(f'{cfg.randomness=}')
    print(f'{cfg.resume_config=}')
    print("<" * 20 + '=' * 20)

    alm = ALManager.from_cfg(cfg)
    alm.train()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # cfg_root = './config/upernet_flash_internimage_b_in1k_768.py'
        # cfg_merge_root = ['./config/2024_annlab_graphene_batch1_768.py', './config/activelearning04_random.py']
        # work_dir = './runs_temp'

        cfg_root = './config/upernet_flash_internimage_b_in1k_768.py'
        cfg_merge_root = ['./config/2024_annlab_MoS2_batch5_768.py', './config/s3_e30_n10_random.py']
        work_dir = './runs_temp'

        params = {
            '--cfg': cfg_root,
            '--work_dir': work_dir,
            '--exp_name': 'exp',
            '--seed': '1234',
            '--cfg_merge': cfg_merge_root,
            '--strategy': 'RandomSampling',
            # '--cfg-options': ['resume_config.model_load_from=/abc',
            #                   'resume_config.manager_load_from=/bfdsbds']
            '--cfg-options': ["model.type='ext-EncoderDecoderCompile'"],
        }
        pprint(params)

        for k, v in params.items():
            sys.argv.append(k)
            if isinstance(v, list):
                sys.argv += v
            else:
                sys.argv.append(v)
        print(sys.argv)
    main()
