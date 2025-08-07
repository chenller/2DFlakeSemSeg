from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import re


def find_max(filepath: str | Path):
    with open(str(filepath), 'r') as f:
        d = f.readlines()
    m = [float(i.split(' ')[1]) for i in d]
    return max(m)


def find_history(filepath: str | Path):
    df = pd.read_csv(filepath, sep='\s+', header=None, index_col=2)
    return df


def get_table(dirpath: str | Path, dirfliter: str = None, metric_fliter: list[str] = None) -> dict[str, pd.DataFrame]:
    subdirs = list(Path(dirpath).glob(dirfliter))
    if metric_fliter is None:
        metric_fliter = list(subdirs[0].glob('**/metrics*'))
    names = [x.name for x in subdirs]
    metric_table = {i: pd.DataFrame() for i in metric_fliter}
    for metric in metric_fliter:
        for subdir in subdirs:
            name = subdir.name
            index = name.split('_')[-1]
            columns = str(name.split('_')[-2][-1])
            mfp = list(subdir.glob(f'**/{metric}'))
            if len(mfp) == 0:
                continue
            elif len(mfp) >= 1:
                max_value = find_max(mfp[0])
                metric_table[metric].loc[index, columns] = max_value
    return {k: df.sort_index(axis=1) for k, df in metric_table.items()}


def get_metric_history(dirpath: str | Path, dirfliter: str = None, save_dir: str | Path = None) -> dict[
    str, pd.DataFrame]:
    metric_history_dict = dict()
    subdirs = list(Path(dirpath).glob(f'{dirfliter}*'))
    for subdir in subdirs:
        metric_paths = list(subdir.rglob('**/metrics'))
        if len(metric_paths) == 1:
            metric_paths = metric_paths[-1]
        metric_paths = list(metric_paths.rglob('**/*'))
        df_history = pd.DataFrame()
        for metric_path in metric_paths:
            if metric_path.is_dir():
                continue
            df = pd.read_csv(metric_path, sep='\s+', header=None, index_col=2)
            if df.shape[0] >= 100:
                continue
            metric_name = str(metric_path).split('/metrics/')[-1]
            df_history[metric_name] = df[1]
        metric_history_dict[str(subdir).split('/')[-1]] = df_history
    if save_dir is not None:
        for k, v in metric_history_dict.items():
            dir = Path(save_dir)
            dir.mkdir(parents=True, exist_ok=True)
            v.to_csv(dir / f'{k}.csv')
    return metric_history_dict


if __name__ == '__main__':

    metric_fliter = [
        'mIoU',
        'RIoU0.50',
        'RP0.50', 'RR0.50', 'region_num_match0.50',
        'RIoU0.75',
        'RP0.75', 'RR0.75', 'region_num_match0.75',
        'region_num_pred',
        'region_num_gt',
    ]

    # root = './runsbatch_graphene_paper'
    root = 'runsbatch5_reversion'
    # # get_run_time(root)
    raw_data = dict()
    metric_table = get_table(root, dirfliter='*batch5*', metric_fliter=metric_fliter)
    for metric, table in metric_table.items():
        print(metric)
        print(table.to_string())
        raw_data[f'graphene_{metric}'] = table.to_dict()
        # print(table.mean(axis=1))
        # print('\n')
        # print(table.style.format({'Value': '{:.2f}'}).to_latex().replace('0000 ', ''))

    # get_run_time('runsbatch_graphene_paper')
    # get_run_time('runsbatch_MoS2_paper')

    # # root = './runsbatch_MoS'
    # root = './runsbatch_MoS2_paper'
    # # get_run_time(root)
    # metric_table = get_table(root, dirfliter='*batch5*', metric_fliter=metric_fliter)
    # for metric, table in metric_table.items():
    #     print(metric)
    #     print(table.to_string())
    #     raw_data[f'MoS2_{metric}'] = table.to_dict()
    #     # print(table.mean(axis=1))
    #     # print('\n')
    #     # print(table.style.format({'Value': '{:.2f}'}).to_latex().replace('0000 ', ''))
    # json.dump(raw_data, open('raw_data_batch_5.json', 'w'))
