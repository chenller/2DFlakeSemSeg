from __future__ import annotations

from pathlib import Path

import pandas as pd

import re

import json
def find_max(filepath: str | Path):
    with open(str(filepath), 'r') as f:
        d = f.readlines()
    m = [float(i.split(' ')[1]) for i in d]
    return max(m)


def get_table(dirpath: str | Path, dirfliter: str = None, metric_fliter: list[str] = None) -> dict[str, pd.DataFrame]:
    subdirs = list(Path(dirpath).glob(f'{dirfliter}*'))
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


def get_run_time(dirpath: str | Path, dirfliter: str = None, ):
    meta = list(Path(dirpath).glob(f'**/meta.yaml'))
    meta.sort()
    if dirfliter is not None:
        meta = [i for i in meta if dirfliter in str(i)]
    all_time_hour = []
    for meta_fp in meta:
        with open(str(meta_fp), 'r') as f:
            d = f.read()
        match_st = re.search(r'start_time: *(\d+)\n', d)
        match_et = re.search(r'end_time: *(\d+)\n', d)
        if match_st and match_et:
            st = int(match_st.group(1))
            et = int(match_et.group(1))
            time_hour = (et - st) / 3600 / 1000
            all_time_hour.append(time_hour)
            name = str(meta_fp).split('/')[-5]
            print(f'{name} = {time_hour:.2f} h')
    print(f'all_time_hour = {sum(all_time_hour):.2f} h / 1x RTX 3090')


if __name__ == '__main__':
    root = './runsbatch_graphene_paper'
    # # get_run_time(root)
    raw_data = dict()
    metric_table = get_table(root, dirfliter='graphene', metric_fliter=[
        'mIoU',
        'RIoU0.50',
    ])
    for metric, table in metric_table.items():
        print(metric)
        print(table.to_string())
        raw_data[f'graphene_{metric}'] = table.to_dict()
        # print(table.mean(axis=1))
        # print('\n')
        # print(table.style.format({'Value': '{:.2f}'}).to_latex().replace('0000 ', ''))

    # get_run_time('runsbatch_graphene_paper')
    # get_run_time('runsbatch_MoS2_paper')

    # root = './runsbatch_MoS'
    root = './runsbatch_MoS2_paper'
    # get_run_time(root)
    metric_table = get_table(root, dirfliter='MoS2', metric_fliter=[
        'mIoU',
        'RIoU0.50'
    ])
    for metric, table in metric_table.items():
        print(metric)
        print(table.to_string())
        raw_data[f'MoS2_{metric}'] = table.to_dict()
        # print(table.mean(axis=1))
        # print('\n')
        # print(table.style.format({'Value': '{:.2f}'}).to_latex().replace('0000 ', ''))
    json.dump(raw_data, open('raw_data_batch_all.json', 'w'))
