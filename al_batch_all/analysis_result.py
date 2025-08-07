from __future__ import annotations

from pathlib import Path

import pandas as pd

import re


def find_max(filepath: str | Path):
    with open(str(filepath), 'r') as f:
        d = f.readlines()
    m = [float(i.split(' ')[1]) for i in d]
    return max(m)


def get_run_name(filepath: str | Path):
    # 找到mlruns字符串，获取mlruns之后的两级路径
    filepath = str(filepath)
    metrics_index = filepath.find('metrics')
    filepath = filepath[:metrics_index]
    filepath = Path(filepath) / 'meta.yaml'
    with open(str(filepath), 'r') as f:
        for line in f:
            if line.startswith('run_name: '):
                return float(line[11:-2])
    return -1


def get_table(dirpath: str | Path, dirfliter: str = None, metric_fliter: list[str] = None) -> dict[str, pd.DataFrame]:
    if dirfliter is None:
        subdirs = list(Path(dirpath).glob(f'*'))
    else:
        subdirs = list(Path(dirpath).glob(f'{dirfliter}*'))
    metric_table = {i: pd.DataFrame() for i in metric_fliter}
    for metric in metric_fliter:
        for subdir in subdirs:
            name = subdir.name
            mfp = list(subdir.glob(f'**/{metric}'))
            index = [get_run_name(i) for i in mfp]
            columns = name
            if len(mfp) == 0:
                continue
            elif len(mfp) >= 1:
                for idx, fp in zip(index, mfp):
                    max_value = find_max(fp)
                    metric_table[metric].loc[idx, name] = max_value
    return {k: df.sort_index(axis=1).sort_index(axis=0) for k, df in metric_table.items()}


def save_table(metric_table: dict[str, pd.DataFrame], filepath: str | Path = './metrics.csv'):
    out_str = ''
    for metric, table in metric_table.items():
        csv = table.to_csv()
        out_str += f'{metric}{csv}\n\n'
    with open(str(filepath), 'w') as f:
        f.write(out_str)


def get_run_time(dirpath: str | Path, dirfliter: str = None, ):
    meta = list(Path(dirpath).glob(f'**/meta.yaml'))
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
    from plot import plot

    root = './run_s3_e30_n10_paper_graphene'
    # root = './run_s3_e30_n10_MoS2'

    metric_fliter = [
        'mIoU',
        'RIoU0.50',
        'RP0.50', 'RR0.50', 'region_num_match0.50',
        'RIoU0.75', 'RP0.75', 'RR0.75', 'region_num_match0.75',
        'region_num_pred',
        'RIoU0.50', 'RIoU0.75',
        'RP0.50', 'RR0.50',
        'RP0.75', 'RR0.75',
        'region_num_match0.50', 'region_num_match0.75',
        'region_num_pred',
    ]
    metric_fliter=list(set(metric_fliter))
    # metric_fliter = None
    metric_table = get_table(root, metric_fliter=metric_fliter)
    save_table(metric_table)

    # plot(metric_table)
    for metric, table in metric_table.items():
        names = ['EntropySampling', 'MarginSampling', 'RandomSampling']
        for name in names:
            col = [i for i in table.columns if name in i]
            table[f'mean/{name}'] = table[col].mean(axis=1)
            # print(table[col].to_string())

    for metric, table in metric_table.items():
        table.to_csv(f'./img/AL_graphene_batch12345_{metric}.csv')
        # print(metric)
        # print(table.to_string())
        # print(table.iloc[1:].mean())
        # print(table.mean().to_string())

    root = './run_s3_e30_n10_MoS2'
    # root = './run_s3_e30_n10_MoS2'

    metric_table = get_table(root, metric_fliter=metric_fliter)
    save_table(metric_table)

    # plot(metric_table)
    for metric, table in metric_table.items():
        names = ['EntropySampling', 'MarginSampling', 'RandomSampling']
        for name in names:
            col = [i for i in table.columns if name in i]
            table[f'mean/{name}'] = table[col].mean(axis=1)
            # print(table[col].to_string())

    for metric, table in metric_table.items():
        table.to_csv(f'./img/AL_MoS2_batch12345_{metric}.csv')
        # print(metric)
        # print(table.to_string())
        # print(table.iloc[1:].mean())
        # print(table.mean().to_string())
