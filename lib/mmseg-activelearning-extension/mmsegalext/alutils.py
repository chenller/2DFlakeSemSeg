from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
from collections import defaultdict


def get_metrics_from_logfile(roots: str | list[str], fliter: list[str] = None, to_dict: bool = False) -> (
        dict[str, dict[str, dict | list]],
        dict[str, dict[str, dict[int, dict[str, float]]]],
        dict[str, pd.DataFrame]):
    """

    :param roots: list or str, e.g. work_dir1 or [work_dir1, work_dir2, ...]
    :param fliter:
    :param to_dict:
    :return:
    """
    strategy_dict = {}
    state_dict = defaultdict(lambda: dict())
    if isinstance(roots, list):
        dirs = [str(i) for root in roots for i in list(Path(root).glob('*'))]
    else:
        dirs = [str(i) for i in list(Path(roots).glob('*'))]
    for directory_to_search in dirs:
        extractor = MetricsExtractor(directory_to_search)
        log_files = extractor.get_log_files()
        name_epoch_metric = extractor.parse_logs(log_files, fliter=fliter)
        metrics_history = extractor.compile_metrics(name_epoch_metric)
        max_metrics = extractor.calculate_max_metrics(fliter, to_dict)  # 'mIoUC', 'mIoUCQ', 'mIoUC1'
        strategy_dict[str(directory_to_search).replace('\\', '/').split('/')[-1]] = name_epoch_metric
        state_dict[directory_to_search]['log_files'] = log_files
        state_dict[directory_to_search]['name_epoch_metric'] = name_epoch_metric
        state_dict[directory_to_search]['metrics_history'] = metrics_history
        state_dict[directory_to_search]['max_metrics'] = max_metrics
    state_dict = defaultdict_to_dict(state_dict)

    max_metrics_all = defaultdict(lambda: dict())
    for k, v in state_dict.items():  # k: str
        for m, v in v['max_metrics'].items():  # v: pd.Series
            max_metrics_all[m][k.split('/')[-1]] = v
    max_metrics_all_dataframe = {k: pd.DataFrame(v) for k, v in max_metrics_all.items()}

    return state_dict, strategy_dict, max_metrics_all_dataframe


def defaultdict_to_dict(d: defaultdict | dict | list) -> dict:
    if isinstance(d, defaultdict) or isinstance(d, dict):
        new_d = dict()
        for k, v in d.items():
            new_d[k] = defaultdict_to_dict(v)
        return new_d
    elif isinstance(d, list):
        L = [defaultdict_to_dict(v) for v in d]
        return L
    return d


class MetricsExtractor:
    def __init__(self, directory):
        """

        :param directory: work_dir
        """
        self.directory = directory
        self.metrics_history = None
        self.max_metrics = None

    def get_log_files(self):
        """从指定目录及其子目录中获取所有.log文件"""
        dir_path = Path(self.directory)
        if dir_path.is_dir():
            log_files = list(dir_path.glob('**/*.log'))
            log_files.sort()
            log_files = [
                [str(fp).replace('\\', '/').split('/')[-4], fp]
                for fp in log_files if 'work_dir' in str(fp)
            ]
            # 合并连续相同的名称
            unique_log_files = []
            for i, (name, fp) in enumerate(log_files[:-1]):
                if name != log_files[i + 1][0]:
                    unique_log_files.append([name, fp])
            unique_log_files.append(log_files[-1])
            return unique_log_files
        else:
            raise NotADirectoryError(f"{self.directory} is not a directory.")

    def parse_logs(self, log_files, fliter: list[str] = None):
        """解析日志文件中的度量信息"""
        name_epoch_metric = {}
        for name, log_file in log_files:
            with open(log_file, 'r') as file:
                content = file.read()
            metrics = []
            pattern = r"Iter\(val\) \[\d+/\d+\]\s+(.*?)(?=\n|$)"
            matches = re.findall(pattern, content, re.DOTALL)
            matches = [i for i in matches if 'acc' in i.lower()]

            for metrics_str in matches:
                pairs = metrics_str.split()
                if len(pairs) % 2 != 0:
                    print(name, log_file, f'{pairs=}')
                    pairs = pairs[:-1]
                result = {}
                for i in range(0, len(pairs) - 1, 2):
                    key = pairs[i].split(':')[0]
                    value = pairs[i + 1]
                    _k = key.strip()
                    if fliter is not None:
                        if _k not in fliter:
                            continue
                    result[key.strip()] = float(value.strip())

                result.pop('data_time', '')
                result.pop('time', '')
                metrics.append(result)

            pattern = r'Saving checkpoint at (\d+) epochs'
            epochs = [int(i) for i in re.findall(pattern, content, re.DOTALL)]
            if len(epochs) != len(metrics):
                print(name, log_file, f'{epochs=}', f'{metrics=}')
                epochs = epochs[:len(metrics)]

            epoch_metrices = {epochs[i]: metrics[i] for i in range(len(epochs))}
            name_epoch_metric[name] = epoch_metrices

        return name_epoch_metric

    def compile_metrics(self, name_epoch_metric):
        """整理度量历史"""
        _metrics_history = defaultdict(lambda: defaultdict(lambda: dict()))
        for name, epoch_metric in name_epoch_metric.items():
            for epoch, metrics in epoch_metric.items():
                for metric_name, value in metrics.items():
                    _metrics_history[metric_name][name][epoch] = value

        metrics_history = {
            metric_name: pd.DataFrame(name_epoch).sort_index(ascending=True)
            for metric_name, name_epoch in _metrics_history.items()
        }

        self.metrics_history = metrics_history
        return metrics_history

    def calculate_max_metrics(self, fliter: list[str] = None, to_dict: bool = False) -> (
            dict[str, pd.Series] | dict[str, dict[str, float]]):
        """计算每个度量的最大值"""
        if fliter:  # 非空
            assert isinstance(fliter, list)
            max_metrics = {k: v.max() for k, v in self.metrics_history.items() if k in fliter}
        else:
            max_metrics = {k: v.max() for k, v in self.metrics_history.items()}
        if to_dict:
            max_metrics = {k: v.to_dict() for k, v in max_metrics.items()}
        self.max_metrics = max_metrics
        return max_metrics


if __name__ == '__main__':
    # 使用示例
    roots = ['./run_multi', '../seg1/run_multi']
    fliter = ['mIoU', ]
    state_dict, strategy_dict, max_metrics_all_dataframe = get_metrics_from_logfile(roots, fliter)
    [print(v.to_string()) for k, v in max_metrics_all_dataframe.items()]

    print(f'{strategy_dict=}')
