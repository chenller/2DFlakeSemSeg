from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# 将厘米转换为英寸
width_in_cm = 10  # 8, 14, 17
width_in_inches = width_in_cm * 0.393701
dpi = 300
fontsize = 6

# 设置全局字体大小
# plt.rcParams['font.size'] = fontsize

# 设置全局字体为 Arial
plt.rcParams['font.family'] = 'Arial'
palette = sns.color_palette()

linestyle_tuple = [
    # linestyle_str
    ('solid', 'solid'),  # Same as (0, ()) or '-'
    # ('dotted', 'dotted'),  # Same as (0, (1, 1)) or ':'
    # ('dashed', 'dashed'),  # Same as '--'
    # ('dashdot', 'dashdot'),  # Same as '-.'
    # linestyle_tuple
    # ('loosely dotted', (0, (1, 10))),  # .  .  .  .  .  .  .  .  .
    # ('dotted', (0, (1, 1))),  # ..................................
    # ('densely dotted', (0, (1, 1))),  # ..................................
    ('long dash with offset', (5, (10, 3))),  # -- -- -- -- -- -- -- -- --
    # ('loosely dashed', (0, (5, 10))),  # -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
    # ('dashed', (0, (5, 5))),  # - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ('densely dashed', (0, (5, 1))),  # ---------------------------------

    # ('loosely dashdotted', (0, (3, 10, 1, 10))),  # -  .  -  .  -  .    -  .    -  .    -  .
    ('dashdotted', (0, (3, 5, 1, 5))),  # - . - . - . - . - . - . - . - . - . - .
    ('densely dashdotted', (0, (3, 1, 1, 1))),  # -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.--.-.-.-

    ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),  # - . . - . . - . . - . . - . . - . . - . . - . . - . . -
    # ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),  # -   .  .  -   .  .  -   .  .  -   .  .  -
    ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]  # -..-..-..-..-..-..-..-..-..-..-..-..-

linestyle_tuple_seaborn = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted']


def compile_metrics(name_epoch_metric):
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

    return metrics_history


def calculate_max_metrics(metrics_history, fliter: list[str] = None, to_dict: bool = False) -> (
        dict[str, pd.Series] | dict[str, dict[str, float]]):
    """计算每个度量的最大值"""
    if fliter:  # 非空
        assert isinstance(fliter, list)
        max_metrics = {k: v.max() for k, v in metrics_history.items() if k in fliter}
    else:
        max_metrics = {k: v.max() for k, v in metrics_history.items()}
    if to_dict:
        max_metrics = {k: v.to_dict() for k, v in max_metrics.items()}
    return max_metrics


def plot_metrics(metric_name: str, strategy_xy: dict[str, list[list]]):
    strategy_xy = {k: v for k, v in strategy_xy.items()}
    print(strategy_xy)

    # 设置Seaborn风格
    sns.set(style='ticks')  # whitegrid

    # 获取Seaborn的当前调色板
    # palette = sns.color_palette()
    # print(palette)

    # 创建图表
    fig, ax = plt.subplots(figsize=(9, 4))

    # 存储平均值用于注释
    average_values = []

    # 绘制每个策略的折线图
    for i, (strategy, values) in enumerate(strategy_xy.items()):
        x_values, y_values = values
        if 'Margin' in strategy:
            i = 0
        elif 'Entropy' in strategy:
            i = 1
        elif 'Random' in strategy:
            i = 2
        color = palette[i % len(palette)]  # 获取新的颜色
        linestyle = linestyle_tuple_seaborn[i]
        linestyle = linestyle_tuple[i][1]
        # sns.lineplot(x=x_values, y=y_values, ax=ax, color=color)
        # sns.lineplot(x=x_values, y=y_values, label=strategy, color=color, linestyle=linestyle, marker='o', )
        plt.plot(x_values, y_values, label=strategy, color=color, linestyle=linestyle, marker='o', )
        # 计算并添加平均线
        average_y = np.mean(y_values)
        average_values.append([strategy, average_y])
        # ax.axhline(average_y, linestyle='--', color=color, linewidth=1)
    arial_font = fm.FontProperties(family='Arial')
    # 在图中标注出平均值
    # for i, (strategy, avg_val) in enumerate(average_values):
    #     ax.text(0.05, avg_val + pianyi, f'{avg_val:.2f}', color=palette[i % len(palette)], fontsize=12,
    #             fontproperties=arial_font,
    #             horizontalalignment='left', verticalalignment='bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 添加标题
    ax.set_title(f'Comparison of {metric_name} for Different Sampling Strategies')
    # 添加X轴和Y轴标签
    ax.set_xlabel('Ratio of Labeled Samples')
    ax.set_ylabel(metric_name)

    ax.set_xlim([0, 0.45])

    # 显示图例
    plt.legend()

    # 自动调整子图参数, 使之填充整个图像区域
    plt.tight_layout()
    plt.savefig(f'./img/batch12345_strategy_{metric_name}.svg')
    # 显示图表
    # plt.show()


def plot(strategy_dict):
    strategy_dataframe = {k: pd.DataFrame(v) for k, v in strategy_dict.items()}
    strategy_dataframe_mean = {}
    metric_strategy_xy_mean = defaultdict(lambda: defaultdict(list))
    for metric, df in strategy_dataframe.items():
        df_new = pd.DataFrame({})
        for i in df.columns:
            j = i[:-1]
            if j not in df_new.columns:
                df_new[j] = df[i]
            else:
                df_new[j] = df_new[j] + df[i]
            print(i)
        df_new = df_new / (len(df.columns) / len(df_new.columns))
        df_list = df_new.to_dict('list')
        x = list(df_new.index)
        for strategy, y in df_list.items():
            x_new = []
            y_new = []
            for i, v in enumerate(y):
                if not np.isnan(v):
                    x_new.append(float(x[i]))
                    y_new.append(v)
            metric_strategy_xy_mean[f'mean_{metric}'][strategy] = [x_new, y_new]

        strategy_dataframe_mean[metric] = df_new

    metric_strategy_xy = defaultdict(lambda: defaultdict(list))
    for metric, df in strategy_dataframe.items():
        df_list = df.to_dict('list')
        x = list(df.index)
        for strategy, y in df_list.items():
            x_new = []
            y_new = []
            for i, v in enumerate(y):
                if not np.isnan(v):
                    x_new.append(float(x[i]))
                    y_new.append(v)
            metric_strategy_xy[metric][strategy] = [x_new, y_new]

    for i, (metric, strategy) in enumerate(metric_strategy_xy_mean.items()):
        print(metric, strategy)
        plot_metrics(metric, strategy)
    for i, (metric, strategy) in enumerate(metric_strategy_xy.items()):
        print(metric, strategy)
        plot_metrics(metric, strategy)


if __name__ == '__main__':
    # plot()
    ...
