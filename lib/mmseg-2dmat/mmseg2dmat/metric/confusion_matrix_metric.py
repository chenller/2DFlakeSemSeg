
import logging
from pprint import pprint
from typing import Sequence, Dict, Optional, List
import numpy as np
import torch
from mmengine import print_log
from mmengine.dist import is_main_process
import mmengine.dist as dist
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from tabulate import tabulate
from torchmetrics.classification import MulticlassConfusionMatrix
from mmseg.visualization import SegLocalVisualizer

from mmseg.registry import METRICS
from torchmetrics.utilities.plot import plot_confusion_matrix


def array_to_markdown(matrix: np.ndarray,
                      xlabel: Optional[List[str]] = None,
                      ylabel: Optional[List[str]] = None,
                      floatfmt: str = None):
    """
    Converts a NumPy matrix to a Markdown table with optional xlabel and ylabel.

    Parameters:
    - matrix: 2D numpy array
    - xlabel: string, optional, label for columns
    - ylabel: string, optional, label for rows

    Returns:
    - markdown_table: string, the matrix formatted as a Markdown table
    """
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.numpy()  # torch.Tensor

    # Check if matrix is 2D
    if matrix.ndim != 2:
        raise ValueError("The input matrix must be 2D")
    if floatfmt is None:
        if np.issubdtype(matrix.dtype, np.integer):
            floatfmt = ",.0f"
        else:
            floatfmt = ".6f"
    markdown_table = tabulate(matrix,
                              tablefmt='pipe',  # markdown
                              headers=['True\\Pred'] + ylabel,
                              showindex=xlabel,
                              floatfmt=floatfmt)

    return markdown_table


def fig_to_array(fig):
    fig.canvas.draw()
    # Now we can save it to a numpy array.
    array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    array = array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return array


def confusion_matrix(pred: torch.Tensor, label: torch.Tensor, num_classes):
    """
    计算混淆矩阵。

    参数:
    - label: 真实标签的Tensor，形状为(N,)，其中N是样本数量。
    - pred: 预测标签的Tensor，形状为(N,)，与label对应。
    - n_classes: 分类的总数。

    返回:
    - conf_matrix: 混淆矩阵，形状为(n_classes, n_classes)。行号为真实标签，列号为预测标签
                    预测标签
    真实标签     0   1   2   3   ...
       0
       1
       2
       3
      ...
    """
    # 确保标签和预测值是LongTensor，以便进行索引操作
    label = label.long().view(-1)
    pred = pred.long().view(-1)
    # 使用广播机制将label和pred扩展为可以进行元素相乘的形状，得到n * label + pred的效果
    # 这里利用了torch的扩展维度功能，n相当于在本例中为n_classes，但直接使用n_classes会更清晰
    indices = label + num_classes * pred
    # 使用bincount计算每个类别组合的数量
    # 注意：PyTorch的bincount要求输入是一维的，所以我们直接对indices应用bincount
    conf_counts = torch.bincount(indices, minlength=num_classes * num_classes)
    # 将一维的结果重塑为n_classes x n_classes的混淆矩阵
    conf_matrix = conf_counts.reshape(num_classes, num_classes)

    return conf_matrix


@METRICS.register_module(name='ext-ConfusionMatrixMetric')
class ConfusionMatrixMetric(BaseMetric):
    iter = 1

    def __init__(self, mean: bool = True,
                 log10=False,
                 ignore_index: int = 255,
                 collect_device: str = 'cpu',
                 prefix: str = None,
                 format_only: bool = False,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.mean = mean
        self.log10 = log10
        self.ignore_index = ignore_index
        self.metric: Dict[str, torch.Tensor]
        self.num_classes: int = None
        self.reset_metric()
        # format_only (bool): Only format result for results commit without
        #             perform evaluation. It is useful when you want to save the result
        #             to a specific format and submit it to the test server.
        #             Defaults to False.
        self.format_only = format_only

    def reset_metric(self):
        self.metric: Dict[str, torch.Tensor] = dict(ConfusionMatrix=None, num=None)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        num_classes = len(self.dataset_meta['classes'])
        self.num_classes = num_classes
        for data_sample in data_samples:
            pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # format_only always for test dataset without ground truth
            if not self.format_only:
                label = data_sample['gt_sem_seg']['data'].squeeze().to(
                    pred_label)
                if self.metric['ConfusionMatrix'] is None:
                    self.metric['ConfusionMatrix'] = confusion_matrix(pred=pred_label, label=label,
                                                                      num_classes=num_classes)
                    self.metric['num'] = torch.tensor(1).long().to(pred_label)
                else:
                    self.metric['ConfusionMatrix'] += confusion_matrix(pred=pred_label, label=label,
                                                                       num_classes=num_classes)
                    self.metric['num'] = self.metric['num'] + 1

        # print_log('ConfusionMatrixMetric - run process', logger='current', )

    def compute_metrics(self, results: list = None) -> Dict[str, float]:
        # print_log('ConfusionMatrixMetric - run compute_metrics', logger='current', )
        return dict()

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # print_log('ConfusionMatrixMetric - run evaluate', logger='current', )
        if self.metric['ConfusionMatrix'] is None:
            print_log('Confusion Matrix Metric is None', logger='current')
            return dict()

        dist.all_reduce_dict(self.metric, op='sum', )

        if is_main_process():
            confusion_matrix = self.metric['ConfusionMatrix']
            num = self.metric['num']
            print_log(f'ConfusionMatrix : length - {num.item()}', logger='current', )
            if self.mean:
                confusion_matrix = confusion_matrix / num.float()
            if self.log10:
                confusion_matrix = torch.log10(confusion_matrix)
            # print
            self.print_confusion_matrix(confusion_matrix)
            # plot
            fig, ax = plot_confusion_matrix(confmat=confusion_matrix)
            vis: SegLocalVisualizer = SegLocalVisualizer.get_current_instance()
            np_array = fig_to_array(fig)
            # logger
            vis.add_image(name=f'ConfusionMatrix_{self.iter:02}.png', image=np_array, step=self.iter)

            self.iter += 1

        # reset the results list
        self.reset_metric()
        return dict()

    def print_confusion_matrix(self, confusion_matrix: torch.Tensor):
        print_log('=' * 20 + 'Confusion Matrix Metric' + '>' * 20, logger='current', )
        cm_array = confusion_matrix.detach().cpu().numpy()
        label = [f'{v} - {i}' for i, v in enumerate(self.dataset_meta['classes'])]
        cm_str = array_to_markdown(cm_array, xlabel=label, ylabel=label)
        print_log('\n' + cm_str, logger='current', )
        print_log('<' * 20 + 'Confusion Matrix Metric' + '=' * 20, logger='current', )


if __name__ == '__main__':
    # 示例使用
    # 假设我们有4个类别，以下为示例标签和预测值
    labels = torch.tensor([0, 0, 0, 0, 0, 4, 0])  # 真实标签
    predictions = torch.tensor([1, 1, 2, 0, 0, 0, 0])  # 预测标签

    # 计算混淆矩阵
    confusion_matrix_result = confusion_matrix(labels, predictions, num_classes=5)
    print(confusion_matrix_result)

    from mmseg.evaluation.metrics.iou_metric import IoUMetric

    iou = IoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'])
    iou.dataset_meta = dict(classes=list(range(5)))
    iou.process(data_batch=1, data_samples=[{
        'pred_sem_seg': {'data': predictions},
        'gt_sem_seg': {'data': labels},
    }])
    iou.compute_metrics(iou.results)


    def calculate_metrics_from_confusion_matrix(cm):
        """
        Calculates IoU, Accuracy, Dice, F-score, Precision, and Recall from a confusion matrix.

        Parameters:
        - cm: ndarray of shape (n_classes, n_classes), confusion matrix

        Returns:
        - metrics: dict, contains IoU, Accuracy, Dice, F-score, Precision, and Recall for each class and overall
        """
        n_classes = cm.shape[0]
        IoU = torch.zeros(n_classes)
        Dice = torch.zeros(n_classes)
        Fscore = torch.zeros(n_classes)
        Precision = torch.zeros(n_classes)
        Recall = torch.zeros(n_classes)

        for i in range(n_classes):
            TP = cm[i, i]
            FN = cm[i, :].sum() - TP
            FP = cm[:, i].sum() - TP

            IoU[i] = TP / (TP + FP + FN)
            Precision[i] = TP / (TP + FP)
            Recall[i] = TP / (TP + FN)
            Dice[i] = (2 * TP) / (2 * TP + FP + FN)
            Fscore[i] = 2 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i])
        print(torch.diag(cm))
        print(torch.sum(cm, dim=1))
        overall_Acc = torch.diag(cm) / torch.sum(cm, dim=1)

        metrics = {
            'IoU': IoU,
            'Accuracy': overall_Acc,
            'Dice': Dice,
            'F-score': Fscore,
            'Precision': Precision,
            'Recall': Recall
        }

        return metrics


    pprint(calculate_metrics_from_confusion_matrix(confusion_matrix_result))
