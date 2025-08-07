import numpy as np
import torch

from mmengine.registry import Registry

ALSTRATEGY = Registry('ALStrategy')
eps = 1e-5


def select(pred: torch.Tensor, yuzhi: float = 0.75, num: int = 5000):
    """
    Select features based on prediction confidence.

    This function reshapes the prediction tensor into a 2D tensor and finds the maximum value for each column.
    It then sorts these maximum values. Depending on the sorted values and a predefined threshold,
    it selects features. If the `num`-th maximum value is less than or equal to the threshold `yuzhi`,
    all features with maximum values less than `yuzhi` are retained. Otherwise, only the top `num` features
    with the smallest maximum values are selected.
    :param pred: torch.Tensor, shape[c,w,h] - The prediction confidence tensor.
    :param yuzhi: float - The confidence threshold (default is 0.75).
    :param num: int - The number of features to select (default is 5000).
    :return:
    pred_new: torch.Tensor, shape[c,n] - The filtered prediction confidence tensor.
    """
    # Reshape the pred tensor into a 2D tensor
    pred = pred.reshape(pred.shape[0], -1)
    # Find the maximum value for each column
    max_values = torch.max(pred, dim=0)[0]
    # Sort the maximum values in ascending order
    max_sort_values, max_sort_indices = torch.sort(max_values)
    # Select features based on the comparison of the `num`-th maximum value and the threshold `yuzhi`
    if max_sort_values[num] <= yuzhi:
        # Retain all features with maximum values less than `yuzhi`
        pred_new = pred[:, max_values < yuzhi]
    else:
        # Select the top `num` features with the smallest maximum values
        pred_new = pred[:, max_sort_indices[:num]]
    return pred_new


@ALSTRATEGY.register_module()
class Strategy:
    """
    A strategy class for evaluating and sampling model predictions.
    """

    def assessment_value(self, preds: list):
        """
        Returns the model's prediction as the assessment value.

        :param preds: A list of model predictions.
        :return: The assessment value.
        """
        assess_value = preds
        return assess_value

    def sample(self, result_list: list, num: int):
        """
        Performs sampling and returns the sampled indices and unlabelled indices.

        :param result_list: A list containing model predictions.
        :param num: The number of samples needed.
        :return: Sampled indices list and unlabelled indices list, returning a list with 0 as the default value.
        """
        sample_idx = [0]
        unlabeled_idx = [0]
        return sample_idx, unlabeled_idx


    def __str__(self):
        return "Strategy"


@ALSTRATEGY.register_module()
class LeastConfidence(Strategy):
    def __init__(self, ):
        super().__init__()

    def assessment_value(self, preds):
        def _uncertainty(_pred: torch.Tensor):
            _pred = select(_pred)

            uncertainties = torch.mean(_pred.max(dim=0)[0])  # mean of max prob
            return uncertainties.cpu()

        preds = [[i.seg_logits.data.softmax(0), i.sample_idx] for i in preds]  # [c,w,h]
        uncertainties = [[_uncertainty(pred), sample_idx] for pred, sample_idx in preds]
        return uncertainties

    def sample(self, result_list, num):
        uncertainties = [i[0] for i in result_list]
        sample_idx_list = [i[1] for i in result_list]
        uncertainties_sort, idx = torch.sort(torch.tensor(uncertainties), descending=False)  # 升序
        sample_idx = [sample_idx_list[i] for i in idx[:num]]
        return sample_idx, None

    def __str__(self):
        return 'LeastConfidence'


@ALSTRATEGY.register_module()
class MarginSampling(Strategy):
    def __init__(self, ):
        super().__init__()

    def assessment_value(self, preds):
        def _uncertainty(prob: torch.Tensor):
            prob = select(prob)

            probs_sorted, idxs = prob.sort(dim=0, descending=True)  # 降序
            uncertainties = torch.mean(probs_sorted[0] - probs_sorted[1])
            return uncertainties.cpu()

        probs = [[i.seg_logits.data.softmax(0), i.sample_idx] for i in preds]  # [c,w,h]
        uncertainties = [[_uncertainty(prob), sample_idx] for prob, sample_idx in probs]
        return uncertainties

    def sample(self, result_list, num):
        uncertainties = [i[0] for i in result_list]
        sample_idx_list = [i[1] for i in result_list]
        uncertainties_sort, idx = torch.sort(torch.tensor(uncertainties), descending=False)  # 升序
        sample_idx = [sample_idx_list[i] for i in idx[:num]]
        return sample_idx, None

    def __str__(self):
        return 'MarginSampling'


@ALSTRATEGY.register_module()
class EntropySampling(Strategy):
    def __init__(self, ):
        super().__init__()

    def assessment_value(self, preds):
        probs = [[i.seg_logits.data.softmax(0), i.sample_idx] for i in preds]  # [c,w,h]
        probs = [[select(prob), sample_idx] for prob, sample_idx in probs]
        uncertainties = [[(prob * prob.log()).sum(0).mean().cpu(), sample_idx] for prob, sample_idx in probs]
        return uncertainties

    def sample(self, result_list, num):
        uncertainties = [i[0] for i in result_list]
        sample_idx_list = [i[1] for i in result_list]
        uncertainties_sort, idx = torch.sort(torch.tensor(uncertainties), descending=False)  # 升序
        sample_idx = [sample_idx_list[i] for i in idx[:num]]
        return sample_idx, None

    def __str__(self):
        return 'EntropySampling'


@ALSTRATEGY.register_module()
class RandomSampling(Strategy):
    def sample(self, result_list, num):
        return None, np.random.choice(list(range(len(result_list))), num, replace=False)

    def __str__(self):
        return "RandomSampling"


if __name__ == '__main__':
    # s = RandomSampling()
    # print(str(s))

    # x = torch.randn((3, 64, 64))
    # probs_sorted, idxs = x.sort(dim=0, descending=True)  # 降序
    # uncertainties = torch.mean(probs_sorted[0] - probs_sorted[1])
    # print(probs_sorted[:, 0, 0], uncertainties)
    x = torch.randn((3, 64, 64))
    y = x.max(dim=0)
    print(x[:, 0, 0], y[0][0, 0])
