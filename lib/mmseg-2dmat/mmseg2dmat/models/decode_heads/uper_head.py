from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.uper_head import UPerHead as UPerHead_Base
from mmseg.utils import ConfigType, SampleList


@MODELS.register_module(name='mmseg2dmat-UPerHead')
class UPerHead(UPerHead_Base):
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits: Tensor = self.forward(inputs)
        c = seg_logits.shape[1]
        seg_logits_new = seg_logits.new_tensor(torch.full_like(seg_logits, fill_value=-100), requires_grad=False)
        # print(seg_logits_new.max())
        for i, data_samples in enumerate(batch_data_samples):
            annotations_offset = data_samples.annotations_offset
            annotations_num_classes = data_samples.annotations_num_classes
            st, et = annotations_offset, annotations_offset + annotations_num_classes + 1
            seg_logits_new[i, st:et, :, :] = seg_logits[i, st:et, :, :]
        # print(seg_logits_new.max())

        losses = self.loss_by_feat(seg_logits_new, batch_data_samples)
        return losses
