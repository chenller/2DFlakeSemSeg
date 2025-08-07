import torch
import torch.nn as nn
from mmseg.registry import MODELS


@MODELS.register_module(name='ext-ToyBackbone')
class ToyBackbone(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_dim, 112, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(112, 112, kernel_size=3, stride=2),
                                  nn.ReLU(),
                                  )
        self.conv1 = nn.Sequential(nn.Conv2d(112, 112, kernel_size=3, stride=2),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(112, 224, kernel_size=3, stride=2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(224, 448, kernel_size=3, stride=2),
                                   nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(448, 896, kernel_size=3, stride=2),
                                   nn.ReLU())

    def forward(self, x):
        x=self.stem(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4
