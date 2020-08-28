# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
import math

"""MODEL"""


class LightNet(nn.Module):

    def __init__(self, depth=17, n_channels=64, kernel_size=3, stride=1, padding=1):
        # def __init__(self, depth=8, n_channels=64, kernel_size=3, stride=1, padding=1):
        super(LightNet, self).__init__()

        layers = []

        layers.append(nn.Conv2d(3, 64, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=kernel_size, padding=padding))
        layers.append(nn.ReLU())
        # layers.append(nn.Conv2d(in_channels=3, out_channels=1, kernel_size=kernel_size, padding=padding))
        # layers.append(nn.ReLU())
        self.tanh = nn.Tanh()
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        x = x.float()
        res = x
        out = self.dncnn(x)
        out_map = out.clone()
        illu_estim = torch.mean(out, dim=(1, 2, 3)).view(res.shape[0], 1, 1, 1)
        out = res / illu_estim
        out = self.tanh(out)

        return out, illu_estim

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


