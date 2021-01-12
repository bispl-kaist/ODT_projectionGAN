import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class GlobalMaxPooling2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor):
        return torch.max(torch.max(tensor, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]


class ChannelAttentionSE(nn.Module):
    def __init__(self, num_chans, reduction=16, use_gap=True, use_gmp=True):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling.
        self.gmp = GlobalMaxPooling2d()

        self.use_gap = use_gap
        self.use_gmp = use_gmp

        self.layer = nn.Sequential(
            nn.Linear(in_features=num_chans, out_features=num_chans // reduction),
            nn.ReLU(),
            nn.Linear(in_features=num_chans // reduction, out_features=num_chans)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        if not (self.use_gap or self.use_gmp):
            return tensor

        batch, chans, _, _ = tensor.shape
        if self.use_gap and self.use_gmp:
            gap = self.gap(tensor).view(batch, chans)
            gmp = self.gmp(tensor).view(batch, chans)
            features = self.layer(gap) + self.layer(gmp)

        elif self.use_gap:
            gap = self.gap(tensor).view(batch, chans)
            features = self.layer(gap)

        elif self.use_gmp:
            gmp = self.gmp(tensor).view(batch, chans)
            features = self.layer(gmp)

        else:
            raise RuntimeError('Impossible logic. Please check for errors.')

        att = self.sigmoid(features).view(batch, chans, 1, 1)

        return tensor * att


class EfficientChannelAttention(nn.Module):
    def __init__(self, num_chans, gamma=2, b=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)

        t = int(torch.abs((torch.log(num_chans, 2)+b) / gamma))
        k = t if t % 2 else t = 1

        self.conv = nn.Conv2d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):

        y = self.gap(tensor)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return tensor * y.expand_as(tensor)