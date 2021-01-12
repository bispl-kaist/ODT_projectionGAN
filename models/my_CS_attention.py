'''
python script containing classes that are made by me
which are variants of channel and spatial attention
e.g) Mixture of channel attention from SE-net, spatial attention from SAGAN
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_
import torch

from models.CBAM import ChannelGate, SpatialGate
from models.Self_attention import Self_Attn


class CBSA(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBSA, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = Self_Attn(gate_channels)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
