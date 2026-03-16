"""
UPerNet segmentation head for multi-scale features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LateralConv(nn.Module):
    """1x1 Conv to adapt feature channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UPerNet(nn.Module):
    """
    UPerNet head: fuse multi-scale features for segmentation.
    Args:
        in_channels_list (List[int]): List of input feature channels.
        num_classes (int): Number of output classes.
        lateral_dim (int): Internal feature dim (default: 256).
    """
    def __init__(self, in_channels_list: List[int], num_classes: int, lateral_dim: int = 256):
        super().__init__()
        self.lateral_dim = lateral_dim
        self.laterals = nn.ModuleList([
            LateralConv(c, lateral_dim) for c in in_channels_list
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(lateral_dim, lateral_dim, 3, padding=1) for _ in in_channels_list
        ])
        self.fuse = nn.Conv2d(lateral_dim * len(in_channels_list), lateral_dim, 1)
        self.seg_head = nn.Conv2d(lateral_dim, num_classes, 1)
    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Adapt channels
        feats = [lat(f) for lat, f in zip(self.laterals, feats)]
        # Top-down FPN fusion
        for i in range(len(feats)-1, 0, -1):
            feats[i-1] = feats[i-1] + F.interpolate(feats[i], size=feats[i-1].shape[2:], mode='bilinear', align_corners=False)
        # FPN convs
        feats = [conv(f) for conv, f in zip(self.fpn_convs, feats)]
        # Upsample all to highest res
        out_size = feats[0].shape[2:]
        feats = [F.interpolate(f, size=out_size, mode='bilinear', align_corners=False) for f in feats]
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        x = self.seg_head(x)
        return x
