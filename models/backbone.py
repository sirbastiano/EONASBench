"""
Backbone macro-architecture assembly for benchmarking framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any

class Conv3x3(nn.Module):
    """
    3x3 Conv + BatchNorm2d + GELU stem.
    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = 'batchnorm'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """MaxPool2d + 1x1 Conv to double channels."""
    def __init__(self, C: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.proj = nn.Conv2d(C, 2*C, 1, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.proj(x)
        return x


class Stage(nn.Module):
    """
    Stage: stack N cells of the same type, each (C,H,W)->(C,H,W).
    Args:
        C (int): Channels.
        cell_builder (callable): Function to build a cell.
        cells (int): Number of cells.
        drop_path_slice (list): DropPath rates for each cell.
    """
    def __init__(self, C: int, cell_builder, cells: int = 3, drop_path_slice=()):
        super().__init__()
        self.blocks = nn.Sequential(*[
            cell_builder(C, drop_path=drop_path_slice[i]) for i in range(cells)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class Backbone(nn.Module):
    """
    Macro-architecture backbone: stem, 3 stages, downsampling, feature collection.
    Args:
        cfg (dict): Model config.
    """
    def __init__(self, cfg: dict):
        super().__init__()
        from layers.registry import build_cell
        C = cfg['stem_out']
        self.stem = Conv3x3(3, C, norm=cfg.get('norm', 'layernorm'))
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        drop_path = torch.linspace(0, cfg['drop_path_rate'], 3*len(cfg['stages'])).tolist()
        k = 0
        for s in cfg['stages']:
            builder = lambda C, drop_path: build_cell(s['layer'], C, drop_path=drop_path)
            self.stages.append(Stage(C, builder, cells=s['cells'], drop_path_slice=drop_path[k:k+s['cells']]))
            k += s['cells']
            self.downs.append(Downsample(C))
            C *= 2
        self.out_channels = [cfg['stem_out'] * (2**i) for i in range(3)]
    def forward(self, x: torch.Tensor):
        feats = []
        x = self.stem(x)
        for stage, down in zip(self.stages, self.downs):
            x = stage(x)
            feats.append(x)
            x = down(x)
        return feats, x
