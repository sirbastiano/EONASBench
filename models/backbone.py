"""
Backbone macro-architecture assembly for benchmarking framework.
"""
import torch
import torch.nn as nn
from typing import List

class Conv3x3(nn.Module):
    """
    3x3 Conv + BatchNorm2d + GELU stem.
    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
    """
    def __init__(self, in_ch: int, out_ch: int, norm: str = 'batchnorm', stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Downsample(nn.Module):
    """MaxPool2d + 1x1 Conv to change channels."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)
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
        channels = cfg.get('channels')
        if channels is None:
            channels = [cfg['stem_out'] * (2 ** i) for i in range(len(cfg['stages']))]
        if len(channels) != len(cfg['stages']):
            raise ValueError("Length of 'channels' must match number of stages")

        stem_stride = cfg.get('stem_stride', 1)
        C = channels[0]
        self.stem = Conv3x3(3, C, norm=cfg.get('norm', 'layernorm'), stride=stem_stride)
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        total_cells = sum(stage['cells'] for stage in cfg['stages'])
        drop_path = torch.linspace(0, cfg['drop_path_rate'], total_cells).tolist() if total_cells else []
        k = 0
        for stage_idx, s in enumerate(cfg['stages']):
            stage_channels = channels[stage_idx]
            cell_kwargs = s.get('cell_kwargs', {})
            builder = lambda C, drop_path, layer=s['layer'], kwargs=cell_kwargs: build_cell(
                layer, C, drop_path=drop_path, **kwargs
            )
            self.stages.append(
                Stage(
                    stage_channels,
                    builder,
                    cells=s['cells'],
                    drop_path_slice=drop_path[k:k + s['cells']],
                )
            )
            k += s['cells']
            next_channels = channels[stage_idx + 1] if stage_idx + 1 < len(channels) else stage_channels * 2
            self.downs.append(Downsample(stage_channels, next_channels))
        self.out_channels = channels
        self.final_channels = channels[-1] * 2 if channels else 1
    def forward(self, x: torch.Tensor):
        feats = []
        x = self.stem(x)
        for stage, down in zip(self.stages, self.downs):
            x = stage(x)
            feats.append(x)
            x = down(x)
        return feats, x
