"""
ConvNeXt Dilated cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath

@register('convnext_dil')
class ConvNeXtDilCell(nn.Module):
    """
    ConvNeXt V1 with dilated DWConv.
    Args:
        C (int): Channels.
        expand (int): MLP expansion ratio.
        dwk (int): Depthwise conv kernel size.
        dilation (int): Dilation for DWConv.
        drop_path (float): DropPath rate.
    """
    def __init__(self, C: int, expand: int = 4, dwk: int = 7, dilation: int = 2, drop_path: float = 0.0):
        super().__init__()
        pad = ((dwk-1)//2) * dilation
        self.dwconv = nn.Conv2d(C, C, dwk, padding=pad, groups=C, dilation=dilation)
        self.norm = nn.LayerNorm(C)
        self.pwmlp = nn.Sequential(
            nn.Linear(C, expand*C),
            nn.GELU(),
            nn.Linear(expand*C, C)
        )
        self.drop_path = DropPath(drop_path)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pwmlp(x)
        x = x.permute(0,3,1,2)
        x = self.drop_path(x)
        return shortcut + x
