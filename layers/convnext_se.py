"""
ConvNeXt SE cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath

class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, C: int, se_ratio: float = 0.0625):
        super().__init__()
        hidden = max(1, int(C * se_ratio))
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, C, 1),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)

@register('convnext_se')
class ConvNeXtSECell(nn.Module):
    """
    ConvNeXt V1 + SE cell.
    Args:
        C (int): Channels.
        expand (int): MLP expansion ratio.
        dwk (int): Depthwise conv kernel size.
        se_ratio (float): SE ratio.
        drop_path (float): DropPath rate.
    """
    def __init__(self, C: int, expand: int = 4, dwk: int = 7, se_ratio: float = 0.0625, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(C, C, dwk, padding=dwk//2, groups=C)
        self.norm = nn.LayerNorm(C)
        self.pwmlp = nn.Sequential(
            nn.Linear(C, expand*C),
            nn.GELU(),
            nn.Linear(expand*C, C)
        )
        self.se = SqueezeExcite(C, se_ratio)
        self.drop_path = DropPath(drop_path)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pwmlp(x)
        x = x.permute(0,3,1,2)
        x = self.se(x)
        x = self.drop_path(x)
        return shortcut + x
