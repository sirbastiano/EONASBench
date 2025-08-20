"""
ConvNeXt V1 cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register

class DropPath(nn.Module):
    """Stochastic depth regularization."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

@register('convnext_v1')
class ConvNeXtV1Cell(nn.Module):
    """
    ConvNeXt V1 cell: DWConv → LN → PW-MLP (expand) + GELU, DropPath, residual.
    Args:
        C (int): Channels.
        expand (int): MLP expansion ratio.
        dwk (int): Depthwise conv kernel size.
        drop_path (float): DropPath rate.
    """
    def __init__(self, C: int, expand: int = 4, dwk: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(C, C, dwk, padding=dwk//2, groups=C)
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
        x = x.permute(0,2,3,1)  # (N,H,W,C)
        x = self.norm(x)
        x = self.pwmlp(x)
        x = x.permute(0,3,1,2)  # (N,C,H,W)
        x = self.drop_path(x)
        return shortcut + x
