"""
ViT Windowed Attention cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath

class WindowPartition:
    """Partition feature map into non-overlapping windows."""
    @staticmethod
    def partition(x: torch.Tensor, window: int) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.view(N, C, H//window, window, W//window, window)
        x = x.permute(0,2,4,3,5,1).contiguous().view(-1, window*window, C)
        return x
    @staticmethod
    def reverse(windows: torch.Tensor, window: int, H: int, W: int, C: int) -> torch.Tensor:
        Nw = (H // window) * (W // window)
        x = windows.view(-1, H//window, W//window, window, window, C)
        x = x.permute(0,5,1,3,2,4).contiguous()
        x = x.view(-1, C, H, W)
        return x

class WindowMHSA(nn.Module):
    """Windowed multi-head self-attention."""
    def __init__(self, C: int, heads: int = 8, window: int = 7):
        super().__init__()
        self.heads = heads
        self.window = window
        self.scale = (C // heads) ** -0.5
        self.qkv = nn.Linear(C, C * 3)
        self.proj = nn.Linear(C, C)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_windows = WindowPartition.partition(x, self.window)  # (num_windows*N, window*window, C)
        qkv = self.qkv(x_windows).reshape(-1, self.window*self.window, 3, self.heads, C//self.heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(-1, self.window*self.window, C)
        out = self.proj(out)
        out = WindowPartition.reverse(out, self.window, H, W, C)
        return out

@register('vit_window')
class ViTWindowCell(nn.Module):
    """
    ViT windowed attention cell (Swin-style, no patch embed).
    Args:
        C (int): Channels.
        heads (int): Number of attention heads.
        window (int): Window size.
        drop_path (float): DropPath rate.
    """
    def __init__(self, C: int, heads: int = 8, window: int = 7, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn = WindowMHSA(C, heads, window)
        self.norm2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, 4*C),
            nn.GELU(),
            nn.Linear(4*C, C)
        )
        self.drop_path = DropPath(drop_path)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = x.permute(0,2,3,1)
        x = self.norm1(x)
        x = x.permute(0,3,1,2)
        x = self.attn(x)
        x = shortcut + self.drop_path(x)
        shortcut2 = x
        x = x.permute(0,2,3,1)
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0,3,1,2)
        x = shortcut2 + self.drop_path(x)
        return x
