"""
ViT Windowed Attention cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath

class WindowPartition:
    """Partition feature map into non-overlapping windows with padding support."""
    @staticmethod
    def partition(x: torch.Tensor, window: int) -> torch.Tensor:
        N, C, H, W = x.shape
        
        # Pad to make dimensions divisible by window size
        pad_h = (window - H % window) % window
        pad_w = (window - W % window) % window
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            H, W = H + pad_h, W + pad_w
            
        x = x.view(N, C, H//window, window, W//window, window)
        x = x.permute(0,2,4,3,5,1).contiguous().view(-1, window*window, C)
        return x
    
    @staticmethod
    def reverse(windows: torch.Tensor, window: int, H: int, W: int, C: int, orig_H: int, orig_W: int) -> torch.Tensor:
        # H, W are padded dimensions
        Nw = (H // window) * (W // window)
        x = windows.view(-1, H//window, W//window, window, window, C)
        x = x.permute(0,5,1,3,2,4).contiguous()
        x = x.view(-1, C, H, W)
        
        # Remove padding to restore original dimensions
        if H != orig_H or W != orig_W:
            x = x[:, :, :orig_H, :orig_W]
            
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
        orig_H, orig_W = H, W
        
        x_windows = WindowPartition.partition(x, self.window)  # (num_windows*N, window*window, C)
        
        # Get padded dimensions for reverse operation
        pad_h = (self.window - H % self.window) % self.window
        pad_w = (self.window - W % self.window) % self.window
        padded_H, padded_W = H + pad_h, W + pad_w
        
        qkv = self.qkv(x_windows).reshape(-1, self.window*self.window, 3, self.heads, C//self.heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(-1, self.window*self.window, C)
        out = self.proj(out)
        out = WindowPartition.reverse(out, self.window, padded_H, padded_W, C, orig_H, orig_W)
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
