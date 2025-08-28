"""
ViT Encoder cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath

class MHSA(nn.Module):
    """Multi-head self-attention with learned 2D pos-embed."""
    def __init__(self, C: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = (C // heads) ** -0.5
        self.qkv = nn.Linear(C, C * 3)
        self.proj = nn.Linear(C, C)
        self.pos_embed = nn.Parameter(torch.zeros(1, 0, C))  # set in forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1,2)  # (N, H*W, C)
        if self.pos_embed.shape[1] != H*W:
            self.pos_embed = nn.Parameter(torch.zeros(1, H*W, C, device=x.device))
        x_flat = x_flat + self.pos_embed
        qkv = self.qkv(x_flat).reshape(N, H*W, 3, self.heads, C//self.heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(N, H*W, C)
        out = self.proj(out)
        out = out.transpose(1,2).reshape(N, C, H, W)
        return out

@register('vit_encoder')
class ViTEncoderCell(nn.Module):
    """
    ViT encoder cell: Pre-LN, MHSA, MLP, DropPath, residual.
    Args:
        C (int): Channels.
        heads (int): Number of attention heads.
        dim_mlp (int): MLP hidden dim.
        drop_path (float): DropPath rate.
    """
    def __init__(self, C: int, heads: int = 8, dim_mlp: int | None = None, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn = MHSA(C, heads)
        self.norm2 = nn.LayerNorm(C)
        self.mlp = nn.Sequential(
            nn.Linear(C, (dim_mlp or 4*C)),
            nn.GELU(),
            nn.Linear((dim_mlp or 4*C), C)
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
