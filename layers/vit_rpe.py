"""
ViT Encoder with Relative Positional Encoding cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath
from layers.vit_encoder import MHSA

class RelPosBias(nn.Module):
    """Relative positional bias for 2D attention."""
    def __init__(self, heads: int, size: int = 32):
        super().__init__()
        self.heads = heads
        self.rel_height = nn.Parameter(torch.zeros((2*size-1, heads)))
        self.rel_width = nn.Parameter(torch.zeros((2*size-1, heads)))
    def forward(self, H: int, W: int) -> torch.Tensor:
        coords_h = torch.arange(H)
        coords_w = torch.arange(W)
        rel_h = coords_h[None, :] - coords_h[:, None] + H - 1
        rel_w = coords_w[None, :] - coords_w[:, None] + W - 1
        bias = self.rel_height[rel_h][:, :, None, :] + self.rel_width[rel_w][None, :, :, :]
        return bias.permute(2,3,0,1)  # (1, heads, H, W)

class MHSA_RPE(MHSA):
    """MHSA with relative positional bias."""
    def __init__(self, C: int, heads: int = 8):
        super().__init__(C, heads)
        self.rpe = RelPosBias(heads)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1,2)
        if self.pos_embed.shape[1] != H*W:
            self.pos_embed = nn.Parameter(torch.zeros(1, H*W, C, device=x.device))
        x_flat = x_flat + self.pos_embed
        qkv = self.qkv(x_flat).reshape(N, H*W, 3, self.heads, C//self.heads).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2,-1)) * self.scale
        bias = self.rpe(H, W).reshape(1, self.heads, H*W, H*W)
        attn = attn + bias
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1,2).reshape(N, H*W, C)
        out = self.proj(out)
        out = out.transpose(1,2).reshape(N, C, H, W)
        return out

@register('vit_rpe')
class ViTRPECell(nn.Module):
    """
    ViT encoder cell with relative positional encoding.
    Args:
        C (int): Channels.
        heads (int): Number of attention heads.
        dim_mlp (int): MLP hidden dim.
        drop_path (float): DropPath rate.
    """
    def __init__(self, C: int, heads: int = 8, dim_mlp: int = None, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(C)
        self.attn = MHSA_RPE(C, heads)
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
