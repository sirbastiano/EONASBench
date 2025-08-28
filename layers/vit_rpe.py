"""
ViT Encoder with Relative Positional Encoding cell implementation.
"""
import torch
import torch.nn as nn
from layers.registry import register
from layers.convnext_v1 import DropPath
from layers.vit_encoder import MHSA

class RelPosBias(nn.Module):
    """Relative positional bias for 2D attention with dynamic sizing."""
    def __init__(self, heads: int, max_size: int = 256):
        super().__init__()
        self.heads = heads
        self.max_size = max_size
        # Pre-allocate for maximum expected size
        self.rel_height = nn.Parameter(torch.zeros((2*max_size-1, heads)))
        self.rel_width = nn.Parameter(torch.zeros((2*max_size-1, heads)))
    
    def forward(self, H: int, W: int) -> torch.Tensor:
        # Ensure we don't exceed the pre-allocated size
        max_dim = max(H, W)
        if max_dim > self.max_size:
            # Reallocate if needed (not ideal, but handles edge cases)
            new_max_size = max_dim * 2
            device = self.rel_height.device
            self.rel_height = nn.Parameter(torch.zeros((2*new_max_size-1, self.heads), device=device))
            self.rel_width = nn.Parameter(torch.zeros((2*new_max_size-1, self.heads), device=device))
            self.max_size = new_max_size
        
        coords_h = torch.arange(H, device=self.rel_height.device)
        coords_w = torch.arange(W, device=self.rel_height.device)
        
        # Create coordinate grids for all pairs
        coords_flatten_h = coords_h.view(-1, 1).repeat(1, W).view(-1)  # [H*W]
        coords_flatten_w = coords_w.view(1, -1).repeat(H, 1).view(-1)  # [H*W]
        
        # Calculate relative coordinates for all pairs
        coords_h_rel = coords_flatten_h[:, None] - coords_flatten_h[None, :] + self.max_size - 1  # [H*W, H*W]
        coords_w_rel = coords_flatten_w[:, None] - coords_flatten_w[None, :] + self.max_size - 1  # [H*W, H*W]
        
        # Get bias values
        bias_h = self.rel_height[coords_h_rel]  # [H*W, H*W, heads]
        bias_w = self.rel_width[coords_w_rel]   # [H*W, H*W, heads]
        bias = bias_h + bias_w  # [H*W, H*W, heads]
        
        return bias.permute(2, 0, 1).unsqueeze(0)  # [1, heads, H*W, H*W]

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
        
        # Get bias - already in shape [1, heads, H*W, H*W]
        bias = self.rpe(H, W)
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
    def __init__(self, C: int, heads: int = 8, dim_mlp: int | None = None, drop_path: float = 0.0):
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
