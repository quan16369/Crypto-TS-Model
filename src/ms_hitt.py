import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from einops import rearrange, repeat

class ResidualBlock(nn.Module):
    def __init__(self, d_model, dilation_rate):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=dilation_rate, dilation=dilation_rate)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.gelu(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.dropout(self.conv2(x.transpose(1, 2)).transpose(1, 2))
        return self.norm(x + residual)

class TemporalPatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        
    def forward(self, x):
        x = rearrange(x, 'b (t p) d -> b t (p d)', p=self.patch_size)
        return self.proj(x)

class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, n_heads, scales=[1,2,4]):
        super().__init__()
        self.scales = scales
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in scales
        ])
        self.mix = nn.Linear(len(scales)*d_model, d_model)
        
    def forward(self, x):
        B, T, D = x.shape
        outputs = []
        for scale, attn in zip(self.scales, self.attention_heads):
            if scale > 1:
                x_scale = F.avg_pool1d(x.transpose(1,2), kernel_size=scale).transpose(1,2)
            else:
                x_scale = x
            out, _ = attn(x_scale, x_scale, x_scale)
            if scale > 1:
                out = F.interpolate(out.transpose(1,2), size=T).transpose(1,2)
            outputs.append(out)
        return self.mix(torch.cat(outputs, dim=-1))

class MSHiTT(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = config['model']
        self.enc_in = cfg['enc_in']
        self.d_model = cfg['d_model']
        self.pred_len = cfg['pred_len']
        self.patch_size = cfg.get('patch_size', 12)
        
        # Input processing
        self.value_embedding = nn.Linear(self.enc_in, self.d_model)
        self.patch_embed = TemporalPatchEmbedding(self.patch_size, self.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, self.d_model))  # Max seq len
        
        # Encoder
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(self.d_model, dilation_rate=2**i),
                MultiScaleAttention(self.d_model, n_heads=8)
            ) for i in range(4)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Linear(self.d_model * 4, self.pred_len)
        )
        
    def forward(self, x):
        B, T, D = x.shape
        
        # Hybrid Embedding
        x_val = self.value_embedding(x)  # Value embedding
        x_patch = self.patch_embed(x)    # Patch embedding
        x = x_val + x_patch
        
        # Positional encoding
        x = x + self.pos_embed[:, :T]
        
        # Multi-scale processing
        for block in self.blocks:
            x = block(x)
        
        # Decoding
        out = self.decoder(x.mean(dim=1))  # Global average pooling
        return out.unsqueeze(-1)  # [B, pred_len, 1]