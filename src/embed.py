import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np
from typing import Dict, Any, Optional, Union, Tuple 

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                   -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class VolatilityEmbedding(nn.Module):
    """Bổ sung embedding cho biến động giá (đặc biệt quan trọng với crypto)"""
    def __init__(self, d_model, lookback=10):
        super().__init__()
        self.lookback = lookback
        self.proj = nn.Linear(1, d_model)
        
    def forward(self, x_close):  # x_close: [B, T, 1] (close price)
        returns = x_close.diff(dim=1).abs()  # Absolute returns
        volatility = returns.unfold(1, self.lookback, 1).std(dim=-1, keepdim=True)  # [B, T, 1]
        return self.proj(volatility)  # [B, T, d_model]

class CryptoTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_size=16):
        super().__init__()
        self.c_in = c_in  # Số features đầu vào (13)
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular')
        )
        
    def forward(self, x):
        B, NP, _ = x.shape
        # Tính toán số features mỗi patch (phải bằng c_in)
        features_per_patch = self.c_in
        
        # Reshape chính xác
        x = x.view(B, NP, self.patch_size, features_per_patch)  # [64, 35, 16, 13]
        x = x.permute(0, 3, 1, 2)  # [B, C, NP, patch_len] -> [64, 13, 35, 16]
        x = x.reshape(B, features_per_patch, -1)  # [64, 13, 560]
        x = self.conv(x)  # [64, d_model, 560]
        x = x.reshape(B, self.conv[-1].out_channels, NP, -1)  # [64, d_model, 35, 16]
        x = x.permute(0, 2, 3, 1)  # [64, 35, 16, d_model]
        return x.reshape(B, NP, -1)  # [64, 35, 16*d_model]

class CryptoTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.minute_embed = nn.Embedding(60, d_model)
        self.hour_embed = nn.Embedding(24, d_model)
        
    def forward(self, x_mark):
        # Sử dụng chỉ số an toàn
        minute_idx = min(3, x_mark.size(-1)-1)  # Lấy chiều cuối cùng nếu không đủ
        hour_idx = min(2, x_mark.size(-1)-1)
        
        minutes = (x_mark[..., minute_idx] * 59).long()  # Chuyển từ [0-1] về phút thực
        hours = (x_mark[..., hour_idx] * 23).long()      # Chuyển từ [0-1] về giờ thực
        
        return self.minute_embed(minutes) + self.hour_embed(hours)

class CryptoDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.token_embedding = CryptoTokenEmbedding(c_in, d_model)
        self.volatility_embedding = VolatilityEmbedding(d_model)
        self.time_embedding = CryptoTimeEmbedding(d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.volatility_gate = nn.Linear(d_model, d_model)

    def forward(self, x, x_mark=None):
        # 1. Embed giá trị
        x_embed = self.token_embedding(x)  # [B, T, D]
        
        # 2. Tính toán volatility từ close price
        volatility = self.volatility_embedding(x[:, :, -1:])  # [B, T, D]
        
        # 3. Time embedding (nếu có)
        if x_mark is not None:
            time_embed = self.time_embedding(x_mark)  # [B, T, D]
            # Đảm bảo time_embed có cùng kích thước
            if time_embed.dim() == 2:
                time_embed = time_embed.unsqueeze(1).expand_as(x_embed)
            elif time_embed.shape[1] != x_embed.shape[1]:
                time_embed = time_embed[:, :x_embed.shape[1], :]
        else:
            time_embed = 0
        
        # 4. Positional embedding
        pos_embed = self.position_embedding(x)  # [1, T, D]
        if pos_embed.shape[1] != x_embed.shape[1]:
            pos_embed = pos_embed[:, :x_embed.shape[1], :]
        
        # 5. Tổng hợp với gate điều chỉnh
        gate = torch.sigmoid(self.volatility_gate(volatility))
        out = (x_embed + time_embed + pos_embed) * gate + volatility
        
        return self.dropout(out)
