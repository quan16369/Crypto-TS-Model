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
        div_term = torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model).exp()
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
    """Phiên bản mở rộng của TokenEmbedding cho dữ liệu OHLCV"""
    def __init__(self, c_in, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, d_model, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular')
        )
        
    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class CryptoTimeEmbedding(nn.Module):
    """Tối ưu cho time features trong trading (phút/giờ)"""
    def __init__(self, d_model):
        super().__init__()
        self.minute_embed = nn.Embedding(60, d_model)  # 60 phút
        self.hour_embed = nn.Embedding(24, d_model)   # 24 giờ
        
    def forward(self, x_mark):
        minute_x = self.minute_embed(x_mark[..., 4].long())  # Phút
        hour_x = self.hour_embed(x_mark[..., 3].long())     # Giờ
        return minute_x + hour_x

class CryptoDataEmbedding(nn.Module):
    """Embedding tổng hợp tối ưu cho crypto"""
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        # 1. Embedding giá OHLCV
        self.token_embedding = CryptoTokenEmbedding(c_in, d_model)
        
        # 2. Embedding biến động (volatility)
        self.volatility_embedding = VolatilityEmbedding(d_model)
        
        # 3. Time embedding (tối giản)
        self.time_embedding = CryptoTimeEmbedding(d_model)
        
        # 4. Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # 5. Dropout & Gate điều chỉnh theo volatility
        self.dropout = nn.Dropout(dropout)
        self.volatility_gate = nn.Linear(d_model, d_model)

    def forward(self, x, x_mark=None):
        # x: [B, T, C] (OHLCV), x_mark: [B, T, time_features]
        # 1. Embed giá trị
        x_embed = self.token_embedding(x)  # [B, T, D]
        
        # 2. Tính toán volatility từ close price (cột cuối)
        volatility = self.volatility_embedding(x[:, :, -1:])  # [B, T, D]
        
        # 3. Time embedding (nếu có)
        time_embed = self.time_embedding(x_mark) if x_mark is not None else 0
        
        # 4. Positional embedding
        pos_embed = self.position_embedding(x)
        
        # 5. Tổng hợp với gate điều chỉnh
        gate = torch.sigmoid(self.volatility_gate(volatility))
        out = (x_embed + time_embed + pos_embed) * gate + volatility
        
        return self.dropout(out)