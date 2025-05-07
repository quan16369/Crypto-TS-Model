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
    def __init__(self, d_model, lookback=10):
        super().__init__()
        self.lookback = lookback
        self.proj = nn.Linear(1, d_model)
        
    def forward(self, x_close):  # x_close: [B, T, 1]
        B, T, _ = x_close.shape
        
        # Tính returns với padding ban đầu
        returns = F.pad(x_close.diff(dim=1).abs(), (0,0,1,0), value=0)  # [B, T, 1]
        
        # Tính volatility với padding đối xứng
        volatility = returns.unfold(1, self.lookback, 1).std(dim=-1, keepdim=True)  # [B, T-lookback+1, 1]
        
        # Padding để đảm bảo output có shape [B, T, 1]
        pad_front = self.lookback // 2
        pad_back = self.lookback - pad_front - 1
        volatility = F.pad(volatility, (0,0,pad_front,pad_back), value=0)
        
        # Project lên d_model
        return self.proj(volatility)  # [B, T, d_model]
    
class CryptoTokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        self.projection = nn.Linear(c_in * 16, d_model)  # Project to d_model
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular'),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, padding_mode='circular')
        )
        
    def forward(self, x):
        B, NP, _ = x.shape
        # Project to d_model dimension first
        x = self.projection(x)  # [B, NP, d_model]
        x = x.permute(0, 2, 1)  # [B, d_model, NP]
        x = self.conv(x)  # [B, d_model, NP]
        return x.permute(0, 2, 1)  # [B, NP, d_model]

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
        B, T, _ = x.shape
        
        # 1. Token embedding
        x_embed = self.token_embedding(x)  # [B, T, D]
        
        # 2. Volatility embedding
        volatility = self.volatility_embedding(x[:, :, -1:])  # [B, T, D]
        
        # 3. Time embedding - lấy mẫu tương ứng patches
        if x_mark is not None:
            # Tính indices cho các patches
            patch_indices = torch.linspace(0, x_mark.size(1)-1, T).long()
            time_embed = self.time_embedding(x_mark[:, patch_indices, :])  # [B, T, D]
        else:
            time_embed = 0
        
        # 4. Positional embedding
        pos_embed = self.position_embedding(x)[:, :T, :]  # [1, T, D]
        
        # 5. Kiểm tra kích thước
        print(f"Final shapes - x_embed: {x_embed.shape}, volatility: {volatility.shape}, "
            f"time_embed: {time_embed.shape if isinstance(time_embed, torch.Tensor) else 'scalar'}, "
            f"pos_embed: {pos_embed.shape}")
        
        # 6. Tính toán output
        gate = torch.sigmoid(self.volatility_gate(volatility))  # [B, T, D]
        out = (x_embed + time_embed + pos_embed) * gate + volatility
        
        return self.dropout(out)
