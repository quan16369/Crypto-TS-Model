import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any

class LightweightHybridNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
        self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
    
    def forward(self, x):
        # Ensure input matches expected dimension
        if x.size(-1) != self.d_model:
            x = x[:, :, :self.d_model]  # Simple truncation
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta

class EfficientTemporalBlock(nn.Module):
    def __init__(self, d_model, dilation, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # Conv branch with proper padding to maintain sequence length
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, 
                     padding=dilation, dilation=dilation, 
                     groups=4),
            nn.GELU(),
            nn.Dropout(dropout),
            LightweightHybridNorm(d_model)
        )
        # LSTM branch that maintains sequence length
        self.lstm = nn.LSTM(d_model, d_model, 
                           batch_first=True,
                           dropout=dropout if dropout > 0 else 0)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, D = x.shape
        # Conv branch [B, T, D]
        conv_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # LSTM branch [B, T, D]
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        
        # Ensure both branches have same length
        min_length = min(conv_out.size(1), lstm_out.size(1))
        return conv_out[:, :min_length, :] + lstm_out[:, :min_length, :]

class OptimizedAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.d_model)
        return self.proj(out)

class OptimizedLSTMCNNAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = config['model']
        self.seq_len = cfg['seq_len']
        self.pred_len = cfg['pred_len']
        self.d_model = cfg['d_model']
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(cfg['enc_in'], self.d_model),
            LightweightHybridNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(cfg['dropout'])
        )
        
        # Temporal processing
        self.blocks = nn.Sequential(
            *[EfficientTemporalBlock(self.d_model, 2**i, cfg['dropout']) 
              for i in range(cfg['e_layers'])]
        )
        
        # Attention
        self.attn = OptimizedAttention(self.d_model, cfg['n_heads'], cfg['dropout'])
        
        # Output
        self.output_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            nn.GELU(),
            nn.Linear(self.d_model//2, self.pred_len * cfg['c_out']),
        )

    def forward(self, x):
        # Input: [B, seq_len, enc_in]
        x = self.input_proj(x)  # [B, seq_len, d_model]
        x = self.blocks(x)      # [B, seq_len, d_model]
        x = self.attn(x)        # [B, seq_len, d_model]
        
        # Use last 'pred_len' timesteps for prediction
        x = x[:, -self.pred_len:, :]  # [B, pred_len, d_model]
        x = x.mean(dim=1)             # [B, d_model]
        
        x = self.output_net(x)        # [B, pred_len * c_out]
        return x.view(x.size(0), self.pred_len, -1)  # [B, pred_len, c_out]
