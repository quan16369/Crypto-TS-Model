import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class LightweightHybridNorm(nn.Module):
    """Phiên bản norm nhẹ cho tốc độ"""
    def __init__(self, d_model):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, d_model))
        self.beta = nn.Parameter(torch.zeros(1, 1, d_model))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + 1e-6) + self.beta

class EfficientTemporalBlock(nn.Module):
    """Khối xử lý thời gian hiệu suất cao"""
    def __init__(self, d_model, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=dilation, dilation=dilation, groups=4),
            nn.GELU(),
            LightweightHybridNorm(d_model)
        )
        self.lstm_cell = nn.LSTMCell(d_model, d_model//2)
    
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        conv_out = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Lightweight LSTM
        h = torch.zeros(B, D//2, device=x.device)
        c = torch.zeros(B, D//2, device=x.device)
        lstm_out = []
        for t in range(T):
            h, c = self.lstm_cell(x[:, t, :], (h, c))
            lstm_out.append(h)
        lstm_out = torch.stack(lstm_out, dim=1)
        
        return conv_out + F.pad(lstm_out, (0, D//2))  # Residual

class OptimizedAttention(nn.Module):
    """Attention tốc độ cao"""
    def __init__(self, d_model):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model*3)
        self.proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(attn)

class OptimizedLSTMCNNAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = config['model']
        self.d_model = cfg['d_model']
        
        # Input projection
        self.input_net = nn.Sequential(
            nn.Linear(cfg['enc_in'], self.d_model),
            LightweightHybridNorm(self.d_model),
            nn.GELU()
        )
        
        # Temporal processing
        self.blocks = nn.Sequential(
            *[EfficientTemporalBlock(self.d_model, d) for d in [1, 2, 4]]
        )
        
        # Attention
        self.attn = OptimizedAttention(self.d_model)
        
        # Output
        self.output_net = nn.Linear(self.d_model, cfg['pred_len'] * cfg.get('output_dim', 1))

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_net(x)
        
        # Parallel temporal processing
        x = self.blocks(x)
        
        # Efficient attention
        x = self.attn(x)
        
        # Output
        return self.output_net(x.mean(dim=1)).view(B, -1, cfg.get('output_dim', 1))
