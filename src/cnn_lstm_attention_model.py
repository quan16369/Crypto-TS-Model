import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EnhancedCNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.InstanceNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(d_model, d_model, kernel_size=3, dilation=2, padding=2),
            nn.InstanceNorm1d(d_model),
            nn.GELU()
        )
        self.res_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

    def forward(self, x):
        # Input: [B, T, D]
        res = self.res_proj(x)
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.conv_blocks(x)
        return x.permute(0, 2, 1) + res  # [B, T, D]

class LSTMCNNAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        self.d_model = model_cfg['d_model']
        self.enc_in = model_cfg['enc_in']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        self.dropout_rate = model_cfg.get('dropout', 0.3)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate)
        )
        # CNN Feature Extractor
        self.cnn = EnhancedCNNFeatureExtractor(self.d_model, self.d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate if 2 > 1 else 0
        )
        
        # Attention Mechanism
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=model_cfg.get('n_heads', 4),
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(self.d_model)
        
        # Output Projection (giữ nguyên structure gốc)
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.d_model * 2, self.pred_len * self.out_dim)
        )

    def forward(self, x, time_features=None):
        B, T, _ = x.shape
        
        # Input processing
        x = self.input_proj(x)
        x = self.cnn(x)
        x = self.pos_encoder(x)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Causal attention
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_out, _ = self.self_attn(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            attn_mask=mask
        )
        attn_out = self.attn_norm(lstm_out + attn_out)
        
        # Output [B, pred_len, out_dim]
        out = self.output_proj(attn_out.mean(dim=1))
        return out.view(B, self.pred_len, self.out_dim)
