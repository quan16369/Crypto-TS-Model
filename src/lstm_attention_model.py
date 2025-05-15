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
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        d_model = model_cfg['d_model']
        enc_in = model_cfg['enc_in']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        self.dropout_rate = model_cfg.get('dropout', 0.3)

        input_dim = enc_in 

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(self.dropout_rate),
            nn.GELU()
        )

        self.pos_encoder = PositionalEncoding(d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True
        )
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.lstm_norm = nn.LayerNorm(d_model)
        self.residual_norm = nn.LayerNorm(d_model)
        self.pool_norm = nn.LayerNorm(d_model)
        
        # residual connection
        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(self.dropout_rate)
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.LayerNorm(d_model*2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model*2, self.pred_len * self.out_dim)
        )

    def attention_weighted_pooling(self, x):
        attn_weights = F.softmax(x.mean(dim=-1), dim=1)  # [B, T]
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled
    
    def forward(self, x, time_features=None):
        B, T, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # LSTM + Norm
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)  # add norm
        
        # Attention
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out, attn_mask=mask)
        attn_out = self.attn_norm(lstm_out + attn_out)
        
        # Pooling + Norm
        # attn_out = self.pool_norm(attn_out)  # add norm before pooling
        context = self.attention_weighted_pooling(attn_out)
        
        # Residual + Norm
        context = context + self.residual(context)
        context = self.residual_norm(context)  # add norm
        
        # Output
        out = self.output_proj(context)
        return out.view(-1, self.pred_len, self.out_dim)
