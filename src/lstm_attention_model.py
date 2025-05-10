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
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x


class LSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        self.dropout_rate = self.config.get('dropout', 0.3)
        d_model = self.config['d_model']
        enc_in = self.config['enc_in']

        # Optional time feature size
        self.time_feat_dim = self.config.get('time_feat_dim', 0)
        input_dim = enc_in + self.time_feat_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(self.dropout_rate),
            nn.GELU()
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=4,
            batch_first=True,
            dropout=self.dropout_rate
        )

        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model, self.config['pred_len'])
        )

    def forward(self, x, time_features=None):
        # x: [B, T, enc_in], time_features: [B, T, time_feat_dim]
        B, T, _ = x.shape

        if time_features is not None:
            x = torch.cat([x, time_features], dim=-1)  # [B, T, enc_in + time_feat_dim]

        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_encoder(x)  # [B, T, d_model]

        lstm_out, _ = self.lstm(x)  # [B, T, d_model]

        # Self-attention
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # [B, T, d_model]

        # Context vector via mean pooling (or attn-weighted sum)
        context = torch.mean(attn_out, dim=1)  # [B, d_model]

        out = self.output(context)  # [B, pred_len]
        return out.unsqueeze(-1)  # [B, pred_len, 1]
