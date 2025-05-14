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
        return x + self.pe[:, :x.size(1)].to(x.device)

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, D, T]
        x = self.conv(x)
        return x.permute(0, 2, 1)  # [B, T, D]

class LSTMCNNAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        d_model = model_cfg['d_model']
        enc_in = model_cfg['enc_in']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        self.dropout_rate = model_cfg.get('dropout', 0.3)
        self.num_heads = model_cfg.get('n_heads', 4)

        self.input_proj = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        self.cnn = CNNFeatureExtractor(d_model, d_model)
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
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)

        self.residual = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(self.dropout_rate)
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model * 2, self.pred_len * self.out_dim)
        )

    def forward(self, x, time_features=None):
        B, T, _ = x.shape

        x = self.input_proj(x)
        x = self.cnn(x)
        x = self.pos_encoder(x)

        lstm_out, _ = self.lstm(x)

        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out, attn_mask=mask)
        attn_out = self.attn_norm(lstm_out + attn_out)

        context = attn_out.mean(dim=1) + self.residual(attn_out.mean(dim=1))
        out = self.output_proj(context)
        return out.view(B, self.pred_len, self.out_dim)
