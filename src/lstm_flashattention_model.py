import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import xformers.ops as xops  # FlashAttention API // install: pip install xformers

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


class FlashAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.d_model = d_model

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # [B, T, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, num_heads, T, head_dim]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D//H]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Add causal mask (lower triangular)
        attn_mask = xops.LowerTriangularMask()

        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_mask)  # FlashAttention with causal masking
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)


class LSTMFlashAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        d_model = model_cfg['d_model']
        enc_in = model_cfg['enc_in']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        self.dropout_rate = model_cfg.get('dropout', 0.3)

        input_dim = enc_in

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
            num_layers=4,
            batch_first=True,
            dropout=self.dropout_rate
        )

        self.attn = FlashAttentionBlock(d_model, num_heads=4)

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model, self.pred_len * self.out_dim)
        )

    def attention_weighted_pooling(self, x):
        attn_weights = F.softmax(x.mean(dim=-1), dim=1)  # [B, T]
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        return pooled

    def forward(self, x, time_features=None):
        x = self.input_proj(x)             # [B, T, d_model]
        x = self.pos_encoder(x)            # [B, T, d_model]
        lstm_out, _ = self.lstm(x)         # [B, T, d_model]

        attn_out = self.attn(lstm_out)     # [B, T, d_model]
        context = self.attention_weighted_pooling(attn_out)   # [B, d_model]

        out = self.output_proj(context)    # [B, pred_len * out_dim]
        return out.view(-1, self.pred_len, self.out_dim)  # [B, pred_len, out_dim]
