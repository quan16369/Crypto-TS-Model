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

class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, d_model, dilation_rates=[1, 2, 3]):
        super().__init__()
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, d_model//len(dilation_rates), 
                         kernel_size=3, 
                         dilation=d,
                         padding=d),  # Same padding
                nn.GELU(),
                nn.InstanceNorm1d(d_model//len(dilation_rates)),
                nn.Dropout(0.2)
            ) for d in dilation_rates
        ])
        self.res_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()

    def forward(self, x):
        # Input: [B, T, D]
        x_res = self.res_proj(x)
        x = x.permute(0, 2, 1)  # [B, D, T]
        
        branch_outputs = []
        for conv in self.conv_branches:
            out = conv(x)  # [B, D//K, T]
            branch_outputs.append(out)
        
        # Concatenate along channel dimension
        out = torch.cat(branch_outputs, dim=1)  # [B, D, T]
        out = out.permute(0, 2, 1)  # [B, T, D]
        
        # Residual connection
        return out + x_res

class LSTMCNNAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']
        self.d_model = model_cfg['d_model']
        self.enc_in = model_cfg['enc_in']
        self.pred_len = model_cfg['pred_len']
        self.out_dim = model_cfg.get('output_dim', 1)
        
        # Enhanced components
        self.input_proj = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        self.cnn = MultiScaleCNN(
            input_dim=self.d_model,
            d_model=self.d_model,
            dilation_rates=model_cfg.get('dilation_rates', [1, 2, 3])
        )
        
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model//2,  # Half for bidirectional
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention with head scaling
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=model_cfg.get('n_heads', 4),
            dropout=0.2,
            batch_first=True
        )
        
        # Gated residual
        self.gate = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )
        
        # Output with uncertainty estimation
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.SiLU(),
            nn.Linear(self.d_model * 2, self.pred_len * self.out_dim * 2),  # 2x for mean/std
        )

    def forward(self, x):
        B, T, _ = x.shape
        
        # Input processing
        x = self.input_proj(x)
        x = self.cnn(x)
        x = self.pos_encoder(x)
        
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # [B, T, D]
        
        # Causal attention
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
        attn_out, _ = self.self_attn(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            attn_mask=mask
        )
        
        # Gated residual
        gate = self.gate(attn_out)
        context = gate * attn_out + (1 - gate) * lstm_out
        
        # Predictive uncertainty
        out = self.output_proj(context.mean(dim=1))
        mean, log_var = out.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        
        return mean.view(B, self.pred_len, self.out_dim), std.view(B, self.pred_len, self.out_dim)
