import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.out_linear(output)

class LSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        self.device = torch.device(config['training']['device'])
        
        # Input projection (thay thế cho embedding)
        self.input_proj = nn.Sequential(
            nn.Linear(self.config['enc_in'], self.config['d_model']),
            nn.LayerNorm(self.config['d_model']),
            nn.Dropout(self.config.get('dropout', 0.1))
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.1) if self.config['e_layers'] > 1 else 0
        )
        
        # Attention
        self.attention = MultiHeadAttention(
            d_model=self.config['d_model'],
            n_heads=self.config['n_heads']
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config.get('d_ff', 512)),
            nn.GELU(),
            nn.Dropout(self.config.get('dropout', 0.1)),
            nn.Linear(self.config.get('d_ff', 512), self.config['c_out'] * self.config['pred_len']),
            nn.Unflatten(-1, (self.config['pred_len'], self.config['c_out']))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # 1. Project input features
        x = self.input_proj(x_enc)
        
        # 2. LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # 3. Attention
        attn_out = self.attention(lstm_out)
        
        # 4. Prediction
        pred = self.predictor(attn_out[:, -self.config['pred_len']:, :])
        return pred
