import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from model import ModelConfig
from embed import CryptoDataEmbedding

class AttentionLayer(nn.Module):
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
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear layer
        output = self.out_linear(output)
        
        return output

class LSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.configs = ModelConfig(config)
        self.device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Embedding layer
        self.embedding = CryptoDataEmbedding(
            c_in=self.configs.enc_in,
            d_model=self.configs.d_model,
            patch_size=self.configs.patch_size,
            lookback=config['model'].get('volatility_lookback', 11),
            dropout=self.configs.dropout
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.configs.d_model,
            hidden_size=self.configs.d_model,
            num_layers=self.configs.e_layers,
            batch_first=True,
            dropout=self.configs.dropout if self.configs.e_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            d_model=self.configs.d_model,
            n_heads=self.configs.n_heads
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.configs.d_model, self.configs.d_ff),
            nn.ReLU(),
            nn.Linear(self.configs.d_ff, self.configs.c_out * self.configs.pred_len),
            nn.Unflatten(-1, (self.configs.pred_len, self.configs.c_out))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # Embedding
        x = self.embedding(x_enc, x_mark_enc)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out = self.attention(lstm_out)
        
        # Only use the last time step's output for prediction
        last_output = attn_out[:, -self.configs.pred_len:, :]
        
        # Prediction
        pred = self.predictor(last_output)
        
        return pred