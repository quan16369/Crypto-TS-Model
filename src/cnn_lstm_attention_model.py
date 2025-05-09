import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
from embed import CryptoDataEmbedding

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

class LSTMWithCNNAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        self.device = torch.device(config['training']['device'])
        
        # Embedding
        self.embedding = CryptoDataEmbedding(
            c_in=self.config['enc_in'],
            d_model=self.config['d_model'],
            patch_size=self.config.get('patch_size', 16),
            lookback=self.config.get('volatility_lookback', 11),
            dropout=self.config.get('dropout', 0.1)
        )
        
        # CNN Feature Extraction 
        self.conv1d = nn.Conv1d(in_channels=self.config['d_model'], 
                                out_channels=self.config['d_model'], 
                                kernel_size=3, padding=1)  # Convolutional layer to capture temporal patterns
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling to downsample the sequence
        
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
        # Embedding
        x = self.embedding(x_enc, x_mark_enc)
        
        # Add time features if available
        if x_mark_enc is not None and hasattr(self, 'time_embed'):
            x = x + self.time_embed(x_mark_enc)  # Add time embeddings to features
        
        # Apply CNN to extract temporal features
        x = self.conv1d(x.permute(0, 2, 1))  # Change shape for CNN input [B, C, L]
        x = self.max_pool(x)  # Apply max pooling to downsample
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Attention
        attn_out = self.attention(lstm_out)
        
        # Prediction
        pred = self.predictor(attn_out[:, -self.config['pred_len']:, :])
        return pred