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
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(input_dim, d_model//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model//2),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        # x: [B, T, D] -> [B, D, T] cho Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)
        return x.permute(0, 2, 1)  # Trả về [B, T', D']

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

        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(enc_in, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout_rate
        )
        
        # Multihead Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Residual Connections
        self.residual1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(self.dropout_rate)
        )
        self.residual2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(self.dropout_rate))
        
        # Output Projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(d_model*2, self.pred_len * self.out_dim)
        )

    def attention_weighted_pooling(self, x):
        attn_weights = F.softmax(x.mean(dim=-1), dim=1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_rate, training=self.training)
        return torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
    
    def forward(self, x, time_features=None):
        B, T, _ = x.shape
        
        # 1. CNN Feature Extraction
        x = self.cnn(x)  # [B, T', d_model]
        
        # 2. Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. LSTM Processing
        lstm_out, _ = self.lstm(x)
        
        # 4. Multihead Attention với Causal Mask
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        attn_out, _ = self.self_attn(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            attn_mask=mask
        )
        attn_out = self.attn_norm(lstm_out + attn_out)  # Add & Norm
        
        # 5. Context Pooling với Residual
        context = self.attention_weighted_pooling(attn_out)
        context = context + self.residual1(context)
        
        # 6. Output Projection
        out = self.output_proj(context)
        return out.view(-1, self.pred_len, self.out_dim)
