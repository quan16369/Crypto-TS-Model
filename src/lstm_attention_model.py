import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class LSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(self.config['enc_in'], self.config['d_model']),
            nn.BatchNorm1d(self.config['d_model']),
            nn.Dropout(self.config.get('dropout', 0.2))
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=2,
            batch_first=True,
            dropout=self.config.get('dropout', 0.2)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config['d_model']),
            nn.Tanh(),
            nn.Linear(self.config['d_model'], 1, bias=False)
        )
        
        # Output layers 
        self.output = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config['d_model']),
            nn.ReLU(),
            nn.Linear(self.config['d_model'], self.config['pred_len'])  # Output pred_len steps
        )

    def forward(self, x, time_features=None):
        # Input shape: [batch, seq_len, enc_in]
        B, T, _ = x.shape
        
        # Project input
        x = self.input_proj(x.reshape(-1, x.size(-1))).reshape(B, T, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, d_model]
        
        # Attention
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, d_model]
        
        # Output projection - đảm bảo output [batch, pred_len]
        pred = self.output(context).unsqueeze(-1)  # [batch, pred_len, 1]
        
        return pred
