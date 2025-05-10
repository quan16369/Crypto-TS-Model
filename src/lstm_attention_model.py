import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class LSTMAttentionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        
        # Input projection layer với BatchNorm
        self.input_proj = nn.Sequential(
            nn.Linear(self.config['enc_in'], self.config['d_model']),
            nn.BatchNorm1d(self.config['d_model']),
            nn.Dropout(self.config.get('dropout', 0.2))
        )
        
        # Bidirectional LSTM với 2 layers
        self.bilstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model']//2,  # Chia 2 do dùng bidirectional
            num_layers=2,  
            batch_first=True,
            bidirectional=True,
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
            nn.BatchNorm1d(self.config['d_model']),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.2)),
            nn.Linear(self.config['d_model'], self.config['c_out'])
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x.reshape(-1, x.size(-1))).reshape(x.size(0), -1, self.config['d_model'])
        
        # BiLSTM processing
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, d_model]
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, d_model]
        
        # Output prediction
        pred = self.output(context)  # [batch, c_out]
        
        return pred.unsqueeze(1)  # [batch, 1, c_out] để phù hợp với pred_len
