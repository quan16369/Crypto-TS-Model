import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        
        # 1. Input Projection - tự động điều chỉnh input size
        self.input_net = nn.Sequential(
            nn.Linear(len(config['data']['feature_names']), self.config['d_model']),
            nn.GELU(),
            nn.LayerNorm(self.config['d_model']),
            nn.Dropout(self.config.get('dropout', 0.3))
        )
        # 2. LSTM 
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.3) if self.config['e_layers'] > 1 else 0)
        
        # 3. Output Network - đảm bảo output đúng pred_len
        self.output_net = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config['d_model']),
            nn.SiLU(),
            nn.Linear(self.config['d_model'], self.config['pred_len']))

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        batch_size = x_enc.size(0)
        
        # 1. Feature projection
        x = x_enc.reshape(-1, x_enc.size(-1))  # [batch*seq_len, num_features]
        x = self.input_net(x)
        x = x.reshape(batch_size, -1, self.config['d_model'])  # [batch, seq_len, d_model]
        
        # 2. LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # 3. Get last hidden state  
        last_hidden = lstm_out[:, -1, :]  # Shape: [batch_size, d_model]
        
        # 4. Prediction - đảm bảo output shape [batch_size, pred_len, 1]
        pred = self.output_net(last_hidden)  # Shape: [batch_size, pred_len]
        return pred.unsqueeze(-1)  # Shape: [batch_size, pred_len, 1]
