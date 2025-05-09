import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        
        # 1. Input Projection
        self.input_net = nn.Sequential(
            nn.Linear(self.config['enc_in'], self.config['d_model']),
            nn.GELU(),
            nn.LayerNorm(self.config['d_model']),
            nn.Dropout(self.config.get('dropout', 0.2))
        )
        
        # 2. LSTM 
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.2) if self.config['e_layers'] > 1 else 0
        )
        
        # 3. Output Network - Đảm bảo đầu ra có kích thước pred_len
        self.output_net = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config['d_model']),
            nn.SiLU(),
            nn.Linear(self.config['d_model'], self.config['pred_len'] * self.config['c_out']),
            nn.Unflatten(-1, (self.config['pred_len'], self.config['c_out']))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # 1. Feature projection
        x = self.input_net(x_enc)
        
        # 2. LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # 3. Lấy hidden state cuối cùng
        last_hidden = lstm_out[:, -1, :]  # Shape: [batch_size, d_model]
        
        # 4. Dự đoán
        pred = self.output_net(last_hidden)  # Shape: [batch_size, pred_len, c_out]
        
        return pred.squeeze(-1)  # Loại bỏ chiều cuối nếu c_out=1
