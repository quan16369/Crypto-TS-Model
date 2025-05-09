import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']  # Lấy phần config model
        
        # 1. Input Projection
        self.input_net = nn.Sequential(
            nn.Linear(self.config['enc_in'], self.config['d_model']*2),
            nn.GELU(),
            nn.LayerNorm(self.config['d_model']*2),
            nn.Dropout(self.config.get('dropout', 0.2)),
            nn.Linear(self.config['d_model']*2, self.config['d_model'])
        )
        
        # 2. LSTM 
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.2) if self.config['e_layers'] > 1 else 0,
            proj_size=self.config['d_model']//2 if self.config['e_layers'] > 2 else None
        )
        
        # 3. Time Features Integration
        if self.config.get('time_features', 0) > 0:
            self.time_net = nn.Sequential(
                nn.Linear(self.config['time_features'], self.config['d_model']),
                nn.SiLU(),
                nn.LayerNorm(self.config['d_model'])
            )
        
        # 4. Output Network
        self.output_net = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config['d_model']//2),
            nn.SiLU(),
            nn.Linear(self.config['d_model']//2, self.config['c_out']),
            nn.Unflatten(1, (self.config['pred_len'], self.config['c_out']))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # 1. Feature projection
        x = self.input_net(x_enc)
        
        # 2. Time features fusion
        if x_mark_enc is not None and hasattr(self, 'time_net'):
            x = x + self.time_net(x_mark_enc)
        
        # 3. LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # 4. Get last pred_len steps
        last_steps = lstm_out[:, -self.config['pred_len']:, :]
        
        # 5. Final prediction
        return self.output_net(last_steps)
