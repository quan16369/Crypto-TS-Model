import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        
        # 1. Input Projection
        self.input_net = nn.Sequential(
            nn.Linear(config['enc_in'], config['d_model']*2),
            nn.GELU(),
            nn.LayerNorm(config['d_model']*2),
            nn.Dropout(config.get('dropout', 0.2)),
            nn.Linear(config['d_model']*2, config['d_model'])
        )
        
        # 2. LSTM 
        self.lstm = nn.LSTM(
            input_size=config['d_model'],
            hidden_size=config['d_model'],
            num_layers=config['e_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.2) if config['e_layers'] > 1 else 0,
            proj_size=config['d_model']//2 if config['e_layers'] > 2 else None
        )
        
        # 3. Time Features Integration
        if config.get('time_features', 0) > 0:
            self.time_net = nn.Sequential(
                nn.Linear(config['time_features'], config['d_model']),
                nn.SiLU(),
                nn.LayerNorm(config['d_model'])
            )
        
        # 4. Output Network
        self.output_net = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']//2),
            nn.SiLU(),
            nn.Linear(config['d_model']//2, config['c_out']),
            nn.Unflatten(1, (config['pred_len'], config['c_out']))
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
