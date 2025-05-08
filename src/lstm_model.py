import torch
import torch.nn as nn
from typing import Dict, Any

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        self.device = torch.device(config['training']['device'])
        
        # === Feature Projection ===
        self.input_proj = nn.Sequential(
            nn.Linear(self.config['enc_in'], self.config['d_model']),
            nn.LayerNorm(self.config['d_model']),
            nn.Dropout(self.config.get('dropout', 0.1))
        )
        
        # === LSTM Core ===
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.1) if self.config['e_layers'] > 1 else 0
        )
        
        # === Time Features Handling ===
        if self.config.get('time_features', 0) > 0:
            self.time_embed = nn.Linear(self.config['time_features'], self.config['d_model'])
        
        # === Output Projection ===
        self.predictor = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config.get('d_ff', 512)),
            nn.ReLU(),
            nn.Linear(self.config.get('d_ff', 512), self.config['c_out'] * self.config['pred_len']),
            nn.Unflatten(-1, (self.config['pred_len'], self.config['c_out']))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # Input shape: [B, L, C]
        
        # 1. Project input features
        x = self.input_proj(x_enc)
        
        # 2. Add time features if available
        if x_mark_enc is not None and hasattr(self, 'time_embed'):
            x = x + self.time_embed(x_mark_enc)
        
        # 3. LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # 4. Get predictions
        pred = self.predictor(lstm_out[:, -self.config['pred_len']:, :])
        return pred
