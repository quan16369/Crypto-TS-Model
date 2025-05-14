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
            nn.Dropout(self.config.get('dropout', 0.3))
        )   
        
        # 2. LSTM 
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.3) if self.config['e_layers'] > 1 else 0
        )
        
        # 3. Output Network 
        self.output_net = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config['d_model']),
            nn.SiLU(),
            nn.Linear(self.config['d_model'], self.config['pred_len']),
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # Get original shape
        batch_size, seq_len, _ = x_enc.shape
        
        # 1. Feature projection
        x = x_enc.reshape(-1, x_enc.size(-1))  # [batch*seq_len, num_features]
        x = self.input_net(x)
        x = x.reshape(batch_size, seq_len, -1)  # [batch, seq_len, d_model]
        
        # 2. LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # 3. Prediction for each time step in pred_len
        preds = []
        for i in range(self.config['pred_len']):
            # Use the last hidden state for each prediction step
            pred = self.output_net(lstm_out[:, -1, :])
            preds.append(pred.unsqueeze(1))
        
        # Stack predictions along time dimension
        pred = torch.cat(preds, dim=1)  # [batch_size, pred_len, 1]
        
        return pred
