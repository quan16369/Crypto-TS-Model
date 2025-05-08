import torch
import torch.nn as nn
from typing import Dict, Any
from embed import CryptoDataEmbedding

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config['model']
        self.device = torch.device(config['training']['device'])
        
        # Embedding layer 
        self.embedding = CryptoDataEmbedding(
            c_in=self.config['enc_in'],
            d_model=self.config['d_model'],
            patch_size=self.config.get('patch_size', 16),
            lookback=self.config.get('volatility_lookback', 11),
            dropout=self.config.get('dropout', 0.1)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],
            hidden_size=self.config['d_model'],
            num_layers=self.config['e_layers'],
            batch_first=True,
            dropout=self.config.get('dropout', 0.1) if self.config['e_layers'] > 1 else 0,
            bidirectional=False
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config.get('d_ff', 512)),
            nn.ReLU(),
            nn.Dropout(self.config.get('dropout', 0.1)),
            nn.Linear(self.config.get('d_ff', 512), self.config['c_out'] * self.config['pred_len']),
            nn.Unflatten(-1, (self.config['pred_len'], self.config['c_out']))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # Embedding
        x = self.embedding(x_enc, x_mark_enc)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Prediction
        pred = self.predictor(lstm_out[:, -self.config['pred_len']:, :])
        return pred
