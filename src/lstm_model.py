import torch
import torch.nn as nn
from typing import Dict, Any
from model import ModelConfig
from embed import CryptoDataEmbedding

class LSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.configs = ModelConfig(config)
        self.device = torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Embedding layer
        self.embedding = CryptoDataEmbedding(
            c_in=self.configs.enc_in,
            d_model=self.configs.d_model,
            patch_size=self.configs.patch_size,
            lookback=config['model'].get('volatility_lookback', 11),
            dropout=self.configs.dropout
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.configs.d_model,
            hidden_size=self.configs.d_model,
            num_layers=self.configs.e_layers,
            batch_first=True,
            dropout=self.configs.dropout if self.configs.e_layers > 1 else 0
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(self.configs.d_model, self.configs.d_ff),
            nn.ReLU(),
            nn.Linear(self.configs.d_ff, self.configs.c_out * self.configs.pred_len),
            nn.Unflatten(-1, (self.configs.pred_len, self.configs.c_out))
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # Embedding
        x = self.embedding(x_enc, x_mark_enc)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Only use the last time step's output for prediction
        last_output = lstm_out[:, -self.configs.pred_len:, :]
        
        # Prediction
        pred = self.predictor(last_output)
        
        return pred