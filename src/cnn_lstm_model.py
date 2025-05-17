import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class CNNLSTMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_cfg = config['model']

        self.input_dim = model_cfg['enc_in']
        self.seq_len = model_cfg['seq_len']
        self.pred_len = model_cfg['pred_len']
        self.cnn_out_channels = model_cfg.get('cnn_out_channels', 64)
        self.lstm_hidden_dim = model_cfg.get('d_model', 128)
        self.dropout = model_cfg.get('dropout', 0.3)

        # CNN module
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.cnn_out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(self.dropout)
        )

        # LSTM module
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_channels,
            hidden_size=self.lstm_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim),
            nn.LayerNorm(self.lstm_hidden_dim),
            nn.GELU(),
            nn.Linear(self.lstm_hidden_dim, self.pred_len)
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc=None):
        x = x_enc.unsqueeze(1)                     # [batch, 1, seq_len, input_dim]
        x = self.cnn(x)                            # [batch, cnn_out, seq_len//2, 1]
        print(x.shape)
        x = x.squeeze(-1).permute(0, 2, 1)         # [batch, seq_len//2, cnn_out]
        lstm_out, _ = self.lstm(x)                 # [batch, seq_len//2, lstm_hidden_dim]
        last_output = lstm_out[:, -1, :]           # [batch, lstm_hidden_dim]
        out = self.output_layer(last_output)       # [batch, pred_len]
        return out.unsqueeze(-1)                   # [batch, pred_len, 1]
