import torch
import torch.nn as nn
from typing import Dict, Any

class CNNLSTMModel(nn.Module):
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
        
        # === CNN Feature Extraction ===
        self.conv1d = nn.Conv1d(in_channels=self.config['d_model'], 
                                out_channels=self.config['d_model'], 
                                kernel_size=3, padding=1)  # Convolutional layer to capture temporal patterns
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Max pooling to downsample the sequence
        
        # === LSTM Core ===
        self.lstm = nn.LSTM(
            input_size=self.config['d_model'],  # LSTM input size is the output size of previous layers
            hidden_size=self.config['d_model'],  # LSTM hidden state size
            num_layers=self.config['e_layers'],  # Number of LSTM layers
            batch_first=True,  # Ensure correct input shape [batch_size, seq_len, features]
            dropout=self.config.get('dropout', 0.1) if self.config['e_layers'] > 1 else 0  # Dropout between LSTM layers
        )
        
        # === Time Features Handling ===
        if self.config.get('time_features', 0) > 0:
            self.time_embed = nn.Linear(self.config['time_features'], self.config['d_model'])  # Embed time features
        
        # === Output Projection ===
        self.predictor = nn.Sequential(
            nn.Linear(self.config['d_model'], self.config.get('d_ff', 512)),  # Project to d_ff dimensions
            nn.ReLU(),  # Apply ReLU activation
            nn.Linear(self.config.get('d_ff', 512), self.config['c_out']),  # Output size (c_out is the number of target features)
            nn.Unflatten(1, (self.config['pred_len'], self.config['c_out']))  # Reshape output to [B, pred_len, c_out]
        )
        
    def forward(self, x_enc: torch.Tensor, x_mark_enc=None) -> torch.Tensor:
        # Input shape: [B, L, C] where B = batch size, L = sequence length, C = feature count
        
        # 1. Project the input features to the model's feature space
        x = self.input_proj(x_enc)
        
        # 2. Add time features if available
        if x_mark_enc is not None and hasattr(self, 'time_embed'):
            x = x + self.time_embed(x_mark_enc)  # Add time embeddings to features
        
        # 3. Apply CNN to extract temporal features
        x = self.conv1d(x.permute(0, 2, 1))  # Change shape for CNN input [B, C, L]
        x = self.max_pool(x)  # Apply max pooling to downsample
        
        # 4. Pass through LSTM to capture long-term dependencies
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))  # Revert back to [B, L, C] for LSTM input
        
        # 5. Get predictions from the output projection
        pred = self.predictor(lstm_out[:, -self.config['pred_len']:, :])  # Predict based on the last `pred_len` steps
        return pred
