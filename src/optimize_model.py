import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

class HybridNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.instance_norm = nn.InstanceNorm1d(d_model)
    
    def forward(self, x):
        return 0.5*(self.layer_norm(x) + self.instance_norm(x.permute(0,2,1)).permute(0,2,1))

class DilatedResidualCNN(nn.Module):
    def __init__(self, d_model, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model//len(dilations), 3, 
                         padding=d, dilation=d),
                nn.ELU(alpha=0.1),
                HybridNorm(d_model//len(dilations))
            ) for d in dilations
        ])
        
    def forward(self, x):
        # x: [B, T, D]
        x = x.permute(0, 2, 1)
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        return out.permute(0, 2, 1) + x.permute(0, 2, 1)  # Residual

class TemporalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        # Gated residual
        return x + self.gate(x) * attn_out

class OptimizedLSTMCNNAttention(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        cfg = config['model']
        self.d_model = cfg['d_model']
        
        # Input processing
        self.input_net = nn.Sequential(
            nn.Linear(cfg['enc_in'], self.d_model),
            HybridNorm(self.d_model),
            nn.ELU(alpha=0.1),
            nn.Dropout(0.2)
        )
        
        # CNN Branch
        self.cnn = DilatedResidualCNN(self.d_model)
        
        # LSTM Branch
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.d_model//2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention
        self.attention = TemporalAttention(self.d_model, cfg.get('n_heads', 4))
        
        # Output
        self.output_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model*2),
            nn.SiLU(),
            nn.Linear(self.d_model*2, cfg['pred_len'] * cfg.get('output_dim', 1))
        )

    def forward(self, x, time_features=None):
        B, T, _ = x.shape
        
        # Feature projection
        x = self.input_net(x)
        
        # Parallel branches
        cnn_out = self.cnn(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, :, :self.d_model//2] + lstm_out[:, :, self.d_model//2:]  # Merge bidirectional
        
        # Fusion
        fused = cnn_out + lstm_out
        
        # Causal attention
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out = self.attention(fused, mask)
        
        # Output (giữ nguyên format)
        out = self.output_net(attn_out.mean(dim=1))
        return out.view(B, -1, cfg.get('output_dim', 1))